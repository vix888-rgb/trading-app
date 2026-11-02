#!/bin/bash
# Repeat the signal scan every SCAN_INTERVAL_SEC for about SCAN_BUDGET_SEC.
#
# GitHub often delays or drops a */10 schedule. One job holds the signal-scan
# concurrency slot and publishes a snapshot per iteration, then exits before
# the 6-hour runner limit so the queued schedule (or a handover dispatch) can
# take the next slot. cancel-in-progress stays false: a new run must not kill
# this one during commit or push.
#
# Latch, frozen weights, and alerts_sent.json stay in pipeline/state on this
# runner and are committed with each snapshot, so the next iteration (and the
# next job) keep closed-bar latching and one-alert-per-bar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INTERVAL="${SCAN_INTERVAL_SEC:-600}"
BUDGET="${SCAN_BUDGET_SEC:-21000}" # 5h50m
MAX_ITER="${SCAN_MAX_ITER_SEC:-540}"
# GitHub-hosted jobs die at 6h. Stop early enough to push and hand over.
HARD_STOP_AFTER="${SCAN_HARD_STOP_SEC:-21420}" # 5h57m

START=$(date +%s)
DEADLINE=$((START + BUDGET))
HARD_STOP=$((START + HARD_STOP_AFTER))
SCANS=0
ALERTS_FAILED=0
LAST_PAGES="skip"
HANDED=0
export GIT_TERMINAL_PROMPT=0

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

snapshot_gen() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("gen",""))' data.json 2>/dev/null || echo "?"
}

align_delay() {
  python3 - "$INTERVAL" data.json <<'PY'
import json, sys
from datetime import datetime
from zoneinfo import ZoneInfo
interval = int(sys.argv[1])
try:
    gen = json.load(open(sys.argv[2])).get("gen") or ""
    prev = datetime.strptime(gen, "%a %d %b %Y, %H:%M").replace(tzinfo=ZoneInfo("Europe/Belgrade"))
    delta = interval - (datetime.now(ZoneInfo("Europe/Belgrade")) - prev).total_seconds()
except Exception:
    print(0)
    raise SystemExit(0)
print(int(delta) if 0 < delta < interval else 0)
PY
}

note_summary() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "- $(date -u +%H:%M:%S) UTC gen=$(snapshot_gen) sha=$(git rev-parse --short HEAD 2>/dev/null || echo none)" >> "$GITHUB_STEP_SUMMARY"
  fi
}

dispatch_workflow() {
  local name="$1" attempt
  for attempt in 1 2 3; do
    if gh workflow run "$name" --ref main; then
      log "dispatched ${name}"
      return 0
    fi
    log "dispatch ${name} failed (attempt ${attempt})"
    sleep $((attempt * 2))
  done
  return 1
}

dispatch_pages() {
  if dispatch_workflow pages.yml; then
    LAST_PAGES="ok"
    return 0
  fi
  LAST_PAGES="fail"
  log "WARNING: Pages deploy was not dispatched"
  return 1
}

rebase_onto_main() {
  git rebase --abort >/dev/null 2>&1 || true
  if git merge-base --is-ancestor HEAD origin/main; then
    git reset --hard origin/main
    return 0
  fi
  if git rebase --strategy-option=theirs origin/main; then
    return 0
  fi
  if [[ -d "$(git rev-parse --git-path rebase-merge)" || -d "$(git rev-parse --git-path rebase-apply)" ]]; then
    git checkout --theirs -- data.json pipeline/state 2>/dev/null || true
    git add data.json pipeline/state
    if GIT_EDITOR=true git rebase --continue; then
      return 0
    fi
    git rebase --abort >/dev/null 2>&1 || true
  fi
  return 1
}

# Pick up commits that landed on main. No-op when this checkout is already main.
# A dirty tree is left alone so a failed publish is not discarded.
integrate_main() {
  # Untracked files (scratch snapshots) must not block a fast-forward.
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    log "worktree dirty; keeping local snapshot"
    return 0
  fi
  git fetch origin main
  rebase_onto_main
  if [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]]; then
    return 0
  fi
  git push origin HEAD:main
  log "flushed unpublished $(git rev-parse --short HEAD)"
  dispatch_pages || true
}

restore_snapshot() {
  git rebase --abort >/dev/null 2>&1 || true
  git checkout -- data.json pipeline/state 2>/dev/null || true
}

publish() {
  python3 -c 'import json; json.load(open("data.json"))'
  git add data.json pipeline/state
  if git diff --cached --quiet; then
    log "nothing to commit"
    return 0
  fi
  git commit -m "Auto scan"
  local attempt
  for attempt in 1 2 3 4 5; do
    git fetch origin main
    if rebase_onto_main && git push origin HEAD:main; then
      log "pushed $(git rev-parse --short HEAD) gen=$(snapshot_gen)"
      note_summary
      dispatch_pages || true
      return 0
    fi
    git rebase --abort >/dev/null 2>&1 || true
    log "push rejected (attempt ${attempt})"
    sleep $((attempt * 2))
  done
  return 1
}

one_scan() {
  integrate_main || log "could not integrate origin/main"
  cp -f data.json /tmp/data.prev.json
  if ! timeout -k 20s "$MAX_ITER" bash pipeline/run.sh; then
    log "pipeline failed; restoring the last published snapshot"
    restore_snapshot
    return 1
  fi
  set +e
  python3 pipeline/alerts.py --prev /tmp/data.prev.json --new data.json --snap pipeline/snap3.json --send
  local alert_rc=$?
  set -e
  if (( alert_rc != 0 )); then
    ALERTS_FAILED=1
    log "alert step failed (rc=${alert_rc}); committing latch and sent-state anyway"
  fi
  publish
}

stop_now() {
  local now="$1"
  if (( now >= DEADLINE || now >= HARD_STOP )); then
    return 0
  fi
  if (( now + MAX_ITER >= HARD_STOP )); then
    return 0
  fi
  return 1
}

handover() {
  if (( HANDED == 1 )); then
    return 0
  fi
  HANDED=1
  local elapsed=$(( $(date +%s) - START ))
  local min_hold=$((BUDGET - INTERVAL))
  if (( elapsed < min_hold )); then
    log "no handover after ${elapsed}s; the */10 schedule will retry"
    return 0
  fi
  log "handover after ${elapsed}s"
  if ! dispatch_workflow scan.yml; then
    log "WARNING: handover dispatch failed; a queued schedule run may still take over"
  fi
}

# Executed only when this file is the process, so tests can source the helpers.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

delay="$(align_delay || echo 0)"
if [[ "$delay" =~ ^[0-9]+$ ]] && (( delay > 0 )); then
  remaining=$((DEADLINE - $(date +%s)))
  if (( delay < remaining )); then
    log "sleeping ${delay}s so this scan stays about ${INTERVAL}s after the previous one"
    sleep "$delay"
  else
    log "skipping align sleep (${delay}s); budget ends sooner"
  fi
fi

log "loop start interval=${INTERVAL}s budget=${BUDGET}s gen=$(snapshot_gen)"

while true; do
  now=$(date +%s)
  if stop_now "$now"; then
    log "stopping loop elapsed=$((now - START))s"
    break
  fi
  iter_start=$now
  log "scan start"
  if one_scan; then
    SCANS=$((SCANS + 1))
    log "scan done #${SCANS} gen=$(snapshot_gen)"
  else
    log "scan iteration failed"
  fi
  now=$(date +%s)
  next=$((iter_start + INTERVAL))
  if stop_now "$next"; then
    log "next scan would pass the budget"
    break
  fi
  sleep_for=$((next - now))
  if (( sleep_for > 0 )); then
    log "sleeping ${sleep_for}s"
    sleep "$sleep_for"
  fi
done

handover
log "finished scans=${SCANS} alerts_failed=${ALERTS_FAILED} pages=${LAST_PAGES}"
if (( SCANS == 0 || ALERTS_FAILED != 0 )) || [[ "$LAST_PAGES" == "fail" ]]; then
  exit 1
fi
