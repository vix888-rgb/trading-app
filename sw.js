// Service worker: app shell cache-first, data.json network-first (falls back to last cached snapshot offline).
const CACHE = "signal-desk-v7";
const SHELL = ["./", "./index.html", "./manifest.json", "./vendor/lightweight-charts.js", "./icons/icon-192.png", "./icons/icon-512.png", "./icons/apple-touch-icon.png", "./data.json"];
self.addEventListener("install", (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))); self.clients.claim(); });
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.protocol === "ws:" || url.protocol === "wss:") return;
  if (/(^|\.)binance\.vision$/i.test(url.hostname)) return; // never cache Binance REST or streams
  if (url.hostname === "ntfy.sh" || url.hostname.endsWith(".ntfy.sh")) return; // never cache ntfy price or alert streams
  if (url.origin !== location.origin) return;
  if (url.pathname.endsWith("data.json")) {
    e.respondWith(fetch(e.request, { cache: "no-store" }).then((r) => { const cp = r.clone(); caches.open(CACHE).then((c) => c.put("./data.json", cp)); return r; }).catch(() => caches.match("./data.json")));
    return;
  }
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
