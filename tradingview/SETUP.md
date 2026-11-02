# Vantage price feed

The site is static. One TradingView alert posts all six Vantage closes to the public price topic every 5 minutes. The scan and the page read that topic. Do not point the price alert at the phone-alert topic `https://ntfy.sh/octa-signals-a6368a35`.

Webhook URL:

`https://ntfy.sh/octa-prices-8223d362`

Delete the six old per-symbol alerts if they are still there. They posted one symbol per minute, and ntfy.sh stopped accepting TradingView's shared webhook addresses.

Use `vantage-price-feed.pine` on a single 5-minute chart. The script requests these symbols on the chart timeframe, so the chart must be 5 minutes:

1. `VANTAGE:XAUUSD`
2. `VANTAGE:BTCUSD`
3. `VANTAGE:EURUSD`
4. `VANTAGE:USDJPY`
5. `VANTAGE:USDCHF`
6. `VANTAGE:EURCHF`

Any chart symbol works. `VANTAGE:XAUUSD` is a good choice. The retired one-symbol script is `vantage-price-feed-single.pine`. Do not add that one.

## Steps

1. Delete the six old price alerts.
2. Open a **5-minute** chart. `VANTAGE:XAUUSD` is fine.
3. Open the Pine Editor, paste `vantage-price-feed.pine`, and add the script to the chart. The overlay name is **Vantage price feed**.
4. Create one alert.
5. Condition: **Vantage price feed** → **Any alert() function call**.
6. Expiration: **Open-ended**.
7. Open the **Notifications** tab. Enable the webhook and set the URL to `https://ntfy.sh/octa-prices-8223d362`.
8. Leave the message box as the default. The script sends the JSON itself with `alert()`.
9. Save the alert.

There should be one price alert. It posts once per 5-minute bar close.

The message looks like:

```json
{"q":{"XAUUSD":3910.47,"BTCUSD":97000.12,"EURUSD":1.08521,"USDJPY":151.234,"USDCHF":0.88123,"EURCHF":0.94567},"tf":"5","t":1730000000000}
```

A symbol with no close is JSON `null` and is skipped. The scan and the page still accept the old one-symbol messages. The page tags a quote Vantage for 12 minutes. The scan shifts levels when the newest quote is at most 15 minutes old.
