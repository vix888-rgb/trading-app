# Vantage price feed

The site is static. TradingView sends each 1-minute Vantage close to the public price topic. The scan and the page read that topic. Do not point price alerts at the phone-alert topic `https://ntfy.sh/octa-signals-a6368a35`.

Webhook URL for every price alert:

`https://ntfy.sh/octa-prices-8223d362`

Use the same script, `vantage-price-feed.pine`, on all six charts. It fills `sym` from the chart ticker and strips suffixes such as `+` and `.r`.

Symbols (one alert each):

1. `VANTAGE:XAUUSD`
2. `VANTAGE:BTCUSD`
3. `VANTAGE:EURUSD`
4. `VANTAGE:USDJPY`
5. `VANTAGE:USDCHF`
6. `VANTAGE:EURCHF`

## Steps (repeat for each symbol)

1. Open a **1-minute** chart of the Vantage symbol.
2. Open the Pine Editor, paste `vantage-price-feed.pine`, and add the script to the chart. The overlay name is **Vantage price feed**.
3. Create an alert.
4. Condition: **Vantage price feed** → **Any alert() function call**.
5. Expiration: **Open-ended**.
6. Open the **Notifications** tab. Enable the webhook and set the URL to `https://ntfy.sh/octa-prices-8223d362`.
7. Leave the message box as the default. The script sends the JSON itself with `alert()`.
8. Save the alert.

After all six symbols, there should be six alerts, all posting to that same webhook.
