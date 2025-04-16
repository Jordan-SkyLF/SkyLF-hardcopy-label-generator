# Hardcopy Label Generator

A desktop tool that generates printable PDF address labels for customers with the "Hardcopy Invoice" label in Splynx.
Integrates USPS API for address validation and caching, includes automatic return-label filler, and outputs polished labels with your business logo.

## ✨ Features

- ✅ Fetches customer billing addresses from Splynx API
- ✅ Validates addresses using USPS API with caching
- ✅ Fills blank spots with branded return labels
- ✅ Outputs printer-ready PDF (Avery-style 3x10)
- ✅ Auto-opens the PDF after generation
- ✅ Supports `.env` configuration for secure credentials
- ✅ Bundled as a standalone `.exe` for Windows

## 🔧 Requirements

- Python 3.8+
- `reportlab`, `python-dotenv`, `requests`

(Or use the provided `.exe` — no install needed)

## 📦 Environment Setup (`.env`)

Create a `.env` file in the root folder:

```
# Logging & timing
DEBUG_MODE=False # off by default
TIME_THRESHOLD=20 # takes around 20 seconds with no cached results, 5 ish with them

# Discord webhook for notifications
DISCORD_WEBHOOK_URL=INSERT

# Splynx API credentials and endpoints
SPLYNX_API_KEY=INSERT
SPLYNX_API_SECRET=INSERT
SPLYNX_API_URL=INSERT
TARGET_LABEL=INSERT
CUSTOMERS_ENDPOINT=INSERT

# USPS OAuth credentials & endpoints
USPS_CONSUMER_KEY=INSERT
USPS_CONSUMER_SECRET=INSERT
USPS_TOKEN_URL=INSERT
USPS_ADDRESS_VALIDATION_URL=INSERT
USPS_CITY_STATE_URL=INSERT

# Persistent cache file names
USPS_CITY_STATE_CACHE_FILE=INSERT
USPS_ADDRESS_VALIDATION_CACHE_FILE=INSERT

```


## 🖨️ How to Use

1. Double-click the `.exe` (or run the script)
2. Script fetches customers with the “Hardcopy Invoice” label
3. Generates a PDF file called `labels.pdf`
4. File opens automatically in your default PDF viewer

## 📁 File Structure

```
project/
├── Hardcopy-label-generator.py
├── .env
├── resources/
│   └── logo.png
├── cache/
│   ├── usps_city_state_cache.json
│   └── usps_address_validation_cache.json
├── logs/
│   └── label_generator.log
```

## 👤 Author

Created by Jordan at SkyLinkFiber.net  
🛠 Built with AI passion and caffeine in The Dalles, Oregon

## 📃 License

This project is proprietary and not open-source.
