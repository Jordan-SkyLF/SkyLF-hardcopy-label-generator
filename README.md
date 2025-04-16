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
SPLYNX_API_KEY=your_key_here
SPLYNX_API_SECRET=your_secret_here
SPLYNX_API_URL=https://splynx.example.com
USPS_CONSUMER_KEY=your_usps_key
USPS_CONSUMER_SECRET=your_usps_secret
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DEBUG_MODE=True
TIME_THRESHOLD=20
```

> ⚠️ **Do not commit this file** — it is ignored via `.gitignore`

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
