import os
import sys
import subprocess
import requests
import base64
import logging
import time
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

def resource_path(rel_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller one‐file exe.
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller stores unpacked files in _MEIPASS
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, rel_path)

# Load environment variables from .env file
# right after your resource_path() helper
dotenv_path = resource_path('.env')
load_dotenv(dotenv_path)


# --- Configuration (from .env) ---
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
TIME_THRESHOLD = int(os.getenv('TIME_THRESHOLD', '20'))
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

SPLYNX_API_KEY = os.getenv('SPLYNX_API_KEY')
SPLYNX_API_SECRET = os.getenv('SPLYNX_API_SECRET')
SPLYNX_API_URL = os.getenv('SPLYNX_API_URL')
TARGET_LABEL = os.getenv('TARGET_LABEL', 'Hardcopy Invoice')
CUSTOMERS_ENDPOINT = os.getenv(
    'CUSTOMERS_ENDPOINT',
    f"{SPLYNX_API_URL}/api/2.0/admin/customers/customer"
)

USPS_CONSUMER_KEY = os.getenv('USPS_CONSUMER_KEY')
USPS_CONSUMER_SECRET = os.getenv('USPS_CONSUMER_SECRET')
USPS_TOKEN_URL = os.getenv('USPS_TOKEN_URL')
USPS_ADDRESS_VALIDATION_URL = os.getenv('USPS_ADDRESS_VALIDATION_URL')
USPS_CITY_STATE_URL = os.getenv('USPS_CITY_STATE_URL')

# Directories
LOGS_DIR = "logs"
CACHE_DIR = "cache"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# JSON cache file paths
CITY_STATE_CACHE_PATH = os.path.join(CACHE_DIR, 'usps_city_state_cache.json')
ADDRESS_VALIDATION_CACHE_PATH = os.path.join(CACHE_DIR, 'usps_address_validation_cache.json')

# Return‑label styling
RETURN_LABEL_COLOR = colors.HexColor("#01ADEF")
FAVICON_PATH     = resource_path(os.path.join("resources", "logo.png"))
FAVICON_WIDTH    = 0.3 * inch
FAVICON_HEIGHT   = 0.3 * inch

def load_json_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json_cache(path: str, data: Dict[str, Any]) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Failed saving cache {path}: {e}")

# Load caches into memory
USPS_CITY_STATE_CACHE: Dict[str, Dict[str, str]] = load_json_cache(CITY_STATE_CACHE_PATH)
USPS_ADDRESS_VALIDATION_CACHE: Dict[str, Any] = load_json_cache(ADDRESS_VALIDATION_CACHE_PATH)

# --- Logging setup ---
LOG_FILE = os.path.join(LOGS_DIR, "label_generator.log")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="a")
file_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
logger.addHandler(console_handler)

# Splynx API headers
GLOBAL_HEADERS = {
    'Authorization': 'Basic ' + base64.b64encode(f'{SPLYNX_API_KEY}:{SPLYNX_API_SECRET}'.encode()).decode(),
    'Content-Type': 'application/json'
}
logger.debug(f"Cached Splynx API Headers: {GLOBAL_HEADERS}")

def send_discord_notification(message: str) -> None:
    payload = {"content": message}
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("Discord notification sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")

def get_usps_access_token() -> str:
    payload = {
        "grant_type": "client_credentials",
        "client_id": USPS_CONSUMER_KEY,
        "client_secret": USPS_CONSUMER_SECRET
    }
    headers = {"Content-Type": "application/json"}
    try:
        logger.debug("Requesting USPS access token.")
        r = requests.post(USPS_TOKEN_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        token = r.json().get("access_token")
        if not token:
            raise ValueError("Missing access token")
        return token
    except Exception as e:
        logger.error(f"Error obtaining USPS access token: {e}")
        raise

def lookup_city_state_by_zip(zip_code: str) -> Dict[str, str]:
    if zip_code in USPS_CITY_STATE_CACHE:
        logger.debug(f"Using cached city/state for ZIP {zip_code}.")
        return USPS_CITY_STATE_CACHE[zip_code]
    try:
        token = get_usps_access_token()
    except Exception:
        return {}
    params = {"ZIPCode": zip_code}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        logger.debug(f"Looking up city/state for ZIP {zip_code}.")
        r = requests.get(USPS_CITY_STATE_URL, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        city = data.get("city", "").strip()
        state = data.get("state", "").strip()
        if not city or not state:
            raise ValueError("Incomplete city/state data")
        USPS_CITY_STATE_CACHE[zip_code] = {"city": city, "state": state}
        save_json_cache(CITY_STATE_CACHE_PATH, USPS_CITY_STATE_CACHE)
        return {"city": city, "state": state}
    except Exception as e:
        logger.error(f"Error during USPS city/state lookup for ZIP {zip_code}: {e}")
        return {}

def validate_address_usps(street: str, city: str, state: str, zip_code: str) -> dict:
    key = f"{street}|{city}|{state}|{zip_code}"
    if key in USPS_ADDRESS_VALIDATION_CACHE:
        logger.debug(f"Using cached address validation for {key}.")
        return USPS_ADDRESS_VALIDATION_CACHE[key]
    try:
        token = get_usps_access_token()
    except Exception as e:
        return {"valid": False, "error": str(e)}
    params = {
        "streetAddress": street,
        "secondaryAddress": "",
        "city": city,
        "state": state,
        "ZIPCode": zip_code
    }
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        logger.debug("Validating address via USPS API.")
        r = requests.get(USPS_ADDRESS_VALIDATION_URL, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        addr = r.json().get("address")
        if not addr:
            raise ValueError("No address in response")
        validated = {
            "valid": True,
            "address": {
                "street": addr.get("streetAddress", "").strip(),
                "city": addr.get("city", "").strip(),
                "state": addr.get("state", "").strip(),
                "zip_code": addr.get("ZIPCode", "").strip()
            }
        }
    except Exception as e:
        validated = {"valid": False, "error": str(e)}
        logger.error(f"Error validating address: {e}")
    USPS_ADDRESS_VALIDATION_CACHE[key] = validated
    save_json_cache(ADDRESS_VALIDATION_CACHE_PATH, USPS_ADDRESS_VALIDATION_CACHE)
    return validated

def get_customers() -> List[Dict[str, Any]]:
    logger.debug(f"Fetching customers from {CUSTOMERS_ENDPOINT}")
    try:
        r = requests.get(CUSTOMERS_ENDPOINT, headers=GLOBAL_HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Error fetching customers: {e}")
        return []

def get_billing_info(cid: int) -> Dict[str, Any]:
    url = f"{SPLYNX_API_URL}/api/2.0/admin/customers/customer-billing/{cid}"
    logger.debug(f"Fetching billing info for CID {cid}")
    try:
        r = requests.get(url, headers=GLOBAL_HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Error fetching billing info for {cid}: {e}")
        return {}

def get_billing_info_parallel(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(get_billing_info, i): i for i in ids}
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                results[cid] = fut.result()
            except Exception as e:
                logger.error(f"Error in parallel billing for {cid}: {e}")
                results[cid] = {}
    return results

def gather_label_data(customers: List[Dict[str, Any]], label: str) -> List[Dict[str, str]]:
    logger.info(f"Gathering label data for '{label}'")
    ids, cmap = [], {}
    for c in customers:
        if any(l.get("label")==label for l in c.get("customer_labels",[])):
            ids.append(c["id"])
            cmap[c["id"]] = c
    logger.info(f"Found {len(ids)} labeled customers")
    billing = get_billing_info_parallel(ids)
    unique_zips = {
        (billing.get(cid,{}).get("billing_zip_code") or cmap[cid].get("zip_code","")).strip()
        for cid in ids
    } - {""}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(lookup_city_state_by_zip,z):z for z in unique_zips}
        for fut in as_completed(futs):
            if not fut.result():
                logger.warning(f"Failed ZIP lookup {futs[fut]}")
    tasks={}
    for cid in ids:
        cust, bill = cmap[cid], billing.get(cid,{})
        if bill.get("billing_person"):
            name=bill.get("billing_person",cust.get("name",""))
            street=bill.get("billing_street_1",cust.get("street_1",""))
            zipc=(bill.get("billing_zip_code") or cust.get("zip_code","")).strip()
        else:
            name=cust.get("name","")
            street=cust.get("street_1","")
            zipc=cust.get("zip_code","").strip()
        loc=USPS_CITY_STATE_CACHE.get(zipc)
        if not loc:
            logger.warning(f"No city/state for CID {cid}, ZIP {zipc}")
            continue
        tasks[cid]={"name":name,"street":street,"city":loc["city"],"state":loc["state"],"zip_code":zipc}
    validated, total, processed = {}, len(tasks), 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(validate_address_usps, t["street"],t["city"],t["state"],t["zip_code"]):cid
              for cid,t in tasks.items()}
        for fut in as_completed(futs):
            processed+=1
            cid=futs[fut]
            res, nm = fut.result(), tasks[cid]["name"].strip()
            if res.get("valid"):
                validated[cid]=res["address"]
                logger.info(f"Validated {processed}/{total} — {nm}")
            else:
                logger.warning(f"Validation failed {processed}/{total} — {nm}: {res.get('error')}")
    labels_out=[{"name":tasks[c]["name"].strip(),
                 "street":addr["street"],
                 "city":addr["city"],
                 "state":addr["state"],
                 "zip_code":addr["zip_code"]}
                for c,addr in validated.items()]
    logger.info(f"Prepared {len(labels_out)} labels")
    return labels_out

def generate_pdf(labels: List[Dict[str, str]], filename: str = "labels.pdf") -> None:
    output_path = os.path.join(os.getcwd(), filename)
    logger.info(f"Generating PDF '{output_path}' with {len(labels)} labels")
    c = canvas.Canvas(output_path, pagesize=letter)
    w, h = letter
    margin = 36
    cols, rows = 3, 10
    page_size = cols * rows
    cell_h = (h - 2 * margin) / rows
    col_w = (w - 2 * margin) / cols
    x_offs = [margin, margin + col_w + 20, margin + 2 * col_w + 35]
    bold_font, regular_font = "Helvetica-Bold", "Helvetica"
    font_size, line_spacing = 10, 12
    total_labels = len(labels)
    rem = total_labels % page_size
    blanks = (page_size - rem) if rem else 0
    total_items = total_labels + blanks

    # Draw customer labels
    c.setFillColor(colors.black)
    for i, lbl in enumerate(labels):
        pos = i % page_size
        col, row = pos % cols, pos // cols
        x = x_offs[col]
        y = h - margin - row * cell_h - 15

        c.setFont(bold_font, font_size)
        c.drawString(x, y, lbl["name"])
        y -= line_spacing
        c.setFont(regular_font, font_size)
        c.drawString(x, y, lbl["street"])
        y -= line_spacing
        c.drawString(x, y, f"{lbl['city']}, {lbl['state']} {lbl['zip_code']}")

        if (i + 1) % page_size == 0 and (i + 1) < total_items:
            c.showPage()
            c.setFillColor(colors.black)

    # Draw return labels (blank spots) with inline logo aligned at top edge
    if blanks:
        for j in range(blanks):
            idx = total_labels + j
            pos = idx % page_size
            col, row = pos % cols, pos // cols
            x = x_offs[col]
            y = h - margin - row * cell_h - 15

            # Draw logo so top edge aligns with top text line
            try:
                img_y = y - FAVICON_HEIGHT
                c.drawImage(
                    FAVICON_PATH,
                    x,
                    img_y,
                    width=FAVICON_WIDTH,
                    height=FAVICON_HEIGHT,
                    mask='auto'
                )
            except Exception as e:
                logger.warning(f"Couldn’t draw logo: {e}")

            text_x = x + FAVICON_WIDTH + 4
            c.setFillColor(RETURN_LABEL_COLOR)
            c.setFont(bold_font, font_size)
            c.drawString(text_x, y, "SkyLinkFiber.net")
            y -= line_spacing
            c.setFont(regular_font, font_size)
            c.drawString(text_x, y, "622 E 3rd St")
            y -= line_spacing
            c.drawString(text_x, y, "The Dalles, OR 97058")

            if (idx + 1) % page_size == 0 and (idx + 1) < total_items:
                c.showPage()
                c.setFillColor(colors.black)

    c.save()
    logger.info(f"PDF saved as '{output_path}'")
    # auto‑open the PDF in the system’s default viewer
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', output_path], check=False)
        else:  # Linux and other Unixes
            subprocess.run(['xdg-open', output_path], check=False)
    except Exception as e:
        logger.warning(f"Could not auto-open PDF: {e}")


def main() -> None:
    start = time.time()
    logger.info("Script started.")
    try:
        custs = get_customers()
        logger.info(f"Total customers: {len(custs)}")
        lbls = gather_label_data(custs, TARGET_LABEL)
        lbls.sort(key=lambda x: x["name"].split()[0].lower())
        logger.info(f"Labels to print: {len(lbls)}")
        generate_pdf(lbls)
    except Exception as e:
        err = f"Script failed: {e}"
        logger.error(err)
        send_discord_notification(err)
    finally:
        total = time.time() - start
        logger.info(f"Completed in {total:.2f}s")
        if total > TIME_THRESHOLD:
            send_discord_notification(f"WARNING: took {total:.2f}s")

if __name__ == "__main__":
    main()