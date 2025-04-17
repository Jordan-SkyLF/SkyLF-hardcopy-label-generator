"""
Hardcopy Label Generator
------------------------
Generates printable PDF address labels for customers with the "Hardcopy Invoice" label in Splynx.
Integrates with USPS API for address validation with caching.

Author: Jordan at SkyLinkFiber.net (Improved by Claude)
"""

import os
import sys
import subprocess
import base64
import logging
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
import tempfile
from functools import lru_cache

import aiohttp
import backoff
import tenacity
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# --------------------------
# Data Models
# --------------------------

@dataclass
class AddressInfo:
    """Model representing a validated mailing address"""
    street: str
    city: str
    state: str
    zip_code: str
    secondary_address: str = ""
    
    def as_cache_key(self) -> str:
        """Generate a consistent cache key from address components"""
        return f"{self.street}|{self.city}|{self.state}|{self.zip_code}"

@dataclass
class CustomerInfo:
    """Model representing a customer with shipping information"""
    id: int
    name: str
    billing_info: Dict[str, Any] = field(default_factory=dict)
    address: Optional[AddressInfo] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LabelEntry:
    """Model for a single label entry in the PDF"""
    name: str
    street: str
    city: str
    state: str
    zip_code: str

    @classmethod
    def from_customer(cls, customer: CustomerInfo) -> Optional['LabelEntry']:
        """Create a label entry from a customer if they have a valid address"""
        if not customer.address:
            return None
        
        return cls(
            name=customer.name.strip(),
            street=customer.address.street,
            city=customer.address.city,
            state=customer.address.state,
            zip_code=customer.address.zip_code
        )


# --------------------------
# Configuration
# --------------------------

class AppConfig:
    """Application configuration management"""
    def __init__(self):
        self._load_environment()
        self._setup_paths()
        
    def _load_environment(self) -> None:
        """Load environment variables from .env file"""
        dotenv_path = os.path.join(
            os.path.dirname(
                sys.executable if getattr(sys, 'frozen', False) else __file__
            ), 
            '.env'
        )
        load_dotenv(dotenv_path)
        
        # Debug settings
        self.debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.time_threshold = int(os.getenv('TIME_THRESHOLD', '20'))
        
        # Discord webhook for notifications
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        # Splynx API settings
        self.splynx_api_key = os.getenv('SPLYNX_API_KEY')
        self.splynx_api_secret = os.getenv('SPLYNX_API_SECRET')
        self.splynx_api_url = os.getenv('SPLYNX_API_URL')
        self.target_label = os.getenv('TARGET_LABEL', 'Hardcopy Invoice')
        self.customers_endpoint = os.getenv(
            'CUSTOMERS_ENDPOINT',
            f"{self.splynx_api_url}/api/2.0/admin/customers/customer"
        ) if self.splynx_api_url else None
        
        # USPS API settings
        self.usps_consumer_key = os.getenv('USPS_CONSUMER_KEY')
        self.usps_consumer_secret = os.getenv('USPS_CONSUMER_SECRET')
        self.usps_token_url = os.getenv('USPS_TOKEN_URL')
        self.usps_address_validation_url = os.getenv('USPS_ADDRESS_VALIDATION_URL')
        self.usps_city_state_url = os.getenv('USPS_CITY_STATE_URL')
        
        # Cache file settings
        self.usps_city_state_cache_file = os.getenv('USPS_CITY_STATE_CACHE_FILE', 'usps_city_state_cache.json')
        self.usps_address_validation_cache_file = os.getenv('USPS_ADDRESS_VALIDATION_CACHE_FILE', 'usps_address_validation_cache.json')
        
        # PDF settings
        self.return_label_color = colors.HexColor("#01ADEF") 
        self.output_filename = "labels.pdf"
        
    def _setup_paths(self) -> None:
        """Set up directory paths"""
        # Directories for logs and cache
        self.logs_dir = Path("logs")
        self.cache_dir = Path("cache")
        self.logs_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # File paths
        self.log_file = self.logs_dir / "label_generator.log"
        self.city_state_cache_path = self.cache_dir / self.usps_city_state_cache_file
        self.address_validation_cache_path = self.cache_dir / self.usps_address_validation_cache_file
        self.favicon_path = self._resolve_resource_path("resources/logo.png")
        
    @staticmethod
    def _resolve_resource_path(rel_path: str) -> Path:
        """Get absolute path to resource, works for dev and for PyInstaller"""
        if getattr(sys, 'frozen', False):
            # PyInstaller stores unpacked files in _MEIPASS
            base = Path(sys._MEIPASS)
        else:
            base = Path.cwd()
        return base / rel_path


# --------------------------
# Cache Management
# --------------------------

class PersistentCache:
    """Manager for persistent JSON cache files"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data: Dict[str, Any] = {}
        self.load()
        
    def load(self) -> None:
        """Load data from cache file"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
                logger.debug(f"Loaded {len(self.data)} entries from {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to load cache from {self.file_path}: {e}")
                self.data = {}
        else:
            logger.debug(f"Cache file {self.file_path} not found, starting with empty cache")
            self.data = {}
            
    def save(self) -> None:
        """Save current data to cache file"""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f)
            logger.debug(f"Saved {len(self.data)} entries to {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.file_path}: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in cache and save"""
        self.data[key] = value
        self.save()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.data


# --------------------------
# API Clients
# --------------------------

class SplynxAPIClient:
    """Client for interacting with Splynx API"""
    
    def __init__(self, config: AppConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.headers = {
            'Authorization': 'Basic ' + base64.b64encode(
                f'{config.splynx_api_key}:{config.splynx_api_secret}'.encode()
            ).decode(),
            'Content-Type': 'application/json'
        }
        
    async def get_customers(self) -> List[Dict[str, Any]]:
        """Fetch all customers from Splynx API"""
        if not self.config.customers_endpoint:
            logger.error("Customers endpoint not configured")
            return []
            
        try:
            logger.debug(f"Fetching customers from {self.config.customers_endpoint}")
            async with self.session.get(
                self.config.customers_endpoint, 
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching customers: {e}")
            return []
    
    async def get_billing_info(self, customer_id: int) -> Dict[str, Any]:
        """Fetch billing info for a specific customer"""
        url = f"{self.config.splynx_api_url}/api/2.0/admin/customers/customer-billing/{customer_id}"
        try:
            logger.debug(f"Fetching billing info for customer ID {customer_id}")
            async with self.session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching billing info for customer {customer_id}: {e}")
            return {}
            
    async def get_billing_info_batch(self, customer_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch billing info for multiple customers in parallel"""
        tasks = [self.get_billing_info(cid) for cid in customer_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        billing_info = {}
        for i, result in enumerate(results):
            cid = customer_ids[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to get billing info for {cid}: {result}")
                billing_info[cid] = {}
            else:
                billing_info[cid] = result
                
        return billing_info


class USPSClient:
    """Client for interacting with USPS API"""
    
    def __init__(self, config: AppConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.city_state_cache = PersistentCache(config.city_state_cache_path)
        self.address_validation_cache = PersistentCache(config.address_validation_cache_path)
        self._token = None
        self._token_expiry = 0
        
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying USPS API call after error, attempt {retry_state.attempt_number}"
        )
    )
    async def get_access_token(self) -> str:
        """Get USPS access token, with retry logic"""
        # Check if we have a valid token
        current_time = time.time()
        if self._token and current_time < self._token_expiry:
            return self._token
            
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.config.usps_consumer_key,
            "client_secret": self.config.usps_consumer_secret
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.debug("Requesting USPS access token")
            async with self.session.post(
                self.config.usps_token_url, 
                json=payload, 
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)  # Default to 1 hour if not specified
                
                if not token:
                    raise ValueError("Missing access token in USPS API response")
                    
                self._token = token
                self._token_expiry = current_time + expires_in - 60  # Buffer of 60 seconds
                return token
                
        except Exception as e:
            logger.error(f"Error obtaining USPS access token: {e}")
            raise
    
    async def lookup_city_state_by_zip(self, zip_code: str) -> Dict[str, str]:
        """Look up city and state by ZIP code, using cache when available"""
        # Check cache first
        if zip_code in self.city_state_cache:
            logger.debug(f"Using cached city/state for ZIP {zip_code}")
            return self.city_state_cache.get(zip_code)
            
        try:
            token = await self.get_access_token()
            params = {"ZIPCode": zip_code}
            headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
            
            logger.debug(f"Looking up city/state for ZIP {zip_code}")
            async with self.session.get(
                self.config.usps_city_state_url, 
                params=params, 
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                city = data.get("city", "").strip()
                state = data.get("state", "").strip()
                
                if not city or not state:
                    raise ValueError(f"Incomplete city/state data for ZIP {zip_code}")
                    
                result = {"city": city, "state": state}
                self.city_state_cache.set(zip_code, result)
                return result
                
        except Exception as e:
            logger.error(f"Error during USPS city/state lookup for ZIP {zip_code}: {e}")
            return {}
    
    async def validate_address(self, address: AddressInfo) -> Tuple[bool, Optional[AddressInfo]]:
        """Validate address using USPS API, with caching"""
        cache_key = address.as_cache_key()
        
        # Check cache first
        if cache_key in self.address_validation_cache:
            logger.debug(f"Using cached address validation for {cache_key}")
            cached = self.address_validation_cache.get(cache_key)
            if cached.get("valid"):
                validated_addr = AddressInfo(
                    street=cached["address"]["street"],
                    city=cached["address"]["city"],
                    state=cached["address"]["state"],
                    zip_code=cached["address"]["zip_code"],
                    secondary_address=cached["address"].get("secondary_address", "")
                )
                return True, validated_addr
            return False, None
            
        try:
            token = await self.get_access_token()
            params = {
                "streetAddress": address.street,
                "secondaryAddress": address.secondary_address,
                "city": address.city,
                "state": address.state,
                "ZIPCode": address.zip_code
            }
            headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
            
            logger.debug(f"Validating address via USPS API: {address}")
            async with self.session.get(
                self.config.usps_address_validation_url, 
                params=params, 
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                addr_data = data.get("address")
                
                if not addr_data:
                    raise ValueError("No address in USPS API response")
                    
                validated_addr = AddressInfo(
                    street=addr_data.get("streetAddress", "").strip(),
                    city=addr_data.get("city", "").strip(),
                    state=addr_data.get("state", "").strip(),
                    zip_code=addr_data.get("ZIPCode", "").strip(),
                    secondary_address=addr_data.get("secondaryAddress", "").strip()
                )
                
                # Cache the result
                self.address_validation_cache.set(cache_key, {
                    "valid": True,
                    "address": asdict(validated_addr)
                })
                
                return True, validated_addr
                
        except Exception as e:
            logger.error(f"Error validating address: {e}")
            # Cache the failure
            self.address_validation_cache.set(cache_key, {
                "valid": False,
                "error": str(e)
            })
            return False, None


# --------------------------
# PDF Generation
# --------------------------

class PDFGenerator:
    """Generates printable PDF labels"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # PDF dimensions and styling
        self.page_width, self.page_height = letter
        self.margin = 36
        self.columns, self.rows = 3, 10
        self.page_size = self.columns * self.rows
        self.cell_height = (self.page_height - 2 * self.margin) / self.rows
        self.column_width = (self.page_width - 2 * self.margin) / self.columns
        self.x_offsets = [
            self.margin, 
            self.margin + self.column_width + 20, 
            self.margin + 2 * self.column_width + 35
        ]
        
        # Font configuration
        self.bold_font = "Helvetica-Bold"
        self.regular_font = "Helvetica"
        self.font_size = 10
        self.line_spacing = 12
        
        # Logo dimensions
        self.favicon_width = 0.3 * inch
        self.favicon_height = 0.3 * inch
        
    def generate(self, labels: List[LabelEntry], output_path: Optional[str] = None) -> str:
        """Generate PDF with labels"""
        if not output_path:
            output_path = os.path.join(os.getcwd(), self.config.output_filename)
            
        logger.info(f"Generating PDF '{output_path}' with {len(labels)} labels")
        
        # Create canvas
        c = canvas.Canvas(output_path, pagesize=letter)
        
        # Calculate layout information
        total_labels = len(labels)
        remainder = total_labels % self.page_size
        blanks = (self.page_size - remainder) if remainder else 0
        total_items = total_labels + blanks
        
        # Draw customer labels
        c.setFillColor(colors.black)
        for i, label in enumerate(labels):
            position = i % self.page_size
            col, row = position % self.columns, position // self.columns
            x = self.x_offsets[col]
            y = self.page_height - self.margin - row * self.cell_height - 15
            
            # Draw label content
            c.setFont(self.bold_font, self.font_size)
            c.drawString(x, y, label.name)
            y -= self.line_spacing
            
            c.setFont(self.regular_font, self.font_size)
            c.drawString(x, y, label.street)
            y -= self.line_spacing
            
            c.drawString(x, y, f"{label.city}, {label.state} {label.zip_code}")
            
            # Create new page if needed
            if (i + 1) % self.page_size == 0 and (i + 1) < total_items:
                c.showPage()
                c.setFillColor(colors.black)
        
        # Draw return labels in any blank spots
        if blanks:
            for j in range(blanks):
                idx = total_labels + j
                position = idx % self.page_size
                col, row = position % self.columns, position // self.columns
                x = self.x_offsets[col]
                y = self.page_height - self.margin - row * self.cell_height - 15
                
                # Draw logo
                try:
                    img_y = y - self.favicon_height
                    c.drawImage(
                        str(self.config.favicon_path),
                        x,
                        img_y,
                        width=self.favicon_width,
                        height=self.favicon_height,
                        mask='auto'
                    )
                except Exception as e:
                    logger.warning(f"Couldn't draw logo: {e}")
                
                # Draw return address
                text_x = x + self.favicon_width + 4
                c.setFillColor(self.config.return_label_color)
                
                c.setFont(self.bold_font, self.font_size)
                c.drawString(text_x, y, "SkyLinkFiber.net")
                y -= self.line_spacing
                
                c.setFont(self.regular_font, self.font_size)
                c.drawString(text_x, y, "622 E 3rd St")
                y -= self.line_spacing
                
                c.drawString(text_x, y, "The Dalles, OR 97058")
                
                # Create new page if needed
                if (idx + 1) % self.page_size == 0 and (idx + 1) < total_items:
                    c.showPage()
                    c.setFillColor(colors.black)
        
        # Save the PDF
        c.save()
        logger.info(f"PDF saved as '{output_path}'")
        
        return output_path


# --------------------------
# Utility Functions
# --------------------------

def open_file(file_path: str) -> None:
    """Open a file with the default system application"""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', file_path], check=False)
        else:  # Linux and other Unix-like
            subprocess.run(['xdg-open', file_path], check=False)
    except Exception as e:
        logger.warning(f"Could not auto-open file: {e}")


async def send_discord_notification(session: aiohttp.ClientSession, webhook_url: str, message: str) -> None:
    """Send a notification to Discord webhook"""
    if not webhook_url:
        logger.warning("Discord webhook URL not configured, skipping notification")
        return
        
    payload = {"content": message}
    try:
        async with session.post(webhook_url, json=payload) as response:
            response.raise_for_status()
            logger.info("Discord notification sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")


# --------------------------
# Label Generation Workflow
# --------------------------

class LabelGenerator:
    """Main workflow handler for label generation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.pdf_generator = PDFGenerator(config)
        
    async def process_customers(self) -> List[LabelEntry]:
        """Process customers and generate label entries"""
        async with aiohttp.ClientSession() as session:
            splynx_client = SplynxAPIClient(self.config, session)
            usps_client = USPSClient(self.config, session)
            
            # Step 1: Get all customers
            customers = await splynx_client.get_customers()
            logger.info(f"Total customers: {len(customers)}")
            
            # Step 2: Filter customers with the target label
            target_customers: List[CustomerInfo] = []
            customer_ids: List[int] = []
            
            for customer in customers:
                has_target_label = any(
                    label.get("label") == self.config.target_label 
                    for label in customer.get("customer_labels", [])
                )
                
                if has_target_label:
                    customer_id = customer["id"]
                    customer_ids.append(customer_id)
                    target_customers.append(CustomerInfo(
                        id=customer_id,
                        name=customer.get("name", ""),
                        raw_data=customer
                    ))
            
            logger.info(f"Found {len(target_customers)} customers with '{self.config.target_label}' label")
            
            # Step 3: Get billing info for target customers
            if target_customers:
                billing_info = await splynx_client.get_billing_info_batch(customer_ids)
                
                # Step 4: Extract ZIP codes and look up city/state info
                zip_codes: Set[str] = set()
                
                for customer in target_customers:
                    customer.billing_info = billing_info.get(customer.id, {})
                    
                    # Determine which address to use (billing or default)
                    if customer.billing_info.get("billing_person"):
                        zip_code = (
                            customer.billing_info.get("billing_zip_code") or 
                            customer.raw_data.get("zip_code", "")
                        ).strip()
                    else:
                        zip_code = customer.raw_data.get("zip_code", "").strip()
                        
                    if zip_code:
                        zip_codes.add(zip_code)
                
                # Get city/state for all ZIP codes in parallel
                city_state_tasks = [
                    usps_client.lookup_city_state_by_zip(zip_code)
                    for zip_code in zip_codes if zip_code
                ]
                await asyncio.gather(*city_state_tasks)
                
                # Step 5: Create address objects for validation
                address_validation_tasks = []
                
                for customer in target_customers:
                    # Determine which address to use (billing or default)
                    if customer.billing_info.get("billing_person"):
                        name = customer.billing_info.get("billing_person", customer.name)
                        street = customer.billing_info.get("billing_street_1", customer.raw_data.get("street_1", ""))
                        zip_code = (
                            customer.billing_info.get("billing_zip_code") or 
                            customer.raw_data.get("zip_code", "")
                        ).strip()
                    else:
                        name = customer.name
                        street = customer.raw_data.get("street_1", "")
                        zip_code = customer.raw_data.get("zip_code", "").strip()
                        
                    customer.name = name
                    
                    # Get city/state from cache
                    location = usps_client.city_state_cache.get(zip_code, {})
                    if not location:
                        logger.warning(f"No city/state for customer {customer.id}, ZIP {zip_code}")
                        continue
                        
                    # Create address for validation
                    address = AddressInfo(
                        street=street,
                        city=location["city"],
                        state=location["state"],
                        zip_code=zip_code
                    )
                    
                    # Add to validation task list
                    address_validation_tasks.append((customer, address))
                
                # Step 6: Validate addresses with USPS
                labels: List[LabelEntry] = []
                total = len(address_validation_tasks)
                processed = 0
                
                for customer, address in address_validation_tasks:
                    processed += 1
                    valid, validated_address = await usps_client.validate_address(address)
                    
                    if valid and validated_address:
                        customer.address = validated_address
                        label_entry = LabelEntry.from_customer(customer)
                        if label_entry:
                            labels.append(label_entry)
                            logger.info(f"Validated {processed}/{total} — {customer.name}")
                    else:
                        logger.warning(f"Validation failed {processed}/{total} — {customer.name}")
                
                # Sort labels by name
                labels.sort(key=lambda x: x.name.split()[0].lower())
                logger.info(f"Prepared {len(labels)} labels")
                
                return labels
            
            return []
        
    async def generate_labels(self) -> str:
        """Run the full label generation process"""
        labels = await self.process_customers()
        
        if not labels:
            logger.warning("No valid labels to generate")
            return ""
            
        # Generate PDF
        output_path = self.pdf_generator.generate(labels)
        
        # Auto-open the PDF
        open_file(output_path)
        
        return output_path


# --------------------------
# Application Entry Point
# --------------------------

async def main() -> None:
    """Main application entry point"""
    start_time = time.time()
    logger.info("Label generator started")
    
    output_path = ""
    try:
        config = AppConfig()
        generator = LabelGenerator(config)
        output_path = await generator.generate_labels()
    except Exception as e:
        error_message = f"Label generator failed: {e}"
        logger.error(error_message)
        
        # Send Discord notification
        async with aiohttp.ClientSession() as session:
            await send_discord_notification(
                session,
                config.discord_webhook_url,
                error_message
            )
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Label generator completed in {execution_time:.2f}s")
        
        # Send warning if execution took too long
        if execution_time > config.time_threshold:
            async with aiohttp.ClientSession() as session:
                await send_discord_notification(
                    session,
                    config.discord_webhook_url,
                    f"WARNING: Label generation took {execution_time:.2f}s"
                )
                
        # Return success/failure status
        return bool(output_path)

# --------------------------
# Logging Setup
# --------------------------

def setup_logging(config: AppConfig) -> None:
    """Configure application logging"""
    global logger
    
    logger = logging.getLogger("label_generator")
    logger.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)
    logger.handlers = []  # Clear any existing handlers


# File handler
    file_handler = logging.FileHandler(config.log_file, mode="a")
    file_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)
    logger.addHandler(console_handler)


# --------------------------
# Script Entry Point
# --------------------------

if __name__ == "__main__":
    # Initial configuration
    config = AppConfig()
    
    # Set up logging
    setup_logging(config)
    
    # Run the main async function
    if sys.version_info >= (3, 7):
        # Python 3.7+ has native support for asyncio.run
        asyncio.run(main())
    else:
        # For older Python versions
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()