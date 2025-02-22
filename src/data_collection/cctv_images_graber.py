import os
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, time as dt_time
from urllib.parse import urlparse, parse_qs
from collections import defaultdict

import requests
from PIL import Image
import imagehash
from selenium import webdriver
from selenium.webdriver.common.by import By

class ImageProcessor:
    """Handles image processing and duplicate detection"""
    
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)
        self.hash_cache = {}  # Cache for image hashes
        self.min_file_size = 1024  # Minimum file size in bytes (1KB)
        self.hash_threshold = 3  # Threshold for hash difference
        
    def get_image_hash(self, image_path):
        """Calculate both average and difference hashes of an image with caching."""
        if image_path in self.hash_cache:
            return self.hash_cache[image_path]
            
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Calculate multiple hash types for better accuracy
                avg_hash = str(imagehash.average_hash(img, hash_size=16))  # Increased hash size
                d_hash = str(imagehash.dhash(img, hash_size=16))
                p_hash = str(imagehash.phash(img, hash_size=16))
                
                hashes = (avg_hash, d_hash, p_hash)
                self.hash_cache[image_path] = hashes
                return hashes
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None, None, None

    def verify_image_similarity(self, img1_path, img2_path):
        """Enhanced duplicate verification using multiple hash types and file attributes"""
        try:
            # Check file size first
            if abs(img1_path.stat().st_size - img2_path.stat().st_size) < self.min_file_size:
                return False
                
            hashes1 = self.get_image_hash(img1_path)
            hashes2 = self.get_image_hash(img2_path)
            
            if not all(hashes1) or not all(hashes2):
                return False
            
            # Compare all hash types
            avg_hash_diff = imagehash.hex_to_hash(hashes1[0]) - imagehash.hex_to_hash(hashes2[0])
            d_hash_diff = imagehash.hex_to_hash(hashes1[1]) - imagehash.hex_to_hash(hashes2[1])
            p_hash_diff = imagehash.hex_to_hash(hashes1[2]) - imagehash.hex_to_hash(hashes2[2])
            
            # Images are considered duplicates only if all hash differences are below threshold
            return all(diff < self.hash_threshold for diff in [avg_hash_diff, d_hash_diff, p_hash_diff])
            
        except Exception as e:
            logging.error(f"Error comparing images {img1_path} and {img2_path}: {e}")
            return False

    def delete_duplicates(self, folder_path):
        """Enhanced duplicate detection and deletion with smart selection"""
        hash_dict = defaultdict(list)
        
        # First pass: collect all image hashes
        for image_path in folder_path.glob('*.jpg'):
            hashes = self.get_image_hash(image_path)
            if hashes and hashes[0]:  # Use average hash as primary key
                hash_dict[hashes[0]].append(image_path)
        
        duplicates_found = 0
        space_saved = 0
        
        for avg_hash, image_paths in hash_dict.items():
            if len(image_paths) > 1:
                # Sort images by quality metrics (file size and creation time)
                image_paths.sort(key=lambda x: (x.stat().st_size, x.stat().st_mtime), reverse=True)
                
                # Keep the highest quality image
                original = image_paths[0]
                verified_duplicates = []
                
                # Verify each potential duplicate
                for dup_candidate in image_paths[1:]:
                    if self.verify_image_similarity(original, dup_candidate):
                        verified_duplicates.append(dup_candidate)
                
                if verified_duplicates:
                    logging.info(f"\nDuplicate set found:")
                    logging.info(f"Keeping: {original} (size: {original.stat().st_size / 1024:.2f}KB)")
                    logging.info("Deleting verified duplicates:")
                    
                    for dup in verified_duplicates:
                        try:
                            file_size = dup.stat().st_size
                            logging.info(f"- {dup} (size: {file_size / 1024:.2f}KB)")
                            
                            if dup.exists():
                                os.remove(dup)
                                duplicates_found += 1
                                space_saved += file_size
                                # Clear hash cache for deleted file
                                self.hash_cache.pop(dup, None)
                                logging.info(f"Successfully deleted: {dup}")
                            else:
                                logging.warning(f"File not found for deletion: {dup}")
                                
                        except Exception as e:
                            logging.error(f"Error deleting {dup}: {e}")
        
        if duplicates_found > 0:
            logging.info(f"\nDuplicate cleanup summary:")
            logging.info(f"Total duplicates deleted: {duplicates_found}")
            logging.info(f"Space saved: {space_saved / (1024*1024):.2f} MB")
        
        return duplicates_found

class TimeManager:
    """Manages capture time windows and scheduling"""
    
    def __init__(self, time_mode='windows', morning_start=(5,0), morning_end=(9,0), 
                 evening_start=(15,0), evening_end=(19,0), 
                 all_day_start=(4,30), all_day_end=(0,0)):
        self.time_mode = time_mode
        self.morning_start = dt_time(*morning_start)
        self.morning_end = dt_time(*morning_end)
        self.evening_start = dt_time(*evening_start)
        self.evening_end = dt_time(*evening_end)
        # For all-day mode
        self.all_day_start = dt_time(*all_day_start)
        self.all_day_end = dt_time(*all_day_end)
        
    def is_capture_time(self):
        """Check if current time is within capture windows"""
        current_time = datetime.now().time()
        
        if self.time_mode == 'all_day':
            # For midnight (00:00) comparison
            if self.all_day_end == dt_time(0, 0):
                return (self.all_day_start <= current_time or 
                       current_time <= self.all_day_end)
            return self.all_day_start <= current_time <= self.all_day_end
        else:
            # Original window mode logic
            is_morning = self.morning_start <= current_time <= self.morning_end
            is_evening = self.evening_start <= current_time <= self.evening_end
            return is_morning or is_evening

    def wait_until_next_window(self):
        """Calculate time until next capture window and wait"""
        current_time = datetime.now()
        today = current_time.date()
        
        if self.time_mode == 'all_day':
            if self.all_day_end == dt_time(0, 0):
                # If current time is before start time
                if current_time.time() < self.all_day_start:
                    next_start = datetime.combine(today, self.all_day_start)
                else:
                    # Wait until tomorrow's start time
                    next_start = datetime.combine(today + timedelta(days=1), 
                                               self.all_day_start)
            else:
                if current_time.time() < self.all_day_start:
                    next_start = datetime.combine(today, self.all_day_start)
                else:
                    next_start = datetime.combine(today + timedelta(days=1), 
                                               self.all_day_start)
                    
            wait_seconds = (next_start - current_time).total_seconds()
            next_window = f"{self.all_day_start.hour:02d}:{self.all_day_start.minute:02d}"
        
        else:
            # Original window mode logic
            morning_start = datetime.combine(today, self.morning_start)
            morning_end = datetime.combine(today, self.morning_end)
            evening_start = datetime.combine(today, self.evening_start)
            evening_end = datetime.combine(today, self.evening_end)
            
            if current_time < morning_start:
                wait_seconds = (morning_start - current_time).total_seconds()
                next_window = f"{self.morning_start.hour:02d}:{self.morning_start.minute:02d}"
            elif morning_end < current_time < evening_start:
                wait_seconds = (evening_start - current_time).total_seconds()
                next_window = f"{self.evening_start.hour:02d}:{self.evening_start.minute:02d}"
            else:
                tomorrow_morning = datetime.combine(today + timedelta(days=1), 
                                                 self.morning_start)
                wait_seconds = (tomorrow_morning - current_time).total_seconds()
                next_window = f"{self.morning_start.hour:02d}:{self.morning_start.minute:02d}"
        
        wait_hours = wait_seconds / 3600
        
        logging.info(f"Outside capture window. Waiting until {next_window}")
        logging.info(f"Will resume in approximately {int(wait_hours)} hours")
        
        return wait_seconds

class CameraCapture:
    """Main class for camera image capture and management"""
    
    def __init__(self, url, img_class, base_folder="captured_images",
                 morning_window=(5,0,9,0), evening_window=(15,0,19,0)):
        self.url = url
        self.img_class = img_class
        self.base_folder = Path(base_folder)
        self.time_manager = TimeManager(
            morning_start=morning_window[:2],
            morning_end=morning_window[2:],
            evening_start=evening_window[:2],
            evening_end=evening_window[2:]
        )
        self.image_processor = ImageProcessor(self.base_folder)
        self.last_saved_hash = None
        self.min_time_between_captures = 1  # Minimum seconds between captures
        self.last_capture_time = 0
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.base_folder / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f'capture_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def setup_folders(self):
        """Create folder structure"""
        images_dir = self.base_folder / 'images'
        images_dir.mkdir(exist_ok=True, parents=True)
        today_folder = images_dir / datetime.now().strftime('%Y-%m-%d')
        today_folder.mkdir(exist_ok=True)
        return today_folder

    def get_timestamp_from_url(self, url):
        """Extract timestamp parameter from URL"""
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        return params.get('t', [None])[0]

    def capture_images(self):
        """Main capture loop"""
        try:
            while True:
                try:
                    if not self.time_manager.is_capture_time():
                        wait_time = self.time_manager.wait_until_next_window()
                        time.sleep(wait_time)
                        continue
                    
                    options = webdriver.ChromeOptions()
                    options.add_argument('--headless')
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    
                    driver = webdriver.Chrome(options=options)
                    driver.get(self.url)
                    
                    self._run_capture_session(driver)
                    
                except KeyboardInterrupt:
                    logging.info("\nMonitoring stopped by user")
                    if 'driver' in locals():
                        driver.quit()
                    break
                except Exception as e:
                    logging.error(f"Session error: {e}")
                    if 'driver' in locals():
                        driver.quit()
                    # Add sleep before retry
                    time.sleep(60)  # Wait 1 minute before retrying
        finally:
            # Cleanup on exit
            if 'driver' in locals():
                driver.quit()
            logging.info("Capture session ended, cleaning up...")

    def _run_capture_session(self, driver):
        """Run a single capture session"""
        js_code = self._get_observer_js()
        save_folder = self.setup_folders()
        last_cleanup_time = time.time()
        cleanup_interval = 3600  # 1 hour in seconds
        
        initial_url = driver.execute_script(js_code, self.img_class)
        initial_timestamp = self.get_timestamp_from_url(initial_url)
        logging.info(f"Starting capture session... Initial timestamp: {initial_timestamp}")
        
        while self.time_manager.is_capture_time():
            current_time = time.time()
            if current_time - last_cleanup_time >= cleanup_interval:
                logging.info(f"Hourly cleanup interval reached. Starting duplicate cleanup...")
                duplicates_removed = self.image_processor.delete_duplicates(save_folder)
                if duplicates_removed:
                    logging.info(f"Cleanup completed. Removed {duplicates_removed} duplicate images.")
                else:
                    logging.info("Cleanup completed. No duplicates found.")
                last_cleanup_time = current_time
            
            change_data = self._handle_timestamp_change(driver)
            self._process_new_image(change_data, save_folder)
        
        logging.info("Capture window ended. Closing browser session.")
        driver.quit()

    def _get_observer_js(self):
        """Get JavaScript code for image observation"""
        return """
        let lastTimestamp = '';
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
                    const newUrl = mutation.target.src;
                    const timestamp = new URLSearchParams(new URL(newUrl).search).get('t');
                    if (timestamp && timestamp !== lastTimestamp) {
                        lastTimestamp = timestamp;
                        const event = new CustomEvent('timestampChanged', {
                            detail: {url: newUrl, timestamp: timestamp}
                        });
                        document.dispatchEvent(event);
                    }
                }
            });
        });
        
        const img = document.querySelector('.' + arguments[0]);
        observer.observe(img, {
            attributes: true,
            attributeFilter: ['src']
        });
        
        return img.src;
        """

    def _handle_timestamp_change(self, driver):
        """Handle timestamp change events"""
        return driver.execute_script("""
            return new Promise((resolve) => {
                document.addEventListener('timestampChanged', (event) => {
                    resolve(event.detail);
                }, {once: true});
            });
        """)

    def _process_new_image(self, change_data, save_folder):
        """Enhanced image processing with immediate duplicate detection"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last capture
            if current_time - self.last_capture_time < self.min_time_between_captures:
                logging.debug(f'Skipping capture - too soon since last capture')
                return
                
            response = requests.get(change_data['url'])
            if response.status_code == 200:
                # Create temporary file to check for duplicates
                temp_file = save_folder / f'temp_{change_data["timestamp"]}.jpg'
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                # Get hash of new image
                new_hashes = self.image_processor.get_image_hash(temp_file)
                
                if not new_hashes or not new_hashes[0]:
                    logging.error(f'Failed to generate hash for new image')
                    temp_file.unlink()
                    return
                
                # Check if this image is too similar to the last saved one
                if self.last_saved_hash and \
                   abs(imagehash.hex_to_hash(new_hashes[0]) - imagehash.hex_to_hash(self.last_saved_hash)) < 3:
                    logging.info(f'Skipping too similar to last saved image: {change_data["timestamp"]}')
                    temp_file.unlink()
                    return
                
                # If we get here, save the image properly
                final_filename = save_folder / f'camera_{change_data["timestamp"]}.jpg'
                temp_file.rename(final_filename)
                self.last_saved_hash = new_hashes[0]
                self.last_capture_time = current_time
                
                logging.info(f'New image captured! Timestamp: {change_data["timestamp"]}')
                
            else:
                logging.error(f'Failed to download image. Status code: {response.status_code}')
        
        except Exception as e:
            logging.error(f'Error downloading image: {str(e)}')
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Camera Image Capture Tool with Duplicate Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url "https://camera.url" --class "camera-class"
  %(prog)s --url "https://camera.url" --class "camera-class" --morning 6:00-10:00 --evening 14:00-18:00
  %(prog)s --url "https://camera.url" --class "camera-class" --all-day
  %(prog)s --help
        """
    )
    
    default_output = str(Path('src') / 'data_collection' / 'captured_images')
    
    parser.add_argument('--url', required=True,
                      help='URL of the camera page')
    parser.add_argument('--class', required=True, dest='img_class',
                      help='HTML class name of the camera image element')
    parser.add_argument('--morning', default='5:00-9:00',
                      help='Morning capture window (format: HH:MM-HH:MM)')
    parser.add_argument('--evening', default='15:00-19:00',
                      help='Evening capture window (format: HH:MM-HH:MM)')
    parser.add_argument('--all-day', action='store_true',
                      help='Run from 05:00 to 22:00')
    parser.add_argument('--output', default=default_output,
                      help='Base output directory for captured images')
    
    args = parser.parse_args()
    
    # Parse time windows
    def parse_time_window(window_str):
        start, end = window_str.split('-')
        start_h, start_m = map(int, start.split(':'))
        end_h, end_m = map(int, end.split(':'))
        return (start_h, start_m, end_h, end_m)
    
    morning_window = parse_time_window(args.morning)
    evening_window = parse_time_window(args.evening)
    
    return args, morning_window, evening_window

def main():
    """Main entry point"""
    args, morning_window, evening_window = parse_args()
    
    camera = CameraCapture(
        url=args.url,
        img_class=args.img_class,
        base_folder=args.output,
        morning_window=morning_window,
        evening_window=evening_window
    )
    
    # Modify CameraCapture initialization based on mode
    if args.all_day:
        camera.time_manager = TimeManager(
            time_mode='all_day',
            all_day_start=(5, 00),
            all_day_end=(22, 0)
        )
    
    camera.capture_images()

if __name__ == "__main__":
    main()