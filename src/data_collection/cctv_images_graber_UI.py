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

import customtkinter as ctk
import threading
import sys
from tkinter import filedialog

class UILogHandler(logging.Handler):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

    def emit(self, record):
        msg = self.format(record)
        self.ui.update_status(msg)

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
        self.stop_capture = False
        self.pause_capture = False
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
        while not self.stop_capture:
            try:
                # Check if paused
                while self.pause_capture and not self.stop_capture:
                    time.sleep(1)
                    logging.info("Capture paused...")
                    continue

                if not self.time_manager.is_capture_time():
                    wait_time = self.time_manager.wait_until_next_window()
                    # Check stop_capture and pause_capture more frequently during wait
                    for _ in range(int(wait_time)):
                        if self.stop_capture:
                            return
                        if self.pause_capture:
                            break
                        time.sleep(1)
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

    def _run_capture_session(self, driver):
        """Run a single capture session"""
        try:
            # Wait for page to load and log initial state
            logging.info("Waiting for page to load...")
            driver.implicitly_wait(10)  # Add implicit wait
            
            # Get initial timestamp
            initial_timestamp = self.get_timestamp_from_url(self.url)
            logging.info(f"Starting capture session... Initial timestamp: {initial_timestamp}")
            
            while not self.stop_capture:
                try:
                    # Check if paused
                    if self.pause_capture:
                        time.sleep(1)
                        continue
                    
                    # Find image element with timeout
                    logging.debug("Looking for image element...")
                    img_element = driver.find_element(By.CLASS_NAME, self.img_class)
                    
                    if not img_element:
                        logging.warning("Image element not found")
                        time.sleep(1)
                        continue
                    
                    current_url = img_element.get_attribute('src')
                    logging.debug(f"Current image URL: {current_url}")
                    
                    if not current_url:
                        logging.warning("No src attribute found")
                        time.sleep(1)
                        continue
                    
                    current_timestamp = self.get_timestamp_from_url(current_url)
                    logging.debug(f"Current timestamp: {current_timestamp}")
                    
                    if current_timestamp and current_timestamp != initial_timestamp:
                        today_folder = self.setup_folders()
                        self._process_new_image({'url': current_url, 'timestamp': current_timestamp}, today_folder)
                        initial_timestamp = current_timestamp
                    
                    # Refresh the page periodically to prevent stale content
                    if not self.stop_capture and not self.pause_capture:
                        driver.refresh()
                        logging.debug("Page refreshed")
                        time.sleep(5)  # Wait after refresh
                    
                except Exception as inner_e:
                    logging.error(f"Error during capture loop: {inner_e}")
                    time.sleep(1)
                
        except Exception as e:
            logging.error(f"Session error: {e}")
        finally:
            try:
                driver.quit()
                logging.info("Browser session closed")
            except Exception as e:
                logging.error(f"Error closing browser: {e}")

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

class CameraCaptureUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("CCTV Image Capture Tool")
        self.root.geometry("800x800")
        
        # Set default output directory
        self.base_output_dir = Path(r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\src\data_collection")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.capture_thread = None
        self.camera = None
        self.is_paused = False
        
        # Create a custom logging handler
        self.log_handler = UILogHandler(self)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        self.setup_ui()

    def setup_ui(self):
        # Create main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title = ctk.CTkLabel(main_frame, text="CCTV Image Capture Tool", 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)

        # Input frame
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(fill="x", padx=20, pady=10)

        # URL input
        url_label = ctk.CTkLabel(input_frame, text="Camera URL:")
        url_label.pack(anchor="w", padx=10, pady=(10,0))
        self.url_entry = ctk.CTkEntry(input_frame, width=400, placeholder_text="Enter camera URL")
        self.url_entry.pack(fill="x", padx=10, pady=(5,10))

        # Image class input
        class_label = ctk.CTkLabel(input_frame, text="Image Class:")
        class_label.pack(anchor="w", padx=10, pady=(10,0))
        self.class_entry = ctk.CTkEntry(input_frame, width=400, placeholder_text="Enter image class name")
        self.class_entry.pack(fill="x", padx=10, pady=(5,10))

        # Time mode selection
        mode_frame = ctk.CTkFrame(main_frame)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(mode_frame, text="Capture Mode:")
        mode_label.pack(anchor="w", padx=10, pady=5)
        
        self.time_mode = ctk.StringVar(value="windows")
        windows_radio = ctk.CTkRadioButton(mode_frame, text="Time Windows", 
                                         variable=self.time_mode, value="windows")
        windows_radio.pack(side="left", padx=20)
        
        allday_radio = ctk.CTkRadioButton(mode_frame, text="All Day", 
                                         variable=self.time_mode, value="all_day")
        allday_radio.pack(side="left", padx=20)

        # Time windows frame
        time_frame = ctk.CTkFrame(main_frame)
        time_frame.pack(fill="x", padx=20, pady=10)
        
        time_label = ctk.CTkLabel(time_frame, text="Time Windows", 
                                 font=ctk.CTkFont(size=16, weight="bold"))
        time_label.pack(pady=10)

        # Morning window
        morning_frame = ctk.CTkFrame(time_frame)
        morning_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(morning_frame, text="Morning:").pack(side="left", padx=10)
        self.morning_start = ctk.CTkEntry(morning_frame, width=100, placeholder_text="HH:MM")
        self.morning_start.pack(side="left", padx=5)
        self.morning_start.insert(0, "5:00")
        
        ctk.CTkLabel(morning_frame, text="to").pack(side="left", padx=5)
        self.morning_end = ctk.CTkEntry(morning_frame, width=100, placeholder_text="HH:MM")
        self.morning_end.pack(side="left", padx=5)
        self.morning_end.insert(0, "9:00")

        # Evening window
        evening_frame = ctk.CTkFrame(time_frame)
        evening_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(evening_frame, text="Evening:").pack(side="left", padx=10)
        self.evening_start = ctk.CTkEntry(evening_frame, width=100, placeholder_text="HH:MM")
        self.evening_start.pack(side="left", padx=5)
        self.evening_start.insert(0, "15:00")
        
        ctk.CTkLabel(evening_frame, text="to").pack(side="left", padx=5)
        self.evening_end = ctk.CTkEntry(evening_frame, width=100, placeholder_text="HH:MM")
        self.evening_end.pack(side="left", padx=5)
        self.evening_end.insert(0, "19:00")

        # Output directory frame with browse button
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.pack(fill="x", padx=20, pady=10)
        
        output_label = ctk.CTkLabel(output_frame, text="Output Directory:", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        output_label.pack(side="left", padx=10, pady=5)
        
        self.browse_button = ctk.CTkButton(output_frame, text="Browse", 
                                         command=self.browse_output_dir,
                                         width=100)
        self.browse_button.pack(side="right", padx=10, pady=5)

        # Status text
        status_label = ctk.CTkLabel(main_frame, text="Status Log:", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        status_label.pack(anchor="w", padx=20, pady=(20,5))
        
        self.status_text = ctk.CTkTextbox(main_frame, height=300)
        self.status_text.pack(fill="x", padx=20, pady=(0,20))

        # Control buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        self.start_button = ctk.CTkButton(button_frame, text="Start Capture", 
                                         command=self.start_capture)
        self.start_button.pack(side="left", padx=10, expand=True)
        
        self.pause_button = ctk.CTkButton(button_frame, text="Pause", 
                                         command=self.pause_capture,
                                         state="disabled",
                                         fg_color="orange",
                                         hover_color="darkorange")
        self.pause_button.pack(side="left", padx=10, expand=True)
        
        self.stop_button = ctk.CTkButton(button_frame, text="Stop Capture", 
                                        command=self.stop_capture, 
                                        state="disabled",
                                        fg_color="red",
                                        hover_color="darkred")
        self.stop_button.pack(side="left", padx=10, expand=True)

    def update_status(self, message):
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")

    def start_capture(self):
        if not self.url_entry.get() or not self.class_entry.get():
            self.show_error("Please enter both URL and Image Class")
            return

        try:
            # Parse time windows
            morning_start_h, morning_start_m = map(int, self.morning_start.get().split(':'))
            morning_end_h, morning_end_m = map(int, self.morning_end.get().split(':'))
            evening_start_h, evening_start_m = map(int, self.evening_start.get().split(':'))
            evening_end_h, evening_end_m = map(int, self.evening_end.get().split(':'))

            morning_window = (morning_start_h, morning_start_m, morning_end_h, morning_end_m)
            evening_window = (evening_start_h, evening_start_m, evening_end_h, evening_end_m)

            # Create output directory using the correct base path
            images_dir = self.base_output_dir / 'captured_images' / 'images'
            today_folder = images_dir / datetime.now().strftime('%Y-%m-%d')
            today_folder.mkdir(parents=True, exist_ok=True)
            
            # Update output path display
            self.browse_button.configure(state="disabled")
            self.browse_button.configure(text=str(today_folder))

            self.camera = CameraCapture(
                url=self.url_entry.get(),
                img_class=self.class_entry.get(),
                base_folder=self.base_output_dir / 'captured_images',
                morning_window=morning_window,
                evening_window=evening_window
            )

            if self.time_mode.get() == "all_day":
                self.camera.time_manager = TimeManager(
                    time_mode='all_day',
                    all_day_start=(5, 0),
                    all_day_end=(22, 0)
                )

            self.capture_thread = threading.Thread(target=self.run_capture)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            logging.info("Capture started...")

        except ValueError:
            self.show_error("Invalid time format. Use HH:MM format.")

    def pause_capture(self):
        """Handle pause/resume functionality"""
        if self.camera:
            if not self.is_paused:
                # Pause the capture
                self.camera.pause_capture = True
                self.is_paused = True
                self.pause_button.configure(text="Resume")
                logging.info("Capture paused...")
            else:
                # Resume the capture
                self.camera.pause_capture = False
                self.is_paused = False
                self.pause_button.configure(text="Pause")
                logging.info("Capture resumed...")

    def stop_capture(self):
        if self.camera:
            self.camera.stop_capture = True
            self.camera.pause_capture = False  # Ensure we're not paused when stopping
            self.start_button.configure(state="normal")
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.pause_button.configure(text="Pause")  # Reset pause button text
            self.is_paused = False
            logging.info("Capture stopped.")

    def show_error(self, message):
        ctk.CTkMessagebox(title="Error", message=message, icon="cancel")

    def run_capture(self):
        try:
            self.camera.capture_images()
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            self.root.after(0, self.stop_capture)  # Safely stop capture on error

    def browse_output_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.base_output_dir,
            title="Select Output Directory"
        )
        if directory:  # If a directory was selected
            self.base_output_dir = Path(directory)
            self.update_status(f"Output directory changed to: {self.base_output_dir}")

    def run(self):
        self.root.mainloop()

def main():
    app = CameraCaptureUI()
    app.run()

if __name__ == "__main__":
    main()