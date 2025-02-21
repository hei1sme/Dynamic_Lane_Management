import os
import argparse
from PIL import Image
import imagehash
from pathlib import Path
from typing import List, Set, Dict

class DuplicateImageRemover:
    def __init__(self, folder_path: str, hash_size: int = 8):
        """
        Initialize the DuplicateImageRemover class.
        
        Args:
            folder_path (str): Path to the folder containing images
            hash_size (int): Size of the hash for image comparison
        """
        self.folder_path = Path(folder_path)
        self.hash_size = hash_size
        self.image_hashes: Dict[str, List[Path]] = {}
        
    def is_image(self, filename: str) -> bool:
        """Check if the file is an image based on its extension."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        return Path(filename).suffix.lower() in valid_extensions
    
    def compute_image_hash(self, image_path: Path) -> str:
        """Compute the perceptual hash of an image."""
        try:
            with Image.open(image_path) as img:
                return str(imagehash.average_hash(img, self.hash_size))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def find_duplicates(self) -> None:
        """Find and group duplicate images based on their hash values."""
        for file_path in self.folder_path.rglob('*'):
            if file_path.is_file() and self.is_image(file_path.name):
                image_hash = self.compute_image_hash(file_path)
                if image_hash:
                    if image_hash in self.image_hashes:
                        self.image_hashes[image_hash].append(file_path)
                    else:
                        self.image_hashes[image_hash] = [file_path]
    
    def remove_duplicates(self, keep_first: bool = True) -> None:
        """
        Remove duplicate images, keeping either the first or last occurrence.
        
        Args:
            keep_first (bool): If True, keeps the first occurrence; if False, keeps the last
        """
        removed_count = 0
        for hash_value, file_paths in self.image_hashes.items():
            if len(file_paths) > 1:
                # Sort files by creation time
                file_paths.sort(key=lambda x: x.stat().st_ctime)
                # Keep either first or last file
                files_to_remove = file_paths[1:] if keep_first else file_paths[:-1]
                
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        print(f"Removed duplicate: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
        
        print(f"\nTotal duplicates removed: {removed_count}")

def main():
    parser = argparse.ArgumentParser(description="Remove duplicate images from a folder")
    parser.add_argument("folder_path", help="Path to the folder containing images")
    parser.add_argument("--hash-size", type=int, default=16,
                        help="Size of the image hash (default: 16)")
    parser.add_argument("--keep-last", action="store_true",
                        help="Keep the last occurrence instead of the first")
    
    args = parser.parse_args()
    
    remover = DuplicateImageRemover(args.folder_path, args.hash_size)
    print("Finding duplicates...")
    remover.find_duplicates()
    print("Removing duplicates...")
    remover.remove_duplicates(keep_first=not args.keep_last)

if __name__ == "__main__":
    main()
