"""
Image Preprocessor for ISIC 2019 SkinVision Project

This module handles image preprocessing including:
- Vignette detection and removal
- Square cropping and resizing
- Data augmentation (rotation, flipping)
- GPU-optimized batch processing

Combines logic from size_correction_and_vignetting.ipynb and resampling.ipynb
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import math  # Add to imports at top


class ImagePreprocessor:
    """
    Handles image preprocessing for dermatological images.
    Optimized for GPU processing and batch operations.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (450, 450)):
        """
        Initialize the Image Preprocessor.
        
        Args:
            target_size: Final size for all processed images (width, height)
        """
        self.target_size = target_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Augmentation methods from your resampling notebook
        self.augmentation_methods = [
            'rot0',     # Original image
            'rot90',    # 90 degree rotation
            'rot180',   # 180 degree rotation  
            'rot270',   # 270 degree rotation
            'flipud',   # Vertical flip
            'fliplr'    # Horizontal flip
        ]
    
    def detect_circular_vignette(self, 
                               image: np.ndarray, 
                               brightness_threshold: float = 0.95,
                               display: bool = False) -> Tuple[np.ndarray, int, int]:
        """
        Detect and remove circular vignettes based on radial brightness profile.
        Crops the image to the square inscribed in the circle of the vignette.
        
        From your size_correction_and_vignetting.ipynb notebook.
        
        Args:
            image: Input image as numpy array (RGB)
            brightness_threshold: Threshold for vignette detection (0.0-1.0)
            display: Whether to show debug plots
            
        Returns:
            Tuple of (cropped_image, new_width, new_height)
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Get image dimensions and center
        h, w = gray.shape
        center = (w / 2.0, h / 2.0)  # Float coordinates for cv2.getRectSubPix
        
        # Compute distance from center for each pixel
        Y, X = np.indices((h, w))
        distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        # Build radial brightness profile
        max_distance = np.max(distance)
        num_bins = 100
        radial_means = np.zeros(num_bins)
        bin_edges = np.linspace(0, max_distance, num_bins + 1)
        
        for i in range(num_bins):
            mask = (distance >= bin_edges[i]) & (distance < bin_edges[i+1])
            if np.any(mask):
                radial_means[i] = np.mean(gray[mask])
        
        # Normalize the radial brightness profile by dividing each value by the maximum value.
        # This scales the profile to the range [0, 1], making it easier to compare against the threshold.
        if np.max(radial_means) > 0:
            radial_means = radial_means / np.max(radial_means)  # Normalize to 0-1 based on actual max

        # Detect where brightness drops below threshold
        valid_bins = np.where(radial_means > brightness_threshold)[0]
        
        # Handle different cases
        if len(valid_bins) == 100 or len(valid_bins) == 0 or valid_bins[-1] == 99:
            # No vignette detected or edge case - return original
            cropped_img, h2, w2 = image, h, w
        else:
            # Vignette detected - crop to inscribed square
            valid_radius = bin_edges[valid_bins[-1]]
            side_length = int(valid_radius * np.sqrt(2))
            h2, w2 = side_length, side_length
            cropped_img = cv2.getRectSubPix(image, (w2, h2), center)
        
        if display:
            self._display_vignette_debug(image, radial_means, cropped_img, brightness_threshold)
        
        return cropped_img, w2, h2
    
    def _display_vignette_debug(self, 
                              original: np.ndarray, 
                              radial_profile: np.ndarray,
                              cropped: np.ndarray, 
                              threshold: float):
        """Display debug information for vignette detection."""
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        
        ax[0].imshow(original)
        ax[0].set_title(f"Original Image {original.shape}")
        ax[0].axis('off')
        
        ax[1].plot(radial_profile)
        ax[1].set_title('Radial Brightness Profile')
        ax[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold {threshold}')
        ax[1].legend()
        
        ax[2].imshow(cropped)
        ax[2].set_title(f'Cropped Image {cropped.shape}')
        ax[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def crop_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to square dimensions (center crop).
        From your resampling.ipynb preprocessing logic.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Square-cropped image
        """
        height, width = image.shape[:2]
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        
        cropped_img = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
        return cropped_img
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target dimensions using area interpolation.
        
        Args:
            image: Input image as numpy array
            target_size: Target (width, height), uses self.target_size if None
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
            
        resized_img = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized_img
    
    def apply_augmentations(self, 
                          image: np.ndarray, 
                          augmentations: List[str]) -> List[np.ndarray]:
        """
        Apply data augmentation techniques to an image.
        From your resampling.ipynb preprocessing logic.
        
        Args:
            image: Input image as numpy array
            augmentations: List of augmentation methods to apply
            
        Returns:
            List of augmented images
        """
        augmented_samples = []
        
        for aug in augmentations:
            if aug == 'rot0':
                augmented_samples.append(image.copy())
            elif aug == 'rot90':
                augmented_samples.append(np.rot90(image, k=1, axes=(0, 1)))
            elif aug == 'rot180':
                augmented_samples.append(np.rot90(image, k=2, axes=(0, 1)))
            elif aug == 'rot270':
                augmented_samples.append(np.rot90(image, k=3, axes=(0, 1)))
            elif aug == 'flipud':
                augmented_samples.append(np.flipud(image))
            elif aug == 'fliplr':
                augmented_samples.append(np.fliplr(image))
            else:
                self.logger.warning(f"Unknown augmentation method: {aug}")
        
        return augmented_samples
    
    def preprocess_single_image(self, 
                          image_path: str,
                          remove_vignette: bool = True,
                          crop_square: bool = True, 
                          resize: bool = True,
                          augmentations: Optional[List[str]] = None) -> List[np.ndarray]:
        """Complete preprocessing pipeline for a single image."""
        
        # Load image with error handling
        try:
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path  # Already loaded image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # Step 1: Vignette removal
        if remove_vignette:
            try:
                image, _, _ = self.detect_circular_vignette(image)
            except Exception as e:
                self.logger.error(f"Failed to remove vignette from {image_path}: {e}")

            return [image]
        
        # Step 2: Square cropping
        if crop_square:
            try:
                image = self.crop_to_square(image)
            except Exception as e:
                self.logger.error(f"Failed to crop image {image_path} to square: {e}")
                self.logger.error(f"Square cropping failed for image {image_path}. Please check the input image and cropping logic.")
            pass  # Continue to the next step
        
        # Step 3: Resizing
        if resize:
            try:
                image = self.resize_image(image)
            except Exception as e:
                self.logger.error(f"Failed to resize image {image_path}: {e}")
            return [image]
        
        # Step 4: Augmentations
        if augmentations is None:
            return [image]
        else:
            return self.apply_augmentations(image, augmentations)
    
    def preprocess_batch(self, 
                        image_paths: List[str],
                        output_dir: str,
                        metadata_df: pd.DataFrame,
                        augmentations_per_class: Optional[Dict[str, List[str]]] = None,
                        batch_size: int = 50) -> pd.DataFrame:
        """
        Process a batch of images with class-specific augmentation.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images
            metadata_df: DataFrame with image metadata and labels
            augmentations_per_class: Dict mapping class names to augmentation lists
            batch_size: Number of images to process at once
            
        Returns:
            DataFrame with processed image information
        """
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            for image_path in tqdm(batch_paths, desc="Processing images"):
                try:
                    # Get image metadata - more robust matching
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    image_row = None  # Use None instead of empty DataFrame

                    # Strategy 1: Exact match
                    image_row = metadata_df[metadata_df['image'] == image_name]

                    # Strategy 2: Try with .jpg extension
                    if image_row.empty:
                        image_row = metadata_df[metadata_df['image'] == f"{image_name}.jpg"]

                    # Strategy 3: Strip .jpg from metadata and match
                    if image_row.empty:
                        clean_image_column = metadata_df['image'].str.replace('.jpg', '', regex=False)
                        image_row = metadata_df[clean_image_column == image_name]
                    
                    if image_row.empty:
                        self.logger.warning(f"No metadata found for {image_name}")
                        continue
                    
                    label = image_row.iloc[0]['dx']
                    
                    # Determine augmentations for this class
                    if augmentations_per_class and label in augmentations_per_class:
                        augs = augmentations_per_class[label]
                    else:
                        augs = ['rot0']  # Just original image
                    
                    # Process image
                    processed_images = self.preprocess_single_image(
                        image_path, 
                        augmentations=augs
                    )
                    
                    # Save processed images
                    for idx, (processed_img, aug_method) in enumerate(zip(processed_images, augs)):
                        output_filename = f"{image_name}_preprocessed_{aug_method}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Convert back to PIL and save
                        pil_img = Image.fromarray(processed_img.astype(np.uint8))
                        pil_img.save(output_path)
                        
                        # Record metadata
                        processed_data.append({
                            'original_image': image_name,
                            'processed_image': output_filename,
                            'local_path': output_path,
                            'label': label,
                            'augmentation': aug_method,
                            'width': self.target_size[0],
                            'height': self.target_size[1]
                        })
                
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    continue
            
            # Memory cleanup after each batch
            if i > 0 and i % (batch_size * 5) == 0:  # Every 5 batches
                gc.collect()
                self.logger.info("Memory cleanup performed")
        
        # Create output DataFrame
        result_df = pd.DataFrame(processed_data)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'preprocessed_metadata.csv')
        result_df.to_csv(metadata_path, index=False)
        
        self.logger.info(f"Processed {len(result_df)} images, saved to {output_dir}")
        return result_df
    
    def create_balanced_dataset(self, 
                          metadata_df: pd.DataFrame,
                          target_samples_per_class: int = 1000) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        Create augmentation strategy and balanced metadata.
        
        Returns:
            Tuple of (augmentation_strategy, balanced_metadata_df)
        """
        augmentation_strategy = {}
        balanced_data = []
        
        # Analyze class distribution
        class_counts = metadata_df['dx'].value_counts()
        self.logger.info(f"Original class distribution:\n{class_counts}")
        
        for class_name, current_count in class_counts.items():
            class_data = metadata_df[metadata_df['dx'] == class_name]
            
            if current_count >= target_samples_per_class:
                # Downsample - randomly select target number
                sampled_data = class_data.sample(n=target_samples_per_class, random_state=42)
                augmentation_strategy[class_name] = ['rot0']
                balanced_data.append(sampled_data)
                
            else:
                # Upsample - determine needed augmentations
                total_needed = target_samples_per_class
                augs_needed = math.ceil(total_needed / current_count)  # Total augmentations per image
                
                # Select augmentation methods
                selected_augs = self.augmentation_methods[:min(augs_needed, len(self.augmentation_methods))]
                augmentation_strategy[class_name] = selected_augs
                balanced_data.append(class_data)  # Use all available data
                
                self.logger.info(f"Class {class_name}: {current_count} -> {target_samples_per_class} "
                               f"using {selected_augs}")
        
        balanced_metadata_df = pd.concat(balanced_data, ignore_index=True)
        return augmentation_strategy, balanced_metadata_df


def main():
    """
    Example usage of the ImagePreprocessor
    """
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(450, 450))
    
    # Example: Load metadata (assuming it exists from data_loader.py)
    metadata_path = '../images/isic/isic_metadata.csv'
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        
        # Create balanced dataset strategy
        augmentation_strategy, balanced_metadata_df = preprocessor.create_balanced_dataset(
            metadata_df, 
            target_samples_per_class=500  # Smaller for testing
        )
        
        # Use balanced metadata instead of original
        image_paths = balanced_metadata_df['local_path'].tolist()[:100]  # Test with first 100
        
        result_df = preprocessor.preprocess_batch(
            image_paths=image_paths,
            output_dir='../images/processed',
            metadata_df=balanced_metadata_df,  # Use balanced metadata
            augmentations_per_class=augmentation_strategy
        )
        
        print(f"Processed {len(result_df)} images")
        print(f"Augmentation distribution:\n{result_df['augmentation'].value_counts()}")
    else:
        print("No metadata found. Run data_loader.py first!")


if __name__ == "__main__":
    main()