"""
Azure Blob Storage Data Loader for ISIC 2019 SkinVision Project

This module handles downloading images and metadata from Azure blob storage
for local processing on Colab GPUs.
"""

import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import hashlib


class AzureBlobLoader:
    """
    Handles downloading images and metadata from Azure blob storage.
    Optimized for batch processing and GPU pipeline integration.
    """
    
    def __init__(self, storage_account: str = "w281saysxxfypm"):
        """
        Initialize the Azure Blob Loader.
        
        Args:
            storage_account: Azure storage account name
        """
        self.storage_account = storage_account
        self.base_url = f"https://{storage_account}.blob.core.windows.net"
        
        # Container configurations from your exploratory notebook
        self.containers = {
            'isic': {
                'name': 'isic2019-images',
                'prefix': 'isic_2019/'
            },
            'ham': {
                'name': 'ham-10000-images', 
                'prefix': 'ham10000/'
            }
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_metadata_url(self, dataset: str, metadata_type: str) -> str:
        """
        Generate URL for metadata CSV files.
        
        Args:
            dataset: 'isic' or 'ham'
            metadata_type: 'training_metadata', 'ground_truth', etc.
            
        Returns:
            Full URL to the metadata CSV
        """
        if dataset == 'isic':
            container = self.containers['isic']
            if metadata_type == 'training_metadata':
                return f"{self.base_url}/{container['name']}/{container['prefix']}ISIC_2019_Training_Metadata.csv"
            elif metadata_type == 'ground_truth':
                return f"{self.base_url}/{container['name']}/{container['prefix']}training_ground_truth/ISIC_2019_Training_GroundTruth.csv"
            elif metadata_type == 'test_ground_truth':
                return f"{self.base_url}/{container['name']}/{container['prefix']}test_ground_truth/ISIC_2019_Test_GroundTruth.csv"
        
        elif dataset == 'ham':
            container = self.containers['ham']
            if metadata_type == 'metadata':
                return f"{self.base_url}/{container['name']}/{container['prefix']}HAM10000_metadata.csv"
        
        raise ValueError(f"Unknown dataset: {dataset} or metadata_type: {metadata_type}")
    
    def load_metadata(self, dataset: str, metadata_type: str) -> pd.DataFrame:
        """
        Load metadata CSV from Azure blob storage.
        
        Args:
            dataset: 'isic' or 'ham'  
            metadata_type: Type of metadata to load
            
        Returns:
            DataFrame with metadata
        """
        csv_url = self.get_metadata_url(dataset, metadata_type)
        self.logger.info(f"Loading metadata from: {csv_url}")
        
        try:
            df = pd.read_csv(csv_url)
            self.logger.info(f"Loaded {len(df)} records from {dataset} {metadata_type}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise
    
    def get_image_url(self, dataset: str, image_id: str, subfolder: str = None) -> str:
        """
        Generate URL for individual image files.
        
        Args:
            dataset: 'isic' or 'ham'
            image_id: Image identifier (with or without .jpg extension)
            subfolder: Subfolder within the dataset (e.g., 'training_data', 'HAM10000_images_part_1')
            
        Returns:
            Full URL to the image
        """
        if not image_id.endswith('.jpg'):
            image_id = f"{image_id}.jpg"
            
        container = self.containers[dataset]
        base_path = f"{self.base_url}/{container['name']}/{container['prefix']}"
        
        if dataset == 'isic':
            subfolder = subfolder or 'training_data'
            return f"{base_path}{subfolder}/{image_id}"
        
        elif dataset == 'ham':
            # HAM10000 images are split into part_1 and part_2
            subfolder = subfolder or 'HAM10000_images_part_1'  # Default to part_1
            return f"{base_path}{subfolder}/{image_id}"
        
        raise ValueError(f"Unknown dataset: {dataset}")
    
    def download_single_image(self, url: str, local_path: str, max_retries: int = 3) -> bool:
        """
        Download a single image with retry logic.
        
        Args:
            url: Image URL
            local_path: Local file path to save image
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Verify it's a valid image
                    img = Image.open(BytesIO(response.content))
                    img.verify()  # Verify the image is valid
                    
                    # Reopen and save (verify() closes the image)
                    img = Image.open(BytesIO(response.content))
                    img.save(local_path)
                    return True
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def download_batch(self, 
                      image_list: List[str], 
                      dataset: str,
                      local_dir: str,
                      max_workers: int = 5,
                      subfolder: str = None) -> Tuple[List[str], List[str]]:
        """
        Download a batch of images in parallel.
        
        Args:
            image_list: List of image IDs to download
            dataset: 'isic' or 'ham'
            local_dir: Local directory to save images
            max_workers: Number of parallel download threads
            subfolder: Subfolder within the dataset
            
        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        os.makedirs(local_dir, exist_ok=True)
        
        successful = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {}
            
            for image_id in image_list:
                local_path = os.path.join(local_dir, f"{image_id}.jpg" if not image_id.endswith('.jpg') else image_id)
                
                # Skip if already exists
                if os.path.exists(local_path):
                    successful.append(image_id)
                    continue
                
                # Simple URL generation for ISIC dataset
                url = self.get_image_url(dataset, image_id, subfolder)
                future = executor.submit(self.download_single_image, url, local_path)
                future_to_image[future] = (image_id, url)
        
            # Process completed downloads
            for future in tqdm(as_completed(future_to_image), total=len(future_to_image), desc="Downloading images"):
                image_id, url = future_to_image[future]
                try:
                    success = future.result()
                    if success:
                        successful.append(image_id)
                    else:
                        failed.append(image_id)
                        self.logger.error(f"Failed to download: {image_id}")
                except Exception as e:
                    failed.append(image_id)
                    self.logger.error(f"Exception downloading {image_id}: {e}")
    
        self.logger.info(f"Downloaded {len(successful)}/{len(image_list)} images")
        return successful, failed
    
    def prepare_isic_dataset(self, local_dir: str, max_images: int = None, batch_size: int = 500) -> pd.DataFrame:
        """
        Download ISIC 2019 dataset with ground truth labels.
        
        Args:
            local_dir: Local directory to save images and metadata
            max_images: Maximum number of images to download (for testing)
            batch_size: Number of images to download per batch
            
        Returns:
            DataFrame with image paths and labels
        """
        self.logger.info("Preparing ISIC 2019 dataset...")
        
        # Load metadata and ground truth
        metadata_df = self.load_metadata('isic', 'training_metadata')
        gt_df = self.load_metadata('isic', 'ground_truth')
        
        # Merge metadata with ground truth
        dx_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        gt_df['dx'] = gt_df[dx_cols].idxmax(axis=1)
        
        merged_df = metadata_df.merge(gt_df[['image', 'dx']], on='image', how='left')
        
        if max_images:
            merged_df = merged_df.head(max_images)
        
        self.logger.info(f"Will download {len(merged_df)} images")
        
        # Download images in batches
        image_dir = os.path.join(local_dir, 'isic_images')
        all_successful = []
        all_failed = []
        
        image_list = merged_df['image'].tolist()
        total_batches = (len(image_list) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_list), batch_size):
            batch_num = i // batch_size + 1
            batch = image_list[i:i+batch_size]
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
            
            successful, failed = self.download_batch(
                batch, 
                'isic', 
                image_dir,
                subfolder='training_data'
            )
            
            all_successful.extend(successful)
            all_failed.extend(failed)
            
            self.logger.info(f"Batch {batch_num} complete: {len(successful)} success, {len(failed)} failed")
        
        # Filter to successfully downloaded images
        final_df = merged_df[merged_df['image'].isin(all_successful)].copy()
        final_df['local_path'] = final_df['image'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
        
        # Save metadata
        metadata_path = os.path.join(local_dir, 'isic_metadata.csv')
        final_df.to_csv(metadata_path, index=False)
        
        self.logger.info(f"ISIC dataset ready: {len(final_df)} images in {local_dir}")
        if all_failed:
            self.logger.warning(f"Failed to download {len(all_failed)} images")
        
        return final_df


def main():
    """
    Example usage of the AzureBlobLoader
    """
    loader = AzureBlobLoader()
    
    # Download full dataset (remove max_images limit)
    isic_df = loader.prepare_isic_dataset(
        local_dir='../images/isic',
        max_images=None  # Set to None to download all images
    )
    
    print(f"Downloaded {len(isic_df)} ISIC images")
    print(f"Label distribution:\n{isic_df['dx'].value_counts()}")


if __name__ == "__main__":
    main()