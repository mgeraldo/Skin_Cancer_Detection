"""
Pipeline Runner for ISIC 2019 SkinVision Project

This script orchestrates the complete end-to-end processing pipeline:
1. Data Loading (from Azure Blob Storage)
2. Image Preprocessing (vignette detection, cropping, augmentation) 
3. Feature Extraction (HOG, LBP, color, GLCM, wavelet, Laplace features)

Optimized for GPU processing on Google Colab.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

# Import our pipeline components
from data_loader import AzureBlobLoader
from image_preprocessor import ImagePreprocessor  
from feature_extractor import FeatureExtractor


class SkinVisionPipeline:
    """
    Complete processing pipeline for ISIC 2019 skin lesion classification.
    Orchestrates data loading, preprocessing, and feature extraction.
    """
    
    def __init__(self, 
                 storage_account: str = "w281saysxxfypm",
                 target_size: Tuple[int, int] = (450, 450),
                 batch_size: int = 100,
                 base_output_dir: str = "../pipeline_output"):
        """
        Initialize the pipeline.
        
        Args:
            storage_account: Azure storage account name
            target_size: Target image size (width, height)
            batch_size: Batch size for processing
            base_output_dir: Base directory for all outputs
        """
        self.storage_account = storage_account
        self.target_size = target_size
        self.batch_size = batch_size
        self.base_output_dir = Path(base_output_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self._create_directories()
        
        # Initialize pipeline components
        self.data_loader = AzureBlobLoader(storage_account=storage_account)
        self.image_preprocessor = ImagePreprocessor(target_size=target_size)
        self.feature_extractor = FeatureExtractor(target_size=target_size)
        
        self.logger.info(f"Pipeline initialized with target size {target_size}, batch size {batch_size}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.base_output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def _create_directories(self):
        """Create all necessary output directories."""
        directories = [
            "data/raw",
            "data/metadata", 
            "data/augmented",  # New: for pre-augmented images
            "images/processed",
            "features/batches",
            "features/final",
            "logs",
            "results"
        ]
        
        for dir_name in directories:
            dir_path = self.base_output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created output directories in {self.base_output_dir}")
    
    def run_data_loading(self, 
                        datasets: List[str] = ["isic_2019"],
                        max_images_per_dataset: Optional[int] = None,
                        force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Load data from Azure Blob Storage.
        
        Args:
            datasets: List of datasets to load ["isic_2019", "ham10000"]
            max_images_per_dataset: Limit images per dataset for testing
            force_reload: Force re-download even if files exists
            
        Returns:
            Dictionary of loaded metadata DataFrames
        """
        self.logger.info("="*50)
        self.logger.info("STEP 1: DATA LOADING")
        self.logger.info("="*50)
        
        start_time = time.time()
        metadata_dfs = {}
        
        for dataset in datasets:
            self.logger.info(f"Loading dataset: {dataset}")
            
            try:
                if dataset == "isic_2019":
                    # Use the correct method name and parameters
                    df = self.data_loader.prepare_isic_dataset(
                        local_dir=str(self.base_output_dir / "data" / "raw"),
                        max_images=max_images_per_dataset,
                        batch_size=self.batch_size
                    )
                    # Keep dx column as is - don't rename since image_preprocessor expects 'dx'
                        
                elif dataset == "ham10000":
                    # HAM10000 is not implemented yet in data_loader
                    self.logger.warning(f"HAM10000 dataset loading not implemented yet")
                    continue
                else:
                    self.logger.error(f"Unknown dataset: {dataset}")
                    continue
                
                metadata_dfs[dataset] = df
                
                # Save metadata
                metadata_file = self.base_output_dir / "data" / "metadata" / f"{dataset}_metadata.csv"
                df.to_csv(metadata_file, index=False)
                
                self.logger.info(f"{dataset}: {len(df)} images loaded and metadata saved to {metadata_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {dataset}: {e}")
                continue
        
        elapsed = time.time() - start_time
        self.logger.info(f"Data loading completed in {elapsed:.2f} seconds")
        
        return metadata_dfs
    
    def run_image_preprocessing(self, 
                              metadata_dfs: Dict[str, pd.DataFrame],
                              apply_augmentation: bool = True,
                              balance_dataset: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Step 2: Preprocess images (vignette detection, cropping, augmentation).
        
        Args:
            metadata_dfs: Metadata DataFrames from data loading step
            apply_augmentation: Whether to apply data augmentation
            balance_dataset: Whether to balance classes through augmentation
            
        Returns:
            Dictionary of preprocessed metadata DataFrames
        """
        self.logger.info("="*50)
        self.logger.info("STEP 2: IMAGE PREPROCESSING")
        self.logger.info("="*50)
        
        start_time = time.time()
        preprocessed_dfs = {}
        
        for dataset, df in metadata_dfs.items():
            self.logger.info(f"Preprocessing dataset: {dataset}")
            
            try:
                # Create output directory for this dataset
                output_dir = self.base_output_dir / "images" / "processed" / dataset
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if apply_augmentation and balance_dataset:
                    # STEP 1: Create balanced strategy (no processing yet)
                    augmentation_strategy, balanced_df = self.image_preprocessor.create_balanced_dataset(
                        metadata_df=df,
                        target_samples_per_class=1000
                    )
                    
                    # STEP 2: Now actually process the balanced dataset with augmentations
                    preprocessed_df = self.image_preprocessor.preprocess_batch(
                        image_paths=balanced_df['local_path'].tolist(),
                        output_dir=str(output_dir),
                        metadata_df=balanced_df,
                        augmentations_per_class=augmentation_strategy,
                        batch_size=self.batch_size
                    )
                else:
                    # Standard preprocessing without balancing
                    augmentations_per_class = None
                    if apply_augmentation:
                        # Create simple augmentation strategy for all classes
                        unique_classes = df['dx'].unique() if 'dx' in df.columns else []
                        augmentations_per_class = {cls: ['rot90', 'flip_h'] for cls in unique_classes}
                    
                    preprocessed_df = self.image_preprocessor.preprocess_batch(
                        image_paths=df['local_path'].tolist(),
                        output_dir=str(output_dir),
                        metadata_df=df,
                        augmentations_per_class=augmentations_per_class,
                        batch_size=self.batch_size
                    )
                
                preprocessed_dfs[dataset] = preprocessed_df
                
                # Save preprocessed metadata
                metadata_file = self.base_output_dir / "data" / "metadata" / f"{dataset}_preprocessed.csv"
                preprocessed_df.to_csv(metadata_file, index=False)
                
                self.logger.info(f"{dataset}: {len(preprocessed_df)} preprocessed images, metadata saved to {metadata_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to preprocess {dataset}: {e}")
                continue
        
        elapsed = time.time() - start_time
        self.logger.info(f"Image preprocessing completed in {elapsed:.2f} seconds")
        
        return preprocessed_dfs
    
    def run_feature_extraction(self, 
                             preprocessed_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Step 3: Extract features from preprocessed images.
        
        Args:
            preprocessed_dfs: Preprocessed metadata DataFrames
            
        Returns:
            Dictionary of extracted features by dataset and feature type
        """
        self.logger.info("="*50)
        self.logger.info("STEP 3: FEATURE EXTRACTION")
        self.logger.info("="*50)
        
        start_time = time.time()
        all_features = {}
        
        for dataset, df in preprocessed_dfs.items():
            self.logger.info(f"Extracting features for dataset: {dataset}")
            
            try:
                # Get image paths - the preprocess_batch method returns 'local_path' column
                if 'local_path' in df.columns:
                    image_paths = df['local_path'].tolist()
                else:
                    # Try any path-like column as fallback
                    path_cols = [col for col in df.columns if 'path' in col.lower()]
                    if path_cols:
                        image_paths = df[path_cols[0]].tolist()
                    else:
                        raise ValueError(f"No image path column found in {dataset} preprocessed data")
                
                # Create output directory for this dataset
                output_dir = self.base_output_dir / "features" / "batches" / dataset
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract features in batches
                features = self.feature_extractor.extract_features_batch(
                    image_paths=image_paths,
                    metadata_df=df,
                    output_dir=str(output_dir),
                    batch_size=self.batch_size
                )
                
                all_features[dataset] = features
                
                # Save final features
                final_dir = self.base_output_dir / "features" / "final" / dataset
                final_dir.mkdir(parents=True, exist_ok=True)
                
                for feature_type, feature_array in features.items():
                    if len(feature_array) > 0:
                        feature_file = final_dir / f"{feature_type}_features.npy"
                        np.save(feature_file, feature_array)
                        self.logger.info(f"{dataset} {feature_type}: {feature_array.shape} saved to {feature_file}")
                
                # Run feature analysis
                analysis = self.feature_extractor.get_feature_analysis(features)
                
                # Save analysis results
                analysis_file = final_dir / "feature_analysis.json"
                # Convert numpy arrays to lists for JSON serialization
                analysis_json = {}
                for ft, data in analysis.items():
                    if 'error' not in data:
                        analysis_json[ft] = {
                            'explained_variance_ratio': data['explained_variance_ratio'].tolist(),
                            'cumulative_variance': data['cumulative_variance'].tolist(),
                            'n_features': int(data['n_features'])
                        }
                    else:
                        analysis_json[ft] = data
                
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_json, f, indent=2)
                
                self.logger.info(f"{dataset}: Feature analysis saved to {analysis_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to extract features for {dataset}: {e}")
                continue
        
        elapsed = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed:.2f} seconds")
        
        return all_features
    
    def run_preaugmentation(self, 
                           metadata_dfs: Dict[str, pd.DataFrame],
                           balance_dataset: bool = True,
                           target_samples_per_class: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        NEW: Pre-generate and store all augmented images for GPU-optimized training.
        
        Args:
            metadata_dfs: Original metadata DataFrames from data loading
            balance_dataset: Whether to balance classes through augmentation
            target_samples_per_class: Target number of samples per class
            
        Returns:
            Dictionary of augmented metadata DataFrames
        """
        self.logger.info("="*50)
        self.logger.info("STEP 2: PRE-AUGMENTATION (GPU OPTIMIZED)")
        self.logger.info("="*50)
        
        start_time = time.time()
        augmented_dfs = {}
        
        for dataset, df in metadata_dfs.items():
            self.logger.info(f"Pre-augmenting dataset: {dataset}")
            
            try:
                # Create augmented output directory for this dataset
                augmented_dir = self.base_output_dir / "data" / "augmented" / dataset
                augmented_dir.mkdir(parents=True, exist_ok=True)
                
                if balance_dataset:
                    # Step 1: Create balanced augmentation strategy
                    augmentation_strategy, balanced_df = self.image_preprocessor.create_balanced_dataset(
                        metadata_df=df,
                        target_samples_per_class=target_samples_per_class
                    )
                    
                    self.logger.info(f"Augmentation strategy: {augmentation_strategy}")
                    
                    # Step 2: Pre-generate ALL augmented images
                    augmented_metadata = []
                    
                    for class_name, aug_methods in augmentation_strategy.items():
                        class_data = balanced_df[balanced_df['dx'] == class_name]
                        
                        self.logger.info(f"Pre-augmenting {len(class_data)} {class_name} images with {aug_methods}")
                        
                        for _, row in tqdm(class_data.iterrows(), 
                                         desc=f"Processing {class_name}", 
                                         total=len(class_data)):
                            
                            original_path = row['local_path']
                            image_name = row['image']
                            
                            # Generate all augmentations for this image
                            processed_images = self.image_preprocessor.preprocess_single_image(
                                original_path, 
                                augmentations=aug_methods
                            )
                            
                            # Save each augmented version
                            for processed_img, aug_method in zip(processed_images, aug_methods):
                                # Create filename for augmented image
                                if image_name.endswith('.jpg'):
                                    base_name = image_name[:-4]
                                else:
                                    base_name = image_name
                                
                                aug_filename = f"{base_name}_aug_{aug_method}.jpg"
                                aug_path = augmented_dir / aug_filename
                                
                                # Save augmented image
                                pil_img = Image.fromarray(processed_img.astype(np.uint8))
                                pil_img.save(aug_path)
                                
                                # Record metadata for augmented image
                                augmented_metadata.append({
                                    'original_image': image_name,
                                    'augmented_image': aug_filename,
                                    'local_path': str(aug_path),
                                    'dx': class_name,
                                    'augmentation': aug_method,
                                    'width': self.target_size[0],
                                    'height': self.target_size[1]
                                })
                    
                    # Create augmented DataFrame
                    augmented_df = pd.DataFrame(augmented_metadata)
                    
                else:
                    # No balancing - just basic preprocessing and minimal augmentation
                    augmented_metadata = []
                    
                    self.logger.info(f"Basic preprocessing for {len(df)} images")
                    
                    for _, row in tqdm(df.iterrows(), desc="Basic processing", total=len(df)):
                        original_path = row['local_path']
                        image_name = row['image']
                        label = row['dx']
                        
                        # Just basic preprocessing (no augmentation)
                        processed_images = self.image_preprocessor.preprocess_single_image(
                            original_path, 
                            augmentations=['rot0']  # Just original
                        )
                        
                        if processed_images:
                            processed_img = processed_images[0]
                            
                            # Create filename
                            if image_name.endswith('.jpg'):
                                base_name = image_name[:-4]
                            else:
                                base_name = image_name
                            
                            proc_filename = f"{base_name}_processed.jpg"
                            proc_path = augmented_dir / proc_filename
                            
                            # Save processed image
                            pil_img = Image.fromarray(processed_img.astype(np.uint8))
                            pil_img.save(proc_path)
                            
                            augmented_metadata.append({
                                'original_image': image_name,
                                'augmented_image': proc_filename,
                                'local_path': str(proc_path),
                                'dx': label,
                                'augmentation': 'rot0',
                                'width': self.target_size[0],
                                'height': self.target_size[1]
                            })
                    
                    augmented_df = pd.DataFrame(augmented_metadata)
                
                augmented_dfs[dataset] = augmented_df
                
                # Save augmented metadata
                metadata_file = self.base_output_dir / "data" / "metadata" / f"{dataset}_augmented.csv"
                augmented_df.to_csv(metadata_file, index=False)
                
                # Log class distribution
                if 'dx' in augmented_df.columns:
                    class_counts = augmented_df['dx'].value_counts()
                    self.logger.info(f"{dataset} augmented class distribution:\n{class_counts}")
                
                self.logger.info(f"{dataset}: {len(augmented_df)} augmented images pre-generated, metadata saved to {metadata_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to pre-augment {dataset}: {e}")
                continue
        
        elapsed = time.time() - start_time
        self.logger.info(f"Pre-augmentation completed in {elapsed:.2f} seconds")
        
        return augmented_dfs

    def run_complete_pipeline(self,
                            datasets: List[str] = ["isic_2019"],
                            max_images_per_dataset: Optional[int] = None,
                            apply_augmentation: bool = True,
                            balance_dataset: bool = True,
                            force_reload: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run the complete end-to-end pipeline.
        
        Args:
            datasets: List of datasets to process
            max_images_per_dataset: Limit for testing
            apply_augmentation: Whether to apply data augmentation
            balance_dataset: Whether to balance classes
            force_reload: Force re-download of data
            
        Returns:
            Dictionary of all extracted features
        """
        pipeline_start = time.time()

        self.logger.info("STARTING SKINVISION PIPELINE")
        self.logger.info(f"Datasets: {datasets}")
        self.logger.info(f"Max images per dataset: {max_images_per_dataset}")
        self.logger.info(f"Apply augmentation: {apply_augmentation}")
        self.logger.info(f"Balance dataset: {balance_dataset}")
        self.logger.info(f"Output directory: {self.base_output_dir}")
        
        try:
            # Step 1: Data Loading
            metadata_dfs = self.run_data_loading(
                datasets=datasets,
                max_images_per_dataset=max_images_per_dataset,
                force_reload=force_reload
            )
            
            if not metadata_dfs:
                raise Exception("No datasets loaded successfully")
            
            # Step 2: Image Preprocessing  
            preprocessed_dfs = self.run_image_preprocessing(
                metadata_dfs=metadata_dfs,
                apply_augmentation=apply_augmentation,
                balance_dataset=balance_dataset
            )
            
            if not preprocessed_dfs:
                raise Exception("No datasets preprocessed successfully")
            
            # Step 3: Feature Extraction
            all_features = self.run_feature_extraction(preprocessed_dfs)
            
            if not all_features:
                raise Exception("No features extracted successfully")
            
            # Generate final summary
            self._generate_pipeline_summary(metadata_dfs, preprocessed_dfs, all_features)
            
            elapsed = time.time() - pipeline_start
            self.logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds")
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"PIPELINE FAILED: {e}")
            raise
    
    def run_complete_pipeline_optimized(self,
                                  datasets: List[str] = ["isic_2019"],
                                  max_images_per_dataset: Optional[int] = None,
                                  apply_augmentation: bool = True,
                                  balance_dataset: bool = True,
                                  force_reload: bool = False,
                                  target_samples_per_class: int = 1000) -> Dict[str, Dict[str, np.ndarray]]:
        """GPU-optimized pipeline with pre-augmentation."""
        pipeline_start = time.time()

        self.logger.info("STARTING GPU-OPTIMIZED SKINVISION PIPELINE")
        
        try:
            # Step 1: Data Loading (download originals)
            metadata_dfs = self.run_data_loading(
                datasets=datasets,
                max_images_per_dataset=max_images_per_dataset,
                force_reload=force_reload
            )
            
            # Step 2: Pre-Augmentation (GPU-optimized)
            if apply_augmentation:
                augmented_dfs = self.run_preaugmentation(
                    metadata_dfs=metadata_dfs,
                    balance_dataset=balance_dataset,
                    target_samples_per_class=target_samples_per_class
                )
            else:
                augmented_dfs = self.run_preaugmentation(
                    metadata_dfs=metadata_dfs,
                    balance_dataset=False
                )
            
            # Step 3: Feature Extraction (from pre-augmented images)
            all_features = self.run_feature_extraction(augmented_dfs)
            
            # Generate summary
            self._generate_pipeline_summary(metadata_dfs, augmented_dfs, all_features)
            
            elapsed = time.time() - pipeline_start
            self.logger.info(f"GPU-OPTIMIZED PIPELINE COMPLETED in {elapsed:.2f} seconds")
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"PIPELINE FAILED: {e}")
            raise
    
    def _generate_pipeline_summary(self, 
                                 metadata_dfs: Dict[str, pd.DataFrame],
                                 preprocessed_dfs: Dict[str, pd.DataFrame], 
                                 all_features: Dict[str, Dict[str, np.ndarray]]):
        """Generate a summary report of the pipeline execution."""
        
        summary = {
            "pipeline_execution": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_size": self.target_size,
                "batch_size": self.batch_size,
                "storage_account": self.storage_account
            },
            "datasets": {}
        }
        
        for dataset in metadata_dfs.keys():
            dataset_summary = {
                "raw_images": len(metadata_dfs[dataset]) if dataset in metadata_dfs else 0,
                "preprocessed_images": len(preprocessed_dfs[dataset]) if dataset in preprocessed_dfs else 0,
                "features_extracted": {},
                "class_distribution": {}
            }
            
            # Feature summary
            if dataset in all_features:
                for ft, features in all_features[dataset].items():
                    if len(features) > 0:
                        dataset_summary["features_extracted"][ft] = {
                            "shape": list(features.shape),
                            "size_mb": round(features.nbytes / (1024 * 1024), 2)
                        }
            
            # Class distribution - handle different column names
            if dataset in preprocessed_dfs:
                df = preprocessed_dfs[dataset]
                diagnosis_col = None
                for col in ['label', 'dx', 'diagnosis']:  # 'label' is what preprocess_batch returns
                    if col in df.columns:
                        diagnosis_col = col
                        break
                
                if diagnosis_col:
                    class_counts = df[diagnosis_col].value_counts().to_dict()
                    dataset_summary["class_distribution"] = class_counts
            
            summary["datasets"][dataset] = dataset_summary
        
        # Save summary
        summary_file = self.base_output_dir / "results" / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Pipeline summary saved to {summary_file}")
        
        # Print summary to console
        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("="*60)
        
        for dataset, data in summary["datasets"].items():
            self.logger.info(f"\n{dataset.upper()}:")
            self.logger.info(f"  Raw images: {data['raw_images']}")
            self.logger.info(f"  Preprocessed images: {data['preprocessed_images']}")
            self.logger.info(f"  Features extracted: {len(data['features_extracted'])} types")
            
            for ft, info in data['features_extracted'].items():
                self.logger.info(f"    {ft}: {info['shape']} ({info['size_mb']} MB)")


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(description="SkinVision Pipeline Runner")
    parser.add_argument("--datasets", nargs="+", default=["isic_2019"], 
                       choices=["isic_2019", "ham10000"],
                       help="Datasets to process")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum images per dataset (for testing)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--target-size", nargs=2, type=int, default=[450, 450],
                       help="Target image size [width height]")
    parser.add_argument("--output-dir", type=str, default="../pipeline_output",
                       help="Output directory")
    parser.add_argument("--no-augmentation", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--no-balancing", action="store_true", 
                       help="Disable dataset balancing")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force re-download of data")
    
    # Add GPU optimization flag
    parser.add_argument("--gpu-optimized", action="store_true",
                       help="Use GPU-optimized pipeline with pre-augmentation")
    parser.add_argument("--target-samples", type=int, default=1000,
                       help="Target samples per class for balancing")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SkinVisionPipeline(
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        base_output_dir=args.output_dir
    )
    
    # Run pipeline
    try:
        if args.gpu_optimized:
            features = pipeline.run_complete_pipeline_optimized(
                datasets=args.datasets,
                max_images_per_dataset=args.max_images,
                apply_augmentation=not args.no_augmentation,
                balance_dataset=not args.no_balancing,
                force_reload=args.force_reload,
                target_samples_per_class=args.target_samples
            )
        else:
            # Use original pipeline
            features = pipeline.run_complete_pipeline(
                datasets=args.datasets,
                max_images_per_dataset=args.max_images,
                apply_augmentation=not args.no_augmentation,
                balance_dataset=not args.no_balancing,
                force_reload=args.force_reload
            )
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()