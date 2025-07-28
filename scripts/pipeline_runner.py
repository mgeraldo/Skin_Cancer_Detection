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
            force_reload: Force re-download even if files exist
            
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
                    # Use the correct method name
                    df = self.data_loader.prepare_isic_dataset(
                        local_dir=str(self.base_output_dir / "data" / "raw"),
                        max_images=max_images_per_dataset,
                        batch_size=self.batch_size
                    )
                    # Rename dx column to diagnosis for consistency
                    if 'dx' in df.columns:
                        df = df.rename(columns={'dx': 'diagnosis'})
                        
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
                
                # Run preprocessing - use correct method names
                if apply_augmentation and balance_dataset:
                    # Full preprocessing with balanced augmentation
                    preprocessed_df = self.image_preprocessor.create_balanced_dataset(
                        df=df,
                        output_dir=str(output_dir),
                        batch_size=self.batch_size
                    )
                else:
                    # Standard preprocessing - use correct method name
                    preprocessed_df = self.image_preprocessor.preprocess_batch(
                        image_paths=df['local_path'].tolist(),
                        metadata_df=df,
                        output_dir=str(output_dir),
                        batch_size=self.batch_size,
                        apply_augmentations=apply_augmentation
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
                # Get image paths - handle different possible column names
                if 'local_path' in df.columns:
                    image_paths = df['local_path'].tolist()
                elif 'output_path' in df.columns:
                    image_paths = df['output_path'].tolist()
                else:
                    # Try any path-like column
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
        
        self.logger.info("🚀 STARTING SKINVISION PIPELINE")
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
            self.logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds")
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"❌ PIPELINE FAILED: {e}")
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
                for col in ['diagnosis', 'dx', 'label']:
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
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SkinVisionPipeline(
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        base_output_dir=args.output_dir
    )
    
    # Run pipeline
    try:
        features = pipeline.run_complete_pipeline(
            datasets=args.datasets,
            max_images_per_dataset=args.max_images,
            apply_augmentation=not args.no_augmentation,
            balance_dataset=not args.no_balancing,
            force_reload=args.force_reload
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()