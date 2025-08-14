"""
Feature Extractor for ISIC 2019 SkinVision Project

This module handles feature extraction from preprocessed images including:
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Color Histograms (HSV)
- GLCM (Gray-Level Co-occurrence Matrix)
- Wavelet features
- Laplace features
- HSV Color features (mean, std, entropy)
- Circularity features (circularity, eccentricity, convexity)
- Contrast features (lesion vs surrounding skin)
- GPU-optimized batch processing

Combines logic from revised_feature_engineering.ipynb and extract_color_and_circularity_features.ipynb
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import glob
import gc
from tqdm import tqdm

# Computer vision and feature extraction libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops,
    hog
)

import pywt
import torch
import torchvision.transforms as T
from model_inference import EfficientNetB3SkinLesionClassifier, load_pretrained_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Handles feature extraction from dermatological images.
    Optimized for GPU processing and batch operations.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (450, 450)):
        """
        Initialize the Feature Extractor.
        
        Args:
            target_size: Expected size of input images (width, height)
        """
        self.target_size = target_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction parameters from your notebook
        self.lbp_points = 24
        self.lbp_radius = 3
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        # Feature types we extract - added new feature types
        # Feature types we extract - added EfficientNet
        # Feature types we extract - EfficientNet features are optional and not included by default
        self.feature_types = ['hog', 'lbp', 'color', 'glcm', 'wavelet', 'laplace', 'hsv_color', 'circularity', 'contrast']
    def extract_efficientnet_features(self, image: np.ndarray, model: Optional[torch.nn.Module] = None, device: str = 'cpu') -> np.ndarray:
        """
        Extract EfficientNet deep features from an image using a pretrained model.
        Args:
            image: Input image as numpy array (RGB, float32 [0,1])
            model: Pretrained EfficientNet model (if None, loads default)
            device: 'cpu' or 'cuda'
        Returns:
            Deep feature vector (numpy array)
        """
        try:
            import torchvision.transforms as T
            # Prepare image for EfficientNet
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((300, 300)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform((image * 255).astype(np.uint8)).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            if model is None:
                model = load_pretrained_model(device=device)
            model.eval()
            with torch.no_grad():
                features = model.extract_features(img_tensor)
                features = torch.flatten(features, 1)
                features_np = features.cpu().numpy().squeeze()
            return features_np
        except Exception as e:
            self.logger.error(f"Error extracting EfficientNet features: {e}")
            # Return zeros of expected size (1536 for B3)
            return np.zeros(1536)
    
    def load_image(self, image_path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load and prepare image for feature extraction."""
        if size is None:
            size = self.target_size
        
        try:
            # Use cv2 for consistency with preprocessor
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to float [0,1] range
            img = img.astype(np.float32) / 255.0
            
            # Convert grayscale to RGB if needed
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            
            # Resize using cv2 for consistency
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            return img
        
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def extract_hog_features(self, 
                           image: np.ndarray, 
                           pixels_per_cell: Tuple[int, int] = (16, 16),
                           cells_per_block: Tuple[int, int] = (2, 2), 
                           orientations: int = 4) -> np.ndarray:
        """Extract HOG features."""
        try:
            gray = rgb2gray(image)
            features = hog(gray,
                          orientations=orientations,
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          block_norm='L2-Hys',
                          feature_vector=True)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting HOG features: {e}")
            # Calculate expected size dynamically
            h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
            cells_per_row = h // pixels_per_cell[0]
            cells_per_col = w // pixels_per_cell[1]
            blocks_per_row = cells_per_row - cells_per_block[0] + 1
            blocks_per_col = cells_per_col - cells_per_block[1] + 1
            expected_size = orientations * cells_per_block[0] * cells_per_block[1] * blocks_per_row * blocks_per_col
            return np.zeros(max(expected_size, 1))
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP (Local Binary Patterns) features.
        From your revised_feature_engineering.ipynb notebook.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            LBP histogram feature vector
        """
        try:
            gray = rgb2gray(image)
            lbp = local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method='default')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), density=True)
            return hist
            
        except Exception as e:
            self.logger.error(f"Error extracting LBP features: {e}")
            return np.zeros(self.lbp_points + 2)
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract color histogram features in HSV space.
        From your revised_feature_engineering.ipynb notebook.
        
        Args:
            image: Input image as numpy array
            bins: Number of histogram bins per channel
            
        Returns:
            Concatenated HSV histogram features
        """
        try:
            hsv = rgb2hsv(image)
            
            # Extract histograms for each HSV channel
            h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1), density=True)[0]
            s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1), density=True)[0]
            v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1), density=True)[0]
            
            return np.concatenate([h_hist, s_hist, v_hist])
            
        except Exception as e:
            self.logger.error(f"Error extracting color features: {e}")
            return np.zeros(bins * 3)
    
    def extract_glcm_features(self, image: np.ndarray, levels: int = 64) -> np.ndarray:
        """Extract GLCM features with reduced levels for efficiency."""
        try:
            gray = rgb2gray(image)
            # Reduce to fewer gray levels for efficiency
            gray = (gray * (levels - 1)).astype(np.uint8)
            
            # Compute GLCM
            glcm = graycomatrix(gray,
                          distances=self.glcm_distances,
                          angles=self.glcm_angles,
                          levels=levels,  # Add levels parameter
                          symmetric=True,
                          normed=True)
            
            # Extract properties
            features = []
            for prop in self.glcm_props:
                features.extend(graycoprops(glcm, prop).flatten())
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting GLCM features: {e}")
            return np.zeros(len(self.glcm_props) * len(self.glcm_distances) * len(self.glcm_angles))
    
    def extract_wavelet_features(self, 
                               image: np.ndarray, 
                               wavelet: str = 'haar', 
                               level: int = 2, 
                               max_coeffs: int = 1000) -> np.ndarray:
        """
        Extract wavelet features using 2D wavelet decomposition.
        From your revised_feature_engineering.ipynb notebook.
        
        Args:
            image: Input image as numpy array
            wavelet: Wavelet type
            level: Decomposition level
            max_coeffs: Maximum number of coefficients to return
            
        Returns:
            Wavelet coefficient features
        """
        try:
            gray = rgb2gray(image)
            coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
            
            # Flatten all coefficients
            coeffs_flat = []
            for coeff in coeffs:
                if isinstance(coeff, tuple):
                    for arr in coeff:
                        coeffs_flat.extend(arr.ravel())
                else:
                    coeffs_flat.extend(coeff.ravel())
            
            # Cap the size and pad if necessary
            coeffs_array = np.array(coeffs_flat[:max_coeffs])
            if len(coeffs_array) < max_coeffs:
                coeffs_array = np.pad(coeffs_array, (0, max_coeffs - len(coeffs_array)), 'constant')
            
            return coeffs_array
            
        except Exception as e:
            self.logger.error(f"Error extracting wavelet features: {e}")
            return np.zeros(max_coeffs)
    
    def extract_laplace_features(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract Laplacian edge features.
        From your revised_feature_engineering.ipynb notebook.
        
        Args:
            image: Input image as numpy array
            bins: Number of histogram bins
            
        Returns:
            Laplacian histogram features
        """
        try:
            gray = rgb2gray(image)
            if gray.dtype != np.uint8:  # Check if scaling is necessary
                gray = (gray * 255).astype(np.uint8)  # Convert to uint8
            laplace = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            
            # Handle case where all values are zero
            if np.max(laplace) == 0:
                hist = np.zeros(bins)
            else:
                hist, _ = np.histogram(laplace.ravel(), bins=bins, range=(0, np.max(laplace)), density=True)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error extracting Laplace features: {e}")
            return np.zeros(bins)
    
    def extract_hsv_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HSV color features from lesion area using Otsu thresholding.
        From extract_color_and_circularity_features.ipynb Step 2.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            HSV feature vector: [hsv_mean (3), hsv_std (3), hsv_entropy (3)] = 9 features
        """
        try:
            # Convert to OpenCV format (BGR) and then to HSV
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            img_hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
            
            # Step 1: Perform segmentation using Otsu thresholding
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is white
            if np.sum(thresh == 255) > np.sum(thresh == 0):
                thresh = cv2.bitwise_not(thresh)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return np.zeros(9)  # Return zeros if no contours found
            
            # Find the largest contour (assuming it's the lesion)
            largest_contour = max(contours, key=cv2.contourArea)
            
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Step 2: Calculate HSV Features
            hsv_pixels = img_hsv[mask == 255]
            
            if len(hsv_pixels) == 0:
                return np.zeros(9)
            
            hsv_mean = np.mean(hsv_pixels, axis=0)
            hsv_std = np.std(hsv_pixels, axis=0)
            
            # Calculate entropy for each HSV channel
            hsv_entropy = []
            for i in range(3):
                hist, _ = np.histogram(hsv_pixels[:, i], bins=32, range=(0, 256), density=True)
                # Avoid log(0) by adding small epsilon
                hist = hist + 1e-8
                entropy = -np.sum(hist * (np.log(hist) / np.log(2)))
                hsv_entropy.append(entropy)
            
            hsv_entropy = np.array(hsv_entropy)
            
            # Combine all features: mean (3) + std (3) + entropy (3) = 9 features
            features = np.concatenate([hsv_mean, hsv_std, hsv_entropy])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting HSV color features: {e}")
            return np.zeros(9)
    
    def extract_circularity_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract circularity features from lesion area.
        From extract_color_and_circularity_features.ipynb Step 3.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Circularity feature vector: [circularity, eccentricity, convexity] = 3 features
        """
        try:
            # Convert to OpenCV format
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # Perform segmentation using Otsu thresholding
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is white
            if np.sum(thresh == 255) > np.sum(thresh == 0):
                thresh = cv2.bitwise_not(thresh)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return np.zeros(3)
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Step 3: Calculate circularity features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return np.zeros(3)
            
            # 1. Circularity = 4 * pi * area / perimeter^2
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # 2. Eccentricity from fitted ellipse
            eccentricity = 0
            if len(largest_contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(largest_contour)
                    (center, axes, angle) = ellipse
                    major, minor = max(axes), min(axes)
                    if major > 0:
                        eccentricity = np.sqrt(1 - (minor / major) ** 2)
                except cv2.error as e:
                    logging.warning(f"Failed to fit ellipse: {e}")
                    eccentricity = 0
            
            # 3. Convexity = Convex Hull Perimeter / Perimeter
            convex_hull = cv2.convexHull(largest_contour)
            convex_perimeter = cv2.arcLength(convex_hull, True)
            convexity = convex_perimeter / perimeter if perimeter > 0 else 0
            
            features = np.array([circularity, eccentricity, convexity])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting circularity features: {e}")
            return np.zeros(3)
    
    def extract_contrast_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract contrast features between lesion and surrounding skin.
        From extract_color_and_circularity_features.ipynb Step 4.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Contrast feature vector: [contrast_hsv (3), contrast_hsv_euclidean (1)] = 4 features
        """
        try:
            # Convert to OpenCV format
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            img_hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
            
            # Perform segmentation using Otsu thresholding
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is white
            if np.sum(thresh == 255) > np.sum(thresh == 0):
                thresh = cv2.bitwise_not(thresh)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return np.zeros(4)
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Get lesion HSV values
            hsv_pixels = img_hsv[mask == 255]
            if len(hsv_pixels) == 0:
                return np.zeros(4)
            
            hsv_mean = np.mean(hsv_pixels, axis=0)
            
            # Step 4: Calculate contrast features
            # Create dilated mask to find surrounding skin
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            surrounding_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(mask))
            
            surrounding_hsv = img_hsv[surrounding_mask == 255]
            
            if len(surrounding_hsv) > 0:
                surrounding_hsv_mean = np.mean(surrounding_hsv, axis=0)
                contrast_hsv = np.abs(hsv_mean - surrounding_hsv_mean)
                contrast_hsv_euclidean = np.linalg.norm(contrast_hsv)
            else:
                contrast_hsv = np.array([0, 0, 0])
                contrast_hsv_euclidean = 0
            
            # Combine features: contrast_hsv (3) + euclidean distance (1) = 4 features
            features = np.concatenate([contrast_hsv, [contrast_hsv_euclidean]])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting contrast features: {e}")
            return np.zeros(4)
    
    def extract_all_features_single(self, image_path: str, efficientnet_model: Optional[torch.nn.Module] = None, device: str = 'cpu') -> Dict[str, np.ndarray]:
        """
        Extract all feature types from a single image, including EfficientNet.
        Args:
            image_path: Path to image file
            efficientnet_model: Pretrained EfficientNet model (optional)
            device: 'cpu' or 'cuda'
        Returns:
            Dictionary with feature type as key and feature vector as value
        """
        image = self.load_image(image_path)
        features = {
            'hog': self.extract_hog_features(image),
            'lbp': self.extract_lbp_features(image),
            'color': self.extract_color_histogram(image),
            'glcm': self.extract_glcm_features(image),
            'wavelet': self.extract_wavelet_features(image),
            'laplace': self.extract_laplace_features(image),
            'hsv_color': self.extract_hsv_color_features(image),
            'circularity': self.extract_circularity_features(image),
            'contrast': self.extract_contrast_features(image),
            'efficientnet': self.extract_efficientnet_features(image, model=efficientnet_model, device=device)
        }
        return features
    
    def extract_features_batch(self, 
                             image_paths: List[str],
                             metadata_df: pd.DataFrame,
                             output_dir: str,
                             batch_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Extract features from a batch of images with batched processing.
        From your revised_feature_engineering.ipynb batched logic.
        
        Args:
            image_paths: List of image file paths
            metadata_df: DataFrame with image metadata
            output_dir: Directory to save batch results
            batch_size: Number of images to process per batch
            
        Returns:
            Dictionary with concatenated features for all images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature lists
        feature_lists = {ft: [] for ft in self.feature_types}
        num_images = len(image_paths)
        
        # Process images in batches
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            batch_paths = image_paths[start:end]
            
            self.logger.info(f"Processing feature batch {start} to {end}")
            
            # Initialize batch feature lists
            batch_features = {ft: [] for ft in self.feature_types}
            
            # Process each image in the batch
            for image_path in tqdm(batch_paths, desc=f"Extracting features batch {start//batch_size + 1}"):
                try:
                    # Extract features for this image
                    img_features = self.extract_all_features_single(image_path)
                    
                    # Add to batch lists
                    for ft in self.feature_types:
                        batch_features[ft].append(img_features[ft])
                        
                    # Clear image from memory immediately
                    del img_features
                    
                except Exception as e:
                    self.logger.error(f"Failed to extract features from {image_path}: {e}")
                    # Use target size for failed images - ensure it's (height, width) for consistency
                    img_shape = (self.target_size[1], self.target_size[0])  # Convert (w,h) to (h,w)

                    for ft in self.feature_types:
                        batch_features[ft].append(self.get_zero_features(ft, img_shape))
            
            # Convert batch lists to arrays and save
            for ft in self.feature_types:
                if batch_features[ft]:  # Check if list is not empty
                    try:
                        # Check if all features have the same shape
                        feature_shapes = [f.shape[0] for f in batch_features[ft]]
                        if len(set(feature_shapes)) > 1:
                            # Different shapes detected - pad to max size
                            max_size = max(feature_shapes)
                            self.logger.warning(f"{ft} features have inconsistent shapes: {set(feature_shapes)}. Padding to {max_size}")
                            
                            padded_features = []
                            for feature_vec in batch_features[ft]:
                                if len(feature_vec) < max_size:
                                    padded_vec = np.pad(feature_vec, (0, max_size - len(feature_vec)), 'constant')
                                    padded_features.append(padded_vec)
                                else:
                                    padded_features.append(feature_vec)
                            
                            batch_array = np.array(padded_features)
                            # Use padded features for the overall list too
                            feature_lists[ft].extend(padded_features)
                        else:
                            # All features have same shape - normal processing
                            batch_array = np.array(batch_features[ft])
                            # Add to overall feature lists
                            feature_lists[ft].extend(batch_features[ft])
            
                        # Save batch to disk
                        batch_filename = os.path.join(output_dir, f"{ft}_{start}_{end}.npy")
                        np.save(batch_filename, batch_array)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to convert {ft} features to array: {e}")
                        # Skip this feature type for this batch
                        continue
            
            self.logger.info(f"Saved feature batch {start}-{end}")
            
            # Memory cleanup after each batch
            gc.collect()
        
        # Convert all feature lists to arrays
        final_features = {}
        for ft in self.feature_types:
            if feature_lists[ft]:
                final_features[ft] = np.array(feature_lists[ft])
                self.logger.info(f"{ft}: extracted {final_features[ft].shape}")
            else:
                self.logger.warning(f"No {ft} features extracted")
                final_features[ft] = np.array([])
        
        return final_features
    
    def load_and_concatenate_features(self, output_dir: str) -> Dict[str, np.ndarray]:
        """
        Load and concatenate features from batch files.
        From your revised_feature_engineering.ipynb load logic.
        
        Args:
            output_dir: Directory containing batch feature files
            
        Returns:
            Dictionary with concatenated features
        """
        features = {}
        
        for ft in self.feature_types:
            # Find all batch files for this feature type
            pattern = os.path.join(output_dir, f"{ft}_*.npy")
            files = sorted(glob.glob(pattern))
            
            if files:
                # Load and concatenate all batches
                arrays = [np.load(f) for f in files]
                features[ft] = np.concatenate(arrays, axis=0)
                self.logger.info(f"{ft}: loaded {len(files)} batches, shape {features[ft].shape}")
            else:
                self.logger.warning(f"No batch files found for {ft} features")
                features[ft] = np.array([])
        
        return features
    
    def get_feature_analysis(self, features: Dict[str, np.ndarray], n_components: int = 10) -> Dict[str, Dict]:
        """
        Analyze feature importance using PCA.
        From your revised_feature_engineering.ipynb PCA analysis.
        
        Args:
            features: Dictionary of feature arrays
            n_components: Number of PCA components to analyze
            
        Returns:
            Dictionary with PCA analysis results
        """
        analysis = {}
        
        for ft, feature_array in features.items():
            if len(feature_array) > 0 and feature_array.shape[0] > n_components:
                try:
                    # Fit PCA
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(feature_array)
                    
                    analysis[ft] = {
                        'explained_variance_ratio': pca.explained_variance_ratio_,
                        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                        'n_features': feature_array.shape[1],
                        'pca_components': X_pca
                    }
                    
                    self.logger.info(f"{ft}: {analysis[ft]['cumulative_variance'][-1]:.3f} "
                                   f"variance explained by {n_components} components")
                    
                except Exception as e:
                    self.logger.error(f"PCA analysis failed for {ft}: {e}")
                    analysis[ft] = {'error': str(e)}
            else:
                analysis[ft] = {'error': 'Insufficient data for PCA'}
        
        return analysis
    
    def plot_feature_analysis(self, analysis: Dict[str, Dict]):
        """
        Plot PCA analysis results.
        From your revised_feature_engineering.ipynb plotting logic.
        
        Args:
            analysis: PCA analysis results from get_feature_analysis
        """
        plt.figure(figsize=(15, 5))
        colors = ['b-', 'm-', 'g-', 'r-', 'c-', 'y-']
        
        for i, (ft, data) in enumerate(analysis.items()):
            if 'cumulative_variance' in data:
                plt.plot(range(1, len(data['cumulative_variance']) + 1), 
                        data['cumulative_variance'], 
                        colors[i % len(colors)], 
                        label=f"{ft.upper()} ({data['n_features']} features)")
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Analysis: Cumulative Explained Variance by Feature Type')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_expected_feature_sizes(self) -> Dict[str, int]:
        """Get expected feature sizes for each feature type."""
        return {
            'hog': None,  # Dynamic based on image size
            'lbp': self.lbp_points + 2,  # 24 + 2 = 26
            'color': 32 * 3,  # 32 bins × 3 HSV channels = 96
            'glcm': len(self.glcm_props) * len(self.glcm_distances) * len(self.glcm_angles),  # 6×3×4 = 72
            'wavelet': 1000,  # Fixed at 1000
            'laplace': 32,  # Fixed at 32 bins
            'hsv_color': 9,  # hsv_mean (3) + hsv_std (3) + hsv_entropy (3)
            'circularity': 3,  # circularity + eccentricity + convexity
            'contrast': 4,  # contrast_hsv (3) + euclidean distance (1)
            'efficientnet': 1536
        }

    def get_zero_features(self, feature_type: str, image_shape: Tuple[int, int] = None) -> np.ndarray:
        """Get zero array for failed feature extraction."""
        expected_sizes = self.get_expected_feature_sizes()
        
        if feature_type == 'hog' and image_shape:
            # Use same parameters as extract_hog_features
            h, w = image_shape
            pixels_per_cell = (16, 16)  # Match extract_hog_features
            cells_per_block = (2, 2)    # Match extract_hog_features
            orientations = 4            # Match extract_hog_features
            
            cells_per_row = h // pixels_per_cell[0]
            cells_per_col = w // pixels_per_cell[1]
            blocks_per_row = max(1, cells_per_row - cells_per_block[0] + 1)
            blocks_per_col = max(1, cells_per_col - cells_per_block[1] + 1)
            expected_size = orientations * cells_per_block[0] * cells_per_block[1] * blocks_per_row * blocks_per_col
            return np.zeros(max(expected_size, 1))
        else:
            return np.zeros(expected_sizes[feature_type])
    
def main():
    """
    Example usage of the FeatureExtractor
    """
    # Initialize feature extractor
    # Initialize feature extractor
    extractor = FeatureExtractor(target_size=(450, 450))
    # Load EfficientNet model once for batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    efficientnet_model = load_pretrained_model(device=device)
    # Example: Load preprocessed metadata
    metadata_path = '../images/processed/preprocessed_metadata.csv'
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        # Get image paths (use first 100 for testing)
        image_paths = metadata_df['local_path'].tolist()[:100]
        # Extract features in batches
        features = extractor.extract_features_batch(
            image_paths=image_paths,
            metadata_df=metadata_df,
            output_dir='../features/batches',
            batch_size=20  # Smaller batches for testing
        )
        # Analyze features
        analysis = extractor.get_feature_analysis(features)
        # Plot analysis
        extractor.plot_feature_analysis(analysis)
        # Print summary
        for ft, feature_array in features.items():
            if len(feature_array) > 0:
                print(f"{ft.upper()}: {feature_array.shape} features extracted")
            else:
                print(f"{ft.upper()}: No features extracted")
    else:
        print("No preprocessed metadata found. Run data_loader.py and image_preprocessor.py first!")


if __name__ == "__main__":
    main()