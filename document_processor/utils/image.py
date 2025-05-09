import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class Image:
    """
    Image processing utilities for document preprocessing.
    
    This class provides methods for enhancing images before OCR and extraction.
    """
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            NumPy array containing the image
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> str:
        """
        Save an image to file.
        
        Args:
            image: NumPy array containing the image
            output_path: Path where to save the image
            
        Returns:
            Path to the saved image
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save image using OpenCV
        cv2.imwrite(output_path, image)
        
        return output_path
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Deskew (straighten) an image.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if not contours:
            logger.warning("No contours found for deskewing")
            return image
        
        # Use largest contour
        largest_contour = contours[0]
        
        # Find minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        
        # Ignore small angles
        if abs(angle) < 0.5:
            logger.info("Skipping deskew, angle too small")
            return image
        
        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """
        Remove noise from an image.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Apply non-local means denoising to color image
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Apply non-local means denoising to grayscale image
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, use_clahe: bool = True, clip_limit: float = 2.0,
                         tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Enhance contrast in an image.
        
        Args:
            image: Input image
            use_clahe: Whether to use CLAHE for adaptive histogram equalization
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of tiles for CLAHE
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            if use_clahe:
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                cl = clahe.apply(l)
            else:
                # Apply regular histogram equalization
                cl = cv2.equalizeHist(l)
            
            # Merge channels
            merged = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            if use_clahe:
                # Apply CLAHE to grayscale image
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                return clahe.apply(image)
            else:
                # Apply regular histogram equalization
                return cv2.equalizeHist(image)
    
    @staticmethod
    def remove_background(image: np.ndarray) -> np.ndarray:
        """
        Remove background from an image.
        
        Args:
            image: Input image
            
        Returns:
            Image with background removed
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Create a copy of the original image
        result = image.copy()
        
        if len(image.shape) == 3:
            # Create a white background
            white_bg = np.ones_like(result) * 255
            
            # Merge result with white background using the mask
            result = np.where(thresh[:, :, np.newaxis] == 0, white_bg, result)
        else:
            # For grayscale, simpler approach
            result = np.where(thresh == 0, 255, result)
        
        return result
    
    @staticmethod
    def binarize(image: np.ndarray, adaptive: bool = True) -> np.ndarray:
        """
        Convert image to binary (black and white).
        
        Args:
            image: Input image
            adaptive: Whether to use adaptive thresholding
            
        Returns:
            Binarized image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if adaptive:
            # Apply adaptive thresholding
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        else:
            # Apply Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary


class PreprocessingPipeline:
    """
    Pipeline for preprocessing document images before OCR and extraction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessing pipeline with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enable", True)
        self.methods = config.get("methods", [])
        self.clahe_params = config.get("clahe", {})
    
    def preprocess(self, file_path: str) -> str:
        """
        Preprocess a document image.
        
        Args:
            file_path: Path to the document image
            
        Returns:
            Path to the preprocessed image
        """
        if not self.enabled:
            logger.info(f"Preprocessing disabled, skipping for {file_path}")
            return file_path
        
        try:
            # Check if the file is an image
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                logger.warning(f"File is not an image: {file_path}")
                return file_path
            
            # Load the image
            image = Image.load_image(file_path)
            
            # Apply preprocessing methods in sequence
            for method in self.methods:
                if method == "deskew":
                    image = Image.deskew(image)
                elif method == "denoise":
                    image = Image.denoise(image)
                elif method == "contrast_enhancement":
                    clip_limit = self.clahe_params.get("clip_limit", 2.0)
                    tile_grid_size_tuple = tuple(self.clahe_params.get("tile_grid_size", (8, 8)))
                    image = Image.enhance_contrast(image, use_clahe=True, 
                                                  clip_limit=clip_limit,
                                                  tile_grid_size=tile_grid_size_tuple)
                elif method == "remove_background":
                    image = Image.remove_background(image)
                elif method == "binarize":
                    image = Image.binarize(image, adaptive=True)
            
            # Save the preprocessed image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                preprocessed_path = temp_file.name
            
            Image.save_image(image, preprocessed_path)
            
            logger.info(f"Preprocessed image saved to {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return file_path