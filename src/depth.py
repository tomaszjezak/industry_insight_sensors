"""
Monocular depth estimation using MiDaS.
Estimates relative depth from a single RGB image.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DepthEstimator:
    """Estimates depth from monocular images using MiDaS."""
    
    # Available MiDaS models (smaller = faster, larger = more accurate)
    MODELS = {
        'small': 'MiDaS_small',      # ~20MB, fastest
        'hybrid': 'DPT_Hybrid',       # ~120MB, good balance
        'large': 'DPT_Large',         # ~300MB, most accurate
    }
    
    def __init__(self, model_type: str = 'small', device: str = None):
        """
        Initialize depth estimator.
        
        Args:
            model_type: One of 'small', 'hybrid', 'large'.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.transform = None
        
        if HAS_TORCH:
            self._init_model()
        else:
            print("[!] PyTorch not available. Depth estimation will use fallback.")
    
    def _init_model(self):
        """Load MiDaS model from torch hub."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = self.MODELS.get(self.model_type, 'MiDaS_small')
        
        print(f"[*] Loading MiDaS model '{model_name}' on {self.device}...")
        
        try:
            # Load model from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            
            if self.model_type == 'small':
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform
            
            print(f"[+] MiDaS model loaded successfully")
            
        except Exception as e:
            print(f"[!] Failed to load MiDaS: {e}")
            print("[*] Will use fallback depth estimation")
            self.model = None
    
    def estimate(self, image: np.ndarray) -> dict:
        """
        Estimate depth from an image.
        
        Args:
            image: BGR image (OpenCV format).
        
        Returns:
            dict with:
                - 'depth_map': Relative depth map (H, W), higher = farther
                - 'depth_normalized': Normalized 0-1 depth map
                - 'depth_colored': Colored depth visualization (BGR)
                - 'method': 'midas' or 'fallback'
        """
        if self.model is not None and HAS_TORCH:
            return self._estimate_midas(image)
        else:
            return self._estimate_fallback(image)
    
    def _estimate_midas(self, image: np.ndarray) -> dict:
        """Estimate depth using MiDaS model."""
        import torch
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform for model
        input_batch = self.transform(image_rgb).to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth (closer = higher value)
        # Invert so higher = farther (more intuitive for pile height)
        depth_map = depth_map.max() - depth_map
        
        # Normalize to 0-1
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Create colored visualization
        depth_colored = self._colorize_depth(depth_normalized)
        
        return {
            'depth_map': depth_map,
            'depth_normalized': depth_normalized,
            'depth_colored': depth_colored,
            'method': 'midas',
        }
    
    def _estimate_fallback(self, image: np.ndarray) -> dict:
        """
        Fallback depth estimation using image gradients and heuristics.
        Assumes debris piles are roughly convex and elevated from ground.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method: Combine multiple cues
        
        # 1. Vertical gradient (things lower in image are closer - perspective)
        vertical_grad = np.linspace(0, 1, h).reshape(-1, 1)
        vertical_grad = np.tile(vertical_grad, (1, w))
        
        # 2. Texture density (more texture = closer, more detail visible)
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(texture)
        texture = cv2.GaussianBlur(texture, (31, 31), 0)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        
        # 3. Brightness (often darker in shadows/recesses)
        brightness = gray.astype(float) / 255.0
        brightness = cv2.GaussianBlur(brightness, (31, 31), 0)
        
        # 4. Edge density (edges often indicate depth discontinuities)
        edges = cv2.Canny(gray, 50, 150).astype(float)
        edge_density = cv2.GaussianBlur(edges, (51, 51), 0)
        edge_density = edge_density / (edge_density.max() + 1e-8)
        
        # Combine cues (weighted average)
        # Higher texture + lower position = likely pile (closer/higher)
        depth_map = (
            0.3 * (1 - vertical_grad) +  # Lower in image = closer
            0.3 * texture +              # More texture = closer
            0.2 * (1 - brightness) +     # Darker might be recessed
            0.2 * edge_density           # Edges indicate structure
        )
        
        # Smooth the result
        depth_map = cv2.GaussianBlur(depth_map.astype(np.float32), (21, 21), 0)
        
        # Normalize
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Colorize
        depth_colored = self._colorize_depth(depth_normalized)
        
        return {
            'depth_map': depth_map,
            'depth_normalized': depth_normalized,
            'depth_colored': depth_colored,
            'method': 'fallback',
        }
    
    def _colorize_depth(self, depth_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized depth to colored visualization."""
        # Use TURBO colormap (good for depth visualization)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        return colored
    
    def get_height_map(
        self,
        depth_result: dict,
        mask: np.ndarray,
        camera_height_m: float = 3.0,
        fov_deg: float = 90.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Convert relative depth to estimated real-world height map.
        
        Args:
            depth_result: Output from estimate().
            mask: Binary mask of the pile area.
            camera_height_m: Assumed camera height in meters.
            fov_deg: Camera field of view in degrees.
        
        Returns:
            Tuple of (height_map, stats_dict)
        """
        depth_norm = depth_result['depth_normalized']
        h, w = depth_norm.shape
        
        # Get ground plane estimate (min depth in non-pile areas)
        non_pile_mask = (mask == 0)
        if non_pile_mask.any():
            ground_depth = np.percentile(depth_norm[non_pile_mask], 10)
        else:
            ground_depth = depth_norm.min()
        
        # Height = depth difference from ground, scaled by camera height
        # This is an approximation assuming roughly overhead view
        height_map = (depth_norm - ground_depth) * camera_height_m
        height_map = np.clip(height_map, 0, camera_height_m)
        
        # Apply mask
        height_map_masked = height_map * (mask > 0).astype(float)
        
        # Calculate stats within pile
        pile_heights = height_map_masked[mask > 0]
        if len(pile_heights) > 0:
            stats = {
                'min_height_m': float(pile_heights.min()),
                'max_height_m': float(pile_heights.max()),
                'mean_height_m': float(pile_heights.mean()),
                'std_height_m': float(pile_heights.std()),
            }
        else:
            stats = {
                'min_height_m': 0.0,
                'max_height_m': 0.0,
                'mean_height_m': 0.0,
                'std_height_m': 0.0,
            }
        
        return height_map_masked, stats


def estimate_depth(image_path: str | Path, model_type: str = 'small') -> dict:
    """
    Convenience function to estimate depth for a single image.
    
    Args:
        image_path: Path to image file.
        model_type: MiDaS model type.
    
    Returns:
        Depth estimation result dict.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    estimator = DepthEstimator(model_type=model_type)
    return estimator.estimate(image)


if __name__ == "__main__":
    from .scraper import list_available_images, DATA_DIR
    
    images = list_available_images()
    if images:
        print(f"[*] Testing depth estimation on: {images[0]}")
        
        estimator = DepthEstimator(model_type='small')
        image = cv2.imread(str(images[0]))
        result = estimator.estimate(image)
        
        print(f"[+] Method: {result['method']}")
        print(f"[+] Depth map shape: {result['depth_map'].shape}")
        print(f"[+] Depth range: {result['depth_map'].min():.2f} - {result['depth_map'].max():.2f}")
        
        # Save visualization
        output_path = DATA_DIR / "depth_test.jpg"
        cv2.imwrite(str(output_path), result['depth_colored'])
        print(f"[+] Saved depth visualization to: {output_path}")
    else:
        print("[!] No images available. Run scraper.py first.")
