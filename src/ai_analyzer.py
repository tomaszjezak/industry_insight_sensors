"""
Multimodal AI analyzer for recycling facility images.
Uses GPT-4V or Claude 3 to detect, outline, and analyze recycling materials.
"""

import numpy as np
import cv2
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False


def _discover_gemini_model(api_key: str, preferred_model: str = None) -> str:
    """
    Discover available Gemini models with vision support.
    
    Args:
        api_key: Google API key
        preferred_model: Preferred model name to try first
    
    Returns:
        Model name that works, or raises ValueError if none found
    """
    # Priority list of models to try (in order of preference)
    model_priority = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro',
        'gemini-2.0-flash',
        'gemini-2.0-flash-exp',
    ]
    
    # If preferred model is specified, try it first
    if preferred_model:
        model_priority.insert(0, preferred_model)
    
    genai.configure(api_key=api_key)
    
    # Try to list available models
    available_models = []
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                # Check if model supports vision (has input token limit > 0)
                model_name = model.name.split('/')[-1]  # Extract just the model name
                available_models.append(model_name)
    except Exception as e:
        print(f"[!] Could not list models: {e}. Trying fallback chain...")
    
    # Try each model in priority order
    for model_name in model_priority:
        # If we have a list of available models, check if this one is available
        if available_models:
            # Check if model name matches (handle variations)
            for avail in available_models:
                if model_name in avail or avail in model_name:
                    try:
                        # Test if we can create a GenerativeModel with it
                        test_model = genai.GenerativeModel(avail)
                        print(f"[+] Using Gemini model: {avail}")
                        return avail
                    except Exception as e:
                        continue
        else:
            # No model list available, try directly
            try:
                test_model = genai.GenerativeModel(model_name)
                print(f"[+] Using Gemini model: {model_name}")
                return model_name
            except Exception as e:
                # Model not available, try next
                continue
    
    # If we got here, none of the models worked
    raise ValueError(
        f"No working Gemini model found. Tried: {', '.join(model_priority)}. "
        f"Available models: {', '.join(available_models) if available_models else 'could not list'}. "
        f"Please check your API key and try specifying a model name manually."
    )


class AIRecyclingAnalyzer:
    """
    Uses multimodal AI (GPT-4V or Claude 3) to analyze recycling facility images.
    
    Detects and outlines individual recycling piles/objects, estimates:
    - Weight/tonnage for each object
    - Volume for each object
    - Material composition for each object
    """
    
    def __init__(
        self,
        provider: str = 'openai',  # 'openai', 'anthropic', or 'google'
        model: str = None,
        api_key: str = None,
    ):
        """
        Initialize AI analyzer.
        
        Args:
            provider: 'openai' for GPT-4V, 'anthropic' for Claude 3, or 'google' for Gemini (FREE!)
            model: Model name (defaults: 'gpt-4o' for OpenAI, 'claude-3-5-sonnet-20241022' for Anthropic, 'gemini-1.5-pro' for Google)
            api_key: API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY env var)
        """
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == 'openai':
            if not HAS_OPENAI:
                raise ImportError("openai package required. Install: pip install openai")
            self.model = model or 'gpt-4o'
            if api_key:
                openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key or None)
        elif self.provider == 'anthropic':
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package required. Install: pip install anthropic")
            self.model = model or 'claude-3-5-sonnet-20241022'
            self.client = anthropic.Anthropic(api_key=api_key or None)
        elif self.provider == 'google':
            if not HAS_GOOGLE:
                raise ImportError("google-generativeai package required. Install: pip install google-generativeai")
            
            # Discover and use available Gemini model
            try:
                self.model = _discover_gemini_model(api_key, model)
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
            except ValueError as e:
                # If discovery fails, provide helpful error message
                raise ValueError(
                    f"Failed to initialize Gemini model: {e}\n"
                    f"Try specifying a model name manually in api_key.txt: google:API_KEY:MODEL_NAME"
                )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'google'")
    
    def analyze(
        self,
        image: np.ndarray,
        camera_height_m: float = 4.0,
    ) -> Dict:
        """
        Analyze recycling facility image using multimodal AI.
        
        Args:
            image: BGR image (OpenCV format)
            camera_height_m: Camera height in meters (for scale estimation)
        
        Returns:
            Dict with detected objects, outlines, estimates
        """
        # Convert BGR to RGB for API
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to base64
        pil_image = Image.fromarray(image_rgb)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        h, w = image.shape[:2]
        
        # Create detailed prompt with specific guidance
        prompt = f"""Analyze this recycling facility image taken from Google Street View (camera height approximately {camera_height_m}m).

CRITICAL: You must ONLY detect actual recycling MATERIALS - compressed bales, material stacks, debris piles, and containers WITH materials inside.

WHAT TO DETECT (ONLY these):
1. COMPRESSED BALES: Rectangular blocks of compressed recyclable materials (paper, plastic, cardboard) - typically stacked in rows
2. MATERIAL STACKS: Piles of loose or semi-compacted recycling materials (concrete, wood, metal scrap)
3. CONTAINERS WITH MATERIALS: Bins, dumpsters, or containers that are FILLED with visible recycling materials
4. DEBRIS PILES: Accumulated piles of construction/demolition debris with visible 3D volume

WHAT TO ABSOLUTELY EXCLUDE (DO NOT DETECT):
- FENCES, GATES, CHAIN-LINK BARRIERS (even if they have materials behind them)
- BUILDINGS, WALLS, STRUCTURES (the facility buildings themselves)
- ROADS, PAVEMENT, EMPTY GROUND, ASPHALT
- VEHICLES (trucks, cars, forklifts)
- SKY, CLOUDS, VEGETATION (trees, bushes, grass)
- SIGNS, LOGOS, TEXT ON BUILDINGS
- EMPTY SPACES, SHADOWS, REFLECTIONS
- ANY FLAT SURFACES (ground, roads, building walls)

BOUNDING BOX REQUIREMENTS:
- Each bounding box must be a RECTANGLE (not a line or thin strip)
- Minimum dimensions: width >= 20 pixels AND height >= 20 pixels
- Maximum dimensions: width <= {w*0.5} pixels AND height <= {h*0.5} pixels (no single pile should be >50% of image)
- Aspect ratio: width/height must be between 0.2 and 5.0 (not extremely wide or tall)
- Bounding box must tightly fit around a SINGLE material stack/bale/container
- DO NOT create horizontal lines across the image
- DO NOT create boxes that span the entire image width

For EACH valid detection, provide:
- Bounding box: [x, y, width, height] in pixels (image size: {w}x{h})
- Volume: estimated cubic meters (m³) - must be > 0 and reasonable (typically 1-500 m³ per pile)
- Tonnage: estimated tons - must be > 0 and reasonable (typically 0.5-1000 tons per pile)
- Materials: percentage breakdown (must sum to ~100%)
- Confidence: 0.0-1.0 (only include if confidence >= 0.6)
- Description: brief description of what material type (e.g., "compressed paper bales", "wood debris stack", "metal scrap pile")

Return your analysis as a JSON object with this exact structure:
{{
    "objects": [
        {{
            "id": 1,
            "bbox": [x, y, width, height],
            "volume_m3": estimated_volume,
            "tonnage_tons": estimated_weight,
            "materials": {{
                "concrete": percentage,
                "wood": percentage,
                "metal": percentage,
                "plastic": percentage,
                "mixed": percentage
            }},
            "confidence": 0.6-1.0,
            "description": "material type description"
        }}
    ],
    "total_volume_m3": sum of all volumes,
    "total_tonnage_tons": sum of all tonnages,
    "overall_materials": {{
        "concrete": overall_percentage,
        "wood": overall_percentage,
        "metal": overall_percentage,
        "plastic": overall_percentage,
        "mixed": overall_percentage
    }}
}}

IMPORTANT: Only include detections you are confident are actual recycling MATERIALS, not infrastructure. If you're unsure, exclude it. Quality over quantity.
"""
        
        # Call AI API
        if self.provider == 'openai':
            response = self._call_openai(img_base64, prompt)
        elif self.provider == 'anthropic':
            response = self._call_anthropic(img_base64, prompt)
        elif self.provider == 'google':
            response = self._call_google(img_base64, prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Parse response
        result = self._parse_response(response, image)
        
        return result
    
    def _call_openai(self, img_base64: str, prompt: str) -> str:
        """Call OpenAI GPT-4V API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            response_format={"type": "json_object"}  # Force JSON response
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, img_base64: str, prompt: str) -> str:
        """Call Anthropic Claude 3 API."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no markdown, no code blocks, just the JSON object."
                        }
                    ]
                }
            ]
        )
        return message.content[0].text
    
    def _call_google(self, img_base64: str, prompt: str) -> str:
        """Call Google Gemini API (FREE tier available!)."""
        # Convert base64 to PIL Image for Gemini
        from PIL import Image
        import base64
        import time
        
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        
        # Reduce image size if too large (Gemini has limits)
        max_size = 2048
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"[!] Resized image to {img.width}x{img.height} for API")
        
        start_time = time.time()
        timeout = 60  # 60 second timeout
        
        # Try with JSON response format first (for newer models)
        try:
            print(f"[*] Calling Gemini API with model: {self.model}")
            response = self.client.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",  # Force JSON response
                    temperature=0.1,  # Lower temperature for more consistent outputs
                ),
                request_options={'timeout': timeout}  # Add timeout
            )
            elapsed = time.time() - start_time
            print(f"[+] Gemini API call completed in {elapsed:.1f}s")
            return response.text
        except Exception as e:
            error_str = str(e)
            elapsed = time.time() - start_time
            
            # Check for timeout
            if elapsed >= timeout or 'timeout' in error_str.lower():
                raise TimeoutError(
                    f"Gemini API call timed out after {elapsed:.1f}s. "
                    f"The API may be slow or the image is too large. "
                    f"Try using a smaller image or check your internet connection."
                )
            
            # Check for 404 model not found errors
            if '404' in error_str or 'not found' in error_str.lower():
                raise ValueError(
                    f"Gemini model '{self.model}' not found or not supported. "
                    f"Error: {error_str}\n"
                    f"Try specifying a different model in api_key.txt: google:API_KEY:MODEL_NAME\n"
                    f"Common working models: gemini-pro, gemini-1.5-flash, gemini-1.5-pro"
                )
            # For other errors, try without JSON constraint
            try:
                print(f"[!] JSON mode failed ({error_str}), trying without JSON constraint...")
                response = self.client.generate_content(
                    [prompt, img],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                    ),
                    request_options={'timeout': timeout}
                )
                elapsed = time.time() - start_time
                print(f"[+] Gemini API call completed in {elapsed:.1f}s (without JSON mode)")
                return response.text
            except Exception as e2:
                # Last resort: no config
                try:
                    response = self.client.generate_content(
                        [prompt, img],
                        request_options={'timeout': timeout}
                    )
                    elapsed = time.time() - start_time
                    print(f"[+] Gemini API call completed in {elapsed:.1f}s (basic mode)")
                    return response.text
                except Exception as e3:
                    raise ValueError(
                        f"Failed to call Gemini API: {e3}\n"
                        f"Model: {self.model}\n"
                        f"Elapsed time: {time.time() - start_time:.1f}s\n"
                        f"Check your API key and model name."
                    )
    
    def _parse_response(self, response_text: str, image: np.ndarray) -> Dict:
        """Parse AI response and convert to pipeline format."""
        h, w = image.shape[:2]
        
        try:
            # Extract JSON from response (handle markdown code blocks if present)
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[!] Failed to parse AI response as JSON: {e}")
            print(f"[!] Response was: {response_text[:500]}")
            return self._empty_result()
        
        # Convert to pipeline format
        objects = data.get('objects', [])
        
        # Create combined mask from all bounding boxes
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        segments = []
        
        for obj in objects:
            bbox = obj.get('bbox', [])
            if len(bbox) == 4:
                x, y, width, height = bbox
                # Draw rectangle on mask
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    combined_mask[y1:y2, x1:x2] = 255
                    
                    segments.append({
                        'id': obj.get('id', len(segments)),
                        'bbox': [x1, y1, x2-x1, y2-y1],
                        'area': (x2-x1) * (y2-y1),
                        'centroid': (x1 + (x2-x1)//2, y1 + (y2-y1)//2),
                    })
        
        # Calculate totals from validated objects only
        total_volume = sum(obj.get('volume_m3', 0) for obj in valid_objects)
        total_tonnage = sum(obj.get('tonnage_tons', 0) for obj in valid_objects)
        
        # Calculate overall materials from validated objects
        overall_materials = {}
        for obj in valid_objects:
            materials = obj.get('materials', {})
            for material, pct in materials.items():
                if material not in overall_materials:
                    overall_materials[material] = 0
                overall_materials[material] += pct
        
        # Normalize to percentages
        total_pct = sum(overall_materials.values())
        if total_pct > 0:
            overall_materials = {k: (v / total_pct * 100) for k, v in overall_materials.items()}
        else:
            # Fallback to data if calculation fails
            overall_materials = data.get('overall_materials', {})
        
        # Normalize material percentages
        total_pct = sum(overall_materials.values())
        if total_pct > 0:
            overall_materials = {k: (v / total_pct * 100) for k, v in overall_materials.items()}
        else:
            overall_materials = {'mixed': 100.0}
        
        return {
            'method': f'ai_{self.provider}',
            'objects': valid_objects,  # Only return validated objects
            'segmentation': {
                'mask': combined_mask,
                'masks': [combined_mask],  # Single combined mask for now
                'segments': segments,
                'num_segments': len(segments),
            },
            'volume': {
                'volume_m3': round(total_volume, 2),
                'tonnage_estimate': round(total_tonnage, 1),
            },
            'materials': {
                'percentages': {k: round(v, 1) for k, v in overall_materials.items()}
            },
            'ai_response': data,  # Store raw AI response for debugging
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'method': f'ai_{self.provider}',
            'objects': [],
            'segmentation': {
                'mask': np.zeros((100, 100), dtype=np.uint8),
                'masks': [],
                'segments': [],
                'num_segments': 0,
            },
            'volume': {
                'volume_m3': 0.0,
                'tonnage_estimate': 0.0,
            },
            'materials': {
                'percentages': {'mixed': 100.0}
            },
            'ai_response': {},
        }
