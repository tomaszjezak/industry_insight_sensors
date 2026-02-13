"""
Multimodal AI analyzer for recycling facility images.
Uses GPT-4V or Claude 3 to detect, outline, and analyze recycling materials.
"""

import numpy as np
import cv2
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional
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
            # Use gemini-1.5-flash (free, fast, supports vision)
            # Alternative: gemini-2.0-flash (newer, if available)
            self.model = model or 'gemini-1.5-flash'
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
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
        
        # Create detailed prompt
        prompt = f"""Analyze this recycling facility image taken from Google Street View (camera height approximately {camera_height_m}m).

Your task:
1. Detect and outline ALL individual recycling piles, bales, containers, and material stacks visible in the image
2. For EACH detected object, provide:
   - Bounding box coordinates (x, y, width, height) in pixels (image size: {w}x{h})
   - Estimated volume in cubic meters (m³)
   - Estimated weight/tonnage in tons
   - Material composition breakdown (concrete, wood, metal, plastic, mixed, etc.) as percentages
   - Confidence level (0-1) for your estimates

Focus on:
- Individual piles of recycling materials (not the entire image)
- Bales of sorted materials
- Containers or bins with materials
- Stacks of debris

Exclude:
- Buildings, roads, vehicles (unless they're part of the recycling operation)
- Sky, vegetation, empty ground

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
            "confidence": 0.0-1.0,
            "description": "brief description"
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

Be precise with bounding boxes - they should tightly fit each individual pile/object.
Use your knowledge of material densities and typical recycling facility operations to estimate volumes and weights.
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
        
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        
        # Gemini can take PIL Image directly
        response = self.client.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",  # Force JSON response
                temperature=0.1,  # Lower temperature for more consistent outputs
            )
        )
        return response.text
    
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
        
        # Calculate totals
        total_volume = data.get('total_volume_m3', sum(obj.get('volume_m3', 0) for obj in objects))
        total_tonnage = data.get('total_tonnage_tons', sum(obj.get('tonnage_tons', 0) for obj in objects))
        overall_materials = data.get('overall_materials', {})
        
        # Normalize material percentages
        total_pct = sum(overall_materials.values())
        if total_pct > 0:
            overall_materials = {k: (v / total_pct * 100) for k, v in overall_materials.items()}
        else:
            overall_materials = {'mixed': 100.0}
        
        return {
            'method': f'ai_{self.provider}',
            'objects': objects,
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
            'ai_response': data,  # Store raw AI response
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
