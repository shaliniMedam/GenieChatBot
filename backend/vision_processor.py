"""
Vision Processor - Uses Qwen2.5-VL for image understanding.
"""

from PIL import Image
import io
import torch

# Global model instance (loaded once)
_model = None
_processor = None

def get_model():
    """Lazy-load the vision model."""
    global _model, _processor
    if _model is None:
        print("Loading Qwen2.5-VL vision model... (first time, ~6GB download)")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        _processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("Qwen2.5-VL loaded successfully!")
    return _model, _processor

def describe_image(image_data: bytes, question: str = "What is in this image?") -> str:
    """Answer questions about image content."""
    try:
        model, processor = get_model()
        image = Image.open(io.BytesIO(image_data))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[Vision Error: {str(e)}]"

def extract_text_from_image(image_data: bytes) -> str:
    """Use vision model to read text from images."""
    return describe_image(image_data, "Read and transcribe all text in this image exactly as it appears.")