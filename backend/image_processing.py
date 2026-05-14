import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply preprocessing to improve OCR accuracy."""
    if image.mode != 'L':
        image = image.convert('L')
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)
    
    sharpener = ImageEnhance.Sharpness(image)
    image = sharpener.enhance(2.0)
    
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    min_width = 1200
    if image.width < min_width:
        ratio = min_width / image.width
        new_size = (min_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    return image

def extract_text_from_image(image_data: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_data))
        processed = preprocess_image(image)
        
        configs = [
            '--psm 6',
            '--psm 3',
            '--psm 11',
        ]
        
        best_text = ""
        for config in configs:
            text = pytesseract.image_to_string(processed, config=config).strip()
            if len(text) > len(best_text):
                best_text = text
        
        return best_text if best_text else "[No readable text detected in image]"
        
    except Exception as e:
        return f"[OCR Error: {str(e)}]"

def extract_text_from_scanned_pdf(pdf_data: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_data, dpi=300)
        all_text = []
        
        for i, img in enumerate(images[:5]):
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, config='--psm 6')
            if text.strip():
                all_text.append(f"--- Page {i+1} ---\n{text.strip()}")
        
        return "\n\n".join(all_text) if all_text else "[No readable text detected in PDF]"
        
    except ImportError:
        return "[OCR Error: pdf2image not installed. Run: pip install pdf2image]"
    except Exception as e:
        return f"[OCR PDF Error: {str(e)}]"
