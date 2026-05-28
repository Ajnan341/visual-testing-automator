# Resize + align with Pillow/OpenCV
from PIL import Image


def align_images(figma_path: str, web_path: str) -> tuple[Image.Image, Image.Image]:
    """Resizes web screenshot to match Figma dimensions."""
    figma = Image.open(figma_path).convert("RGB")
    web   = Image.open(web_path).convert("RGB")
    web_resized = web.resize(figma.size, Image.LANCZOS)
    return figma, web_resized
