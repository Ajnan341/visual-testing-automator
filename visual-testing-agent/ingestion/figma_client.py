# Image upload loader — replaces Figma REST API
from pathlib import Path
from PIL import Image


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_design_image(image_path: str) -> tuple[str, Image.Image]:
    """
    Validates and loads an uploaded Figma design image.

    Args:
        image_path: Path to the uploaded Figma/design image file.

    Returns:
        Tuple of (resolved_path_str, PIL.Image object).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(image_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Design image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    img = Image.open(path).convert("RGB")
    return str(path), img


def load_design_images(image_paths: dict) -> dict:
    """
    Load multiple design images from a name → path mapping.

    Args:
        image_paths: {page_name: file_path}

    Returns:
        {page_name: (resolved_path_str, PIL.Image)}
    """
    results = {}
    for name, path in image_paths.items():
        results[name] = load_design_image(path)
    return results
