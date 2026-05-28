# SSIM heatmap + pixel diff
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_diff(figma: Image.Image, web: Image.Image) -> dict:
    f_arr = np.array(figma.convert("L"))
    w_arr = np.array(web.convert("L"))

    score, diff_arr = ssim(f_arr, w_arr, full=True)
    diff_norm = (1 - diff_arr) * 255  # highlight divergence

    heatmap = Image.fromarray(diff_norm.astype(np.uint8)).convert("RGB")
    pixel_diff = ImageChops.difference(figma, web)

    return {
        "ssim_score": round(float(score), 4),
        "heatmap": heatmap,
        "pixel_diff": pixel_diff,
    }