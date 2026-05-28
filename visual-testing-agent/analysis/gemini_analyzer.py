# google-genai vision comparison
from pathlib import Path
from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are a senior visual QA engineer.
You will receive two UI screenshots:
  - Image 1: Figma design (source of truth)
  - Image 2: Live website implementation

Compare them thoroughly and return a beautifully formatted Markdown report detailing the visual discrepancies.
Include the following in your report:
- An Executive Summary
- An overall AI Score (0-100) and Severity (PASS, WARNING, or FAIL)
- The calculated pixel-match SSIM Score (which will be provided in the prompt context)
- A categorized table or list of detected issues (Layout, Typography, Color, Spacing) with Severity levels (Low, Medium, High).
- Lists of missing elements and extra elements.
- Any accessibility notes.

Format the output strictly as clean Markdown. Do NOT wrap it in a JSON block."""

def analyze_pair(figma_path: str, web_path: str, ssim_score: float, model: str = "gemini-3.5-flash") -> str:
    """
    Sends the Figma design + web screenshot to Gemini for AI visual QA analysis.

    Args:
        figma_path: Path to the Figma design image.
        web_path:   Path to the web screenshot image.
        ssim_score: Computed pixel-level similarity score.
        model:      Gemini model ID to use.

    Returns:
        Formatted Markdown string report from Gemini.
    """
    from config import settings

    client = genai.Client(api_key=settings.gemini_api_key)

    figma_bytes = Path(figma_path).read_bytes()
    web_bytes   = Path(web_path).read_bytes()

    context_prompt = f"The computed pixel-match SSIM score is {ssim_score*100:.1f}% (where 100% is a perfect match). Please include this context in your report."

    response = client.models.generate_content(
        model=model,
        contents=[
            SYSTEM_PROMPT,
            context_prompt,
            types.Part.from_bytes(data=figma_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=web_bytes,   mime_type="image/png"),
        ]
    )

    return response.text.strip()
