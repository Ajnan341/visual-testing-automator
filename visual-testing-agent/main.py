# CLI orchestrator
import argparse
import json
from pathlib import Path

from ingestion.figma_client import load_design_image
from ingestion.web_capture import load_web_image
from processing.preprocessor import align_images
from processing.diff_engine import compute_diff
from analysis.gemini_analyzer import analyze_pair
from config import settings


def run(page_name: str, figma_image_path: str, web_image_path: str, output_dir: str):
    """
    Core pipeline for a single page comparison.

    Args:
        page_name:        Human-readable label for the page (e.g. "Home").
        figma_image_path: Path to the uploaded Figma design image.
        web_image_path:   Path to the uploaded web screenshot image.
        output_dir:       Directory where diff images and report are saved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load uploaded images ──────────────────────────────────────────────
    print(f">> Loading Figma design image:  {figma_image_path}")
    figma_path, _ = load_design_image(figma_image_path)

    print(f">> Loading web screenshot:      {web_image_path}")
    web_path, _ = load_web_image(web_image_path)

    # -- 2. Align (resize web to Figma dimensions) --
    print(">> Aligning images...")
    figma_img, web_img = align_images(figma_path, web_path)

    # -- 3. Pixel-level diff + SSIM heatmap --
    print(">> Computing diff...")
    diff = compute_diff(figma_img, web_img)
    heatmap_path = str(out / f"diff_{page_name}.png")
    diff["heatmap"].save(heatmap_path)
    print(f"   SSIM score: {diff['ssim_score']}")

    # -- 4. Gemini AI visual analysis --
    print(">> Running Gemini AI analysis...")
    ai_result = analyze_pair(figma_path, web_path, diff["ssim_score"], settings.model)

    report_path = out / f"{page_name}_report.md"
    report_path.write_text(ai_result, encoding="utf-8")

    print(f"\n[OK] Report saved to: {report_path}")


def run_batch(pairs: list[dict], output_dir: str):
    """
    Run comparisons for multiple pages.

    Args:
        pairs: List of dicts with keys: name, figma, web
        output_dir: Root output directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for pair in pairs:
        name = pair["name"]
        print(f"\n{'='*50}")
        print(f"  Page: {name}")
        print(f"{'='*50}")

        figma_path, _ = load_design_image(pair["figma"])
        web_path, _   = load_web_image(pair["web"])

        figma_img, web_img = align_images(figma_path, web_path)
        diff = compute_diff(figma_img, web_img)

        heatmap_path = str(out / f"diff_{name}.png")
        diff["heatmap"].save(heatmap_path)

        ai_result = analyze_pair(figma_path, web_path, diff["ssim_score"], settings.model)

        report_path = out / f"{name}_report.md"
        report_path.write_text(ai_result, encoding="utf-8")
        print(f"\n[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Testing Agent — compare Figma designs to web screenshots"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Single page comparison ────────────────────────────────────────────────
    single = subparsers.add_parser("compare", help="Compare a single page")
    single.add_argument("--name",   required=True,             help="Page label (e.g. Home)")
    single.add_argument("--figma",  required=True,             help="Path to Figma design image")
    single.add_argument("--web",    required=True,             help="Path to web screenshot image")
    single.add_argument("--output", default=settings.output_dir, help="Output directory")

    # ── Batch comparison via JSON config ──────────────────────────────────────
    batch = subparsers.add_parser("batch", help="Compare multiple pages from a JSON config")
    batch.add_argument(
        "--config", required=True,
        help='Path to JSON file: [{"name":"Home","figma":"path.png","web":"path.png"}, ...]'
    )
    batch.add_argument("--output", default=settings.output_dir, help="Output directory")

    args = parser.parse_args()

    if args.command == "compare":
        run(args.name, args.figma, args.web, args.output)

    elif args.command == "batch":
        with open(args.config, encoding="utf-8") as f:
            pairs = json.load(f)
        run_batch(pairs, args.output)
