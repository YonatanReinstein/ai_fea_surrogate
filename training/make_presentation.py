"""
Builds/updates a PowerPoint deck that tracks GNN architecture iterations:
one slide per run, showing what changed and the resulting loss curves.

Usage:
    python -m training.make_presentation <run_name> --notes "what changed in this run"

Re-running with the same <run_name> replaces that run's slide instead of duplicating it.
"""
import argparse
import json
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

DECK_PATH = "training/runs/gnn_iteration_tracking.pptx"
RUNS_DIR = "training/runs"

TITLE_COLOR = RGBColor(0x1F, 0x2A, 0x44)
BODY_COLOR = RGBColor(0x33, 0x33, 0x33)


def _run_marker(slide):
    """Hidden textbox on a run slide storing the run name, used to find/replace it later."""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.startswith("run_id::"):
            return shape.text_frame.text.split("::", 1)[1]
    return None


def _load_run(run_name: str):
    run_dir = os.path.join(RUNS_DIR, run_name)
    with open(os.path.join(run_dir, "losses.json")) as f:
        losses = json.load(f)
    image_path = os.path.join(run_dir, "loss_plot.png")
    metrics = {
        key: {"final": values[-1], "best": min(values)}
        for key, values in losses.items()
    }
    return metrics, image_path


def _add_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.8), Inches(11.7), Inches(1.5))
    tf = box.text_frame
    tf.text = "GNN Surrogate — Architecture Iteration Log"
    tf.paragraphs[0].font.size = Pt(36)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = TITLE_COLOR

    sub = slide.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(11.7), Inches(0.6))
    sub.text_frame.text = "One slide per run: what changed, and the resulting train/val loss curves."
    sub.text_frame.paragraphs[0].font.size = Pt(16)
    sub.text_frame.paragraphs[0].font.color.rgb = BODY_COLOR
    return slide


def add_run_slide(prs, run_name: str, notes: str):
    metrics, image_path = _load_run(run_name)

    slide = prs.slides.add_slide(prs.slide_layouts[6])

    marker = slide.shapes.add_textbox(Inches(0), Inches(0), Inches(0.1), Inches(0.1))
    marker.text_frame.text = f"run_id::{run_name}"
    marker.text_frame.paragraphs[0].font.size = Pt(1)

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.7))
    title.text_frame.text = f"Run: {run_name}"
    title.text_frame.paragraphs[0].font.size = Pt(28)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = TITLE_COLOR

    notes_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(5.8), Inches(2.0))
    ntf = notes_box.text_frame
    ntf.word_wrap = True
    ntf.text = notes
    ntf.paragraphs[0].font.size = Pt(15)
    ntf.paragraphs[0].font.color.rgb = BODY_COLOR

    metrics_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.3), Inches(5.8), Inches(3.6))
    mtf = metrics_box.text_frame
    mtf.word_wrap = True
    mtf.text = "Metrics (final / best)"
    mtf.paragraphs[0].font.size = Pt(15)
    mtf.paragraphs[0].font.bold = True
    mtf.paragraphs[0].font.color.rgb = TITLE_COLOR
    for key, vals in metrics.items():
        p = mtf.add_paragraph()
        p.text = f"{key}:  {vals['final']:.4f} / {vals['best']:.4f}"
        p.font.size = Pt(13)
        p.font.color.rgb = BODY_COLOR

    if os.path.exists(image_path):
        slide.shapes.add_picture(image_path, Inches(6.5), Inches(1.1), width=Inches(6.3))

    return slide


def build_or_update_deck(run_name: str, notes: str):
    if os.path.exists(DECK_PATH):
        prs = Presentation(DECK_PATH)
    else:
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        _add_title_slide(prs)

    existing_idx = None
    for idx, slide in enumerate(prs.slides):
        if _run_marker(slide) == run_name:
            existing_idx = idx
            break

    if existing_idx is not None:
        xml_slides = prs.slides._sldIdLst
        slide_id = list(xml_slides)[existing_idx]
        xml_slides.remove(slide_id)

    add_run_slide(prs, run_name, notes)
    prs.save(DECK_PATH)
    print(f"Deck saved to: {DECK_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Name of the run directory under training/runs/")
    parser.add_argument("--notes", required=True, help="What changed in this run")
    args = parser.parse_args()
    build_or_update_deck(args.run_name, args.notes)
