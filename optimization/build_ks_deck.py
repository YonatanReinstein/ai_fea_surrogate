"""Builds Kreisselmeier-Steinhauser.pptx from scratch: six slides.

Slide 1 -- how topology optimization itself is done (the SIMP-style OC loop
over per-tile design variables, see optimization/topopt_optimizer.py).
Slides 1a-1c -- term-by-term walkthrough of slide 1's equations: the design
variable and problem statement, the OC update rule, and why lambda needs a
bisection search / sensitivities need filtering. Added because slide 1 packs
five equations into one card with no room to define symbols.
Slide 2 -- what the KS function is (formula + convergence chart only).
Slide 3 -- how KS is used as the optimizer's stress constraint, and how the
GNN surrogate produces it (ks_pool inside GNN.forward(), see
utils/gnn_surrogate.py and optimization/topopt_optimizer.py).

Rerun after editing optimization/artifacts/ks_vs_max.png (make_ks_chart.py)
or this file.
"""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

PATH = "optimization/Kreisselmeier-Steinhauser.pptx"
CHART = "optimization/artifacts/ks_vs_max.png"
CHART_ASPECT = 11.0 / 2.96  # matches make_ks_chart.py's figsize=(11.0, 2.96)

SLIDE_W, SLIDE_H = Emu(12191695), Emu(6858000)
LEFT = Emu(548640)
CONTENT_W = Emu(11064240)
TOPBAR_H = Emu(109728)

INK_PRIMARY = RGBColor(0x1B, 0x1F, 0x2A)
INK_SECONDARY = RGBColor(0x55, 0x5D, 0x6B)
INK_MUTED = RGBColor(0x89, 0x87, 0x81)
ACCENT = RGBColor(0x2F, 0x6F, 0xED)
CARD_FILL = RGBColor(0xF4, 0xF6, 0xFA)
CARD_BORDER = RGBColor(0xDD, 0xE1, 0xE8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def add_textbox(slide, left, top, width, height, text, size, color, bold=False, align=PP_ALIGN.LEFT, anchor=None,
                 font_name=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if font_name:
        run.font.name = font_name
    return box


def add_card(slide, left, top, width, height, fill=CARD_FILL, border=CARD_BORDER, line_w=Pt(1)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = line_w
    shape.shadow.inherit = False
    return shape


def add_base(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()
    bg.shadow.inherit = False

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, TOPBAR_H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False

    bg._element.getparent().remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)
    slide.shapes._spTree.insert(3, bar._element)


def add_title_block(slide, title, subtitle):
    add_textbox(slide, LEFT, Emu(320040), CONTENT_W, Emu(760000), title, 40, INK_PRIMARY, bold=True)
    add_textbox(slide, LEFT, Emu(1140040), CONTENT_W, Emu(400000), subtitle, 20, INK_SECONDARY)


def build_slide1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "Kreisselmeier–Steinhauser (KS) Function",
                     "A smooth, differentiable approximation of max()")

    formula_top, formula_h = Emu(1690200), Emu(1550000)
    add_card(slide, LEFT, formula_top, CONTENT_W, formula_h)
    add_textbox(slide, LEFT, formula_top + Emu(120000), CONTENT_W, Emu(480000),
                "KS(g) = (1/ρ) · ln [ Σᵢ exp(ρ · gᵢ) ]",
                30, INK_PRIMARY, bold=True, align=PP_ALIGN.CENTER)
    add_textbox(slide, LEFT, formula_top + Emu(640000), CONTENT_W, Emu(320000),
                "= gₘₐₓ + (1/ρ) · ln [ Σᵢ exp(ρ·(gᵢ − gₘₐₓ)) ]   — the numerically-stable form used in code",
                17, INK_SECONDARY, align=PP_ALIGN.CENTER)
    add_textbox(slide, LEFT, formula_top + Emu(1000000), CONTENT_W, Emu(340000),
                "As ρ → ∞,  KS(g) → max(gᵢ)",
                18, INK_SECONDARY, align=PP_ALIGN.CENTER)

    chart_head_top = formula_top + formula_h + Emu(150000)
    add_textbox(slide, LEFT, chart_head_top, CONTENT_W, Emu(380000),
                "KS(ρ) is a conservative bound: KS(g) ≥ max(gᵢ) for every ρ > 0",
                22, ACCENT, bold=True)

    img_top = chart_head_top + Emu(380000) + Emu(60000)
    img_h = SLIDE_H - Emu(400000) - img_top
    img_w = Emu(int(img_h * CHART_ASPECT))
    img_left = LEFT + (CONTENT_W - img_w) // 2
    slide.shapes.add_picture(CHART, img_left, img_top, width=img_w, height=img_h)


def add_arrow(slide, cx, cy, size=Emu(360000)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, cx - size // 2, cy - Emu(140000), size, Emu(280000))
    shape.fill.solid()
    shape.fill.fore_color.rgb = INK_MUTED
    shape.line.fill.background()
    shape.shadow.inherit = False


def add_bullet_card(slide, left, top, width, height, heading, bullets, bullet_size=19, space_after=22):
    add_card(slide, left, top, width, height)
    pad = Emu(220000)
    add_textbox(slide, left + pad, top + Emu(160000), width - 2 * pad, Emu(400000),
                heading, 22, ACCENT, bold=True)
    box = slide.shapes.add_textbox(left + pad, top + Emu(620000), width - 2 * pad, height - Emu(780000))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = "•  " + text
        run.font.size = Pt(bullet_size)
        run.font.color.rgb = INK_PRIMARY


def add_two_column_bullets(slide, col_top, col_bottom, left_heading, left_bullets, right_heading, right_bullets,
                            bullet_size=19, space_after=22):
    col_gap = Emu(300000)
    col_w = Emu((CONTENT_W - col_gap) // 2)
    col_left_x = LEFT
    col_right_x = LEFT + col_w + col_gap
    col_h = col_bottom - col_top
    add_bullet_card(slide, col_left_x, col_top, col_w, col_h, left_heading, left_bullets,
                     bullet_size=bullet_size, space_after=space_after)
    add_bullet_card(slide, col_right_x, col_top, col_w, col_h, right_heading, right_bullets,
                     bullet_size=bullet_size, space_after=space_after)


def add_bullet_card_with_formula(slide, left, top, width, height, heading, formula, bullets,
                                  bullet_size=16, space_after=14):
    """Like add_bullet_card, but with a centered formula line pinned under
    the heading and the bullets filling the rest of the card."""
    add_card(slide, left, top, width, height)
    pad = Emu(220000)
    add_textbox(slide, left + pad, top + Emu(160000), width - 2 * pad, Emu(400000),
                heading, 22, ACCENT, bold=True)
    add_textbox(slide, left + pad, top + Emu(620000), width - 2 * pad, Emu(440000),
                formula, 19, INK_PRIMARY, bold=True, align=PP_ALIGN.CENTER)
    box = slide.shapes.add_textbox(left + pad, top + Emu(1080000), width - 2 * pad, height - Emu(1240000))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = "•  " + text
        run.font.size = Pt(bullet_size)
        run.font.color.rgb = INK_PRIMARY


def add_pipeline(slide, label_top, label, row_top, nodes, accent_idx=()):
    add_textbox(slide, LEFT, label_top, CONTENT_W, Emu(380000), label, 20, INK_PRIMARY, bold=True)

    row_h = Emu(700000)
    node_w = Emu(2300000)
    gap_w = Emu((CONTENT_W - len(nodes) * node_w) // (len(nodes) - 1))

    x = LEFT
    centers = []
    for i, node_label in enumerate(nodes):
        card = add_card(slide, x, row_top, node_w, row_h)
        tf = card.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = node_label.split("\n")
        for j, line in enumerate(lines):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = line
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = ACCENT if i in accent_idx else INK_PRIMARY
        centers.append(x + node_w // 2)
        x = x + node_w + gap_w

    for i in range(len(nodes) - 1):
        gap_center = (centers[i] + node_w // 2 + centers[i + 1] - node_w // 2) // 2
        add_arrow(slide, gap_center, row_top + row_h // 2)


FORMULA_LINES = [
    ("minimize V(x)  s.t.  G(x) = stress(x)/σ_yield − 1 ≤ 0,  lo ≤ x ≤ hi", 21, True, INK_PRIMARY),
    ("OC update each tile e (λ tuned by bisection so G(x_new) ≈ 0):", 14, False, INK_MUTED),
    ("B_e(λ) = λ·(−∂G/∂xₑ) / (∂V/∂xₑ)", 20, False, INK_PRIMARY),
    ("xₑ ← clip( xₑ·B_e(λ)^η,  xₑ ± Δ,  [loₑ, hiₑ] )", 20, False, INK_PRIMARY),
    ("sensitivity filter (Sigmund 1997), suppresses checkerboarding:", 14, False, INK_MUTED),
    ("s̃ₑ = (1/xₑ) · Σᵢ Hₑᵢxᵢsᵢ  /  Σᵢ Hₑᵢ", 20, False, INK_PRIMARY),
]


def build_slide_topopt(prs, use_video=True):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "Topology Optimization: the OC Loop",
                     "A SIMP-style optimality-criteria search over per-tile design variables")

    col_top, col_bottom = Emu(1690200), Emu(5078000)
    col_gap = Emu(300000)
    col_w = Emu((CONTENT_W - col_gap) // 2)
    col_left_x = LEFT
    col_right_x = LEFT + col_w + col_gap
    col_h = col_bottom - col_top
    pad = Emu(220000)

    add_card(slide, col_left_x, col_top, col_w, col_h)
    box = slide.shapes.add_textbox(col_left_x + pad, col_top, col_w - 2 * pad, col_h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (text, size, bold, color) in enumerate(FORMULA_LINES):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(14)
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color

    add_card(slide, col_right_x, col_top, col_w, col_h)
    media_w = col_w - Emu(240000)
    media_aspect = 1280 / 560 if use_video else 2560 / 560  # before/after still is 2 frames wide
    media_h = Emu(int(media_w / media_aspect))
    caption_h = Emu(300000)
    block_h = media_h + Emu(60000) + caption_h
    media_top = col_top + (col_h - block_h) // 2
    media_left = col_right_x + (col_w - media_w) // 2
    if use_video:
        slide.shapes.add_movie(
            "optimization/artifacts/topopt_evolution.mp4",
            media_left, media_top, media_w, media_h,
            poster_frame_image="optimization/artifacts/topopt_poster.png",
            mime_type="video/mp4",
        )
    else:
        slide.shapes.add_picture(
            "optimization/artifacts/topopt_before_after.jpg",
            media_left, media_top, width=media_w, height=media_h,
        )
    add_textbox(slide, col_right_x + pad, media_top + media_h + Emu(60000), col_w - 2 * pad, caption_h,
                "60 generations, CCX-verified: volume 705 → 419 (−41%), stress held at σ_yield",
                13, INK_SECONDARY, align=PP_ALIGN.CENTER)


def build_slide_topopt_vars(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "OC Loop, Term by Term (1/3): the Design Variable",
                     "What x, V(x), and G(x) mean, before any update rule touches them")

    col_top, col_bottom = Emu(1690200), Emu(6458000)
    add_two_column_bullets(
        slide, col_top, col_bottom,
        "The design variable x",
        [
            "hollow_cube's structure is divided into 840 small tiles arranged on a 3-D grid; each tile e has "
            "one number xₑ, its material amount (a frame/wall thickness)",
            "xₑ is free to move between a lower bound loₑ and an upper bound hiₑ, set per tile",
            "The full design is the vector x = (x₁, …, x₈₄₀) — one thickness value per tile, all optimized together",
        ],
        "The problem being solved",
        [
            "V(x) — total volume of material in design x; the objective, minimized to build the lightest structure possible",
            "stress(x) — the worst (peak) stress anywhere in the structure under design x; σ_yield is the "
            "material's fixed yield strength — the stress at which it begins to permanently deform",
            "G(x) = stress(x)/σ_yield − 1 ≤ 0 requires stress(x) ≤ σ_yield: the worst stress anywhere must "
            "stay under what the material can take",
        ],
        bullet_size=16, space_after=14,
    )


def build_slide_topopt_update(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "OC Loop, Term by Term (2/3): the Update Rule",
                     "How one generation turns sensitivities into a new xₑ for every tile e")

    formula_top, formula_h = Emu(1690200), Emu(820000)
    add_card(slide, LEFT, formula_top, CONTENT_W, formula_h)
    add_textbox(slide, LEFT, formula_top + Emu(140000), CONTENT_W, Emu(340000),
                "B_e(λ) = λ·(−∂G/∂xₑ) / (∂V/∂xₑ)",
                22, INK_PRIMARY, bold=True, align=PP_ALIGN.CENTER)
    add_textbox(slide, LEFT, formula_top + Emu(460000), CONTENT_W, Emu(340000),
                "xₑ ← clip( xₑ·B_e(λ)^η,  xₑ ± Δ,  [loₑ, hiₑ] )",
                22, INK_PRIMARY, bold=True, align=PP_ALIGN.CENTER)

    col_top = formula_top + formula_h + Emu(60000)
    col_bottom = Emu(6458000)
    add_two_column_bullets(
        slide, col_top, col_bottom,
        "Sensitivities and λ",
        [
            "∂G/∂xₑ — how much the stress constraint changes if tile e gets a sliver more material, found via "
            "finite differences (perturb tile e, rerun the evaluator). Usually negative — more material lowers stress",
            "∂V/∂xₑ — how much extra volume that sliver of material costs. Roughly the same number for every tile",
            "λ (lambda) — one shared number, the same for all 840 tiles this generation, chosen by bisection "
            "so the design lands at the constraint boundary",
        ],
        "Turning B_e into a new xₑ",
        [
            "B_e(λ) — a per-tile scaling factor: large where adding material helps the stress constraint a "
            "lot (tiles on the load path), small where it doesn't — those shrink",
            "η (eta) — a fixed damping exponent (e.g. 0.5) softening how hard B_e can push xₑ in one step, "
            "for numerical stability",
            "Δ (move limit) caps how far xₑ can change in one generation; clip(…, [loₑ, hiₑ]) then clamps the "
            "result back inside tile e's allowed range",
        ],
        bullet_size=15, space_after=10,
    )


def build_slide_topopt_filter(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "OC Loop, Term by Term (3/3): λ's Search and the Filter",
                     "Why λ needs 60 bisection steps, and why raw sensitivities get smoothed first")

    col_top, col_bottom = Emu(1690200), Emu(6458000)
    add_bullet_card(
        slide, LEFT, col_top, Emu((CONTENT_W - Emu(300000)) // 2), col_bottom - col_top,
        "Bisection on λ",
        [
            "λ must be one shared number that, once every tile applies its update, pushes the whole design "
            "right up to the constraint boundary G(x_new) ≈ 0 — not overly conservative, not infeasible",
            "There's no closed-form solution for λ, so the code guesses a value, checks a cheap linear "
            "estimate of G(x_new), and halves the search range — repeated up to 60 times",
            "All 60 steps only touch sensitivities already computed this generation; the one expensive step, "
            "an actual stress evaluation, runs once per generation, not once per bisection step",
        ],
        bullet_size=15, space_after=14,
    )
    add_bullet_card_with_formula(
        slide, LEFT + Emu((CONTENT_W - Emu(300000)) // 2) + Emu(300000), col_top,
        Emu((CONTENT_W - Emu(300000)) // 2), col_bottom - col_top,
        "Sensitivity filter — checkerboarding",
        "s̃ₑ = (1/xₑ) · Σᵢ Hₑᵢxᵢsᵢ / Σᵢ Hₑᵢ",
        [
            "sᵢ — a raw, tile-by-tile sensitivity before any smoothing",
            "Hₑᵢ — weight for how much tile i influences tile e's smoothed value: large for tiles physically "
            "near e, zero for tiles far away",
            "Σᵢ — a sum over every tile i in the grid, not just e itself",
            "s̃ₑ — the smoothed sensitivity that actually reaches B_e(λ) on the previous slide; the raw, "
            "unfiltered ∂G/∂xₑ or ∂V/∂xₑ never does",
            "Without this, per-tile sensitivities can flip one tile to full material next to a neighbor "
            "flipped to none — a checkerboard pattern, a numerical artifact with no structural meaning "
            "(Sigmund, 1997)",
        ],
        bullet_size=14, space_after=10,
    )


def build_slide1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "Kreisselmeier–Steinhauser (KS) Function",
                     "A smooth, differentiable approximation of max()")

    formula_top, formula_h = Emu(1690200), Emu(1550000)
    add_card(slide, LEFT, formula_top, CONTENT_W, formula_h)
    add_textbox(slide, LEFT, formula_top + Emu(120000), CONTENT_W, Emu(480000),
                "KS(g) = (1/ρ) · ln [ Σᵢ exp(ρ · gᵢ) ]",
                30, INK_PRIMARY, bold=True, align=PP_ALIGN.CENTER)
    add_textbox(slide, LEFT, formula_top + Emu(640000), CONTENT_W, Emu(320000),
                "= gₘₐₓ + (1/ρ) · ln [ Σᵢ exp(ρ·(gᵢ − gₘₐₓ)) ]   — the numerically-stable form used in code",
                17, INK_SECONDARY, align=PP_ALIGN.CENTER)
    add_textbox(slide, LEFT, formula_top + Emu(1000000), CONTENT_W, Emu(340000),
                "As ρ → ∞,  KS(g) → max(gᵢ)",
                18, INK_SECONDARY, align=PP_ALIGN.CENTER)

    chart_head_top = formula_top + formula_h + Emu(150000)
    add_textbox(slide, LEFT, chart_head_top, CONTENT_W, Emu(380000),
                "KS(ρ) is a conservative bound: KS(g) ≥ max(gᵢ) for every ρ > 0",
                22, ACCENT, bold=True)

    img_top = chart_head_top + Emu(380000) + Emu(60000)
    img_h = SLIDE_H - Emu(400000) - img_top
    img_w = Emu(int(img_h * CHART_ASPECT))
    img_left = LEFT + (CONTENT_W - img_w) // 2
    slide.shapes.add_picture(CHART, img_left, img_top, width=img_w, height=img_h)


def build_slide2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide)
    add_title_block(slide, "Using KS in the Optimization Loop",
                     "From per-node GNN predictions to a single differentiable constraint")

    col_top, col_bottom = Emu(1690200), Emu(5078000)
    add_two_column_bullets(
        slide, col_top, col_bottom,
        "Optimization constraint",
        [
            "Replaces the hard max-stress constraint with g(x) = KS(x) / σ_yield − 1 ≤ 0",
            "Per-tile sensitivities dg/dx come from finite differences, driving an optimality-criteria (OC) update",
            "KS spreads gradient signal across every near-peak node, not just one — smoother, more stable sensitivities than a hard max",
        ],
        "GNN surrogate output",
        [
            "The model head predicts per-node stress across the mesh (node_pred)",
            "GNN.forward() pools it with ks_pool() at the same ρ the training target used (ρ = 0.15 MPa⁻¹ default)",
            "forward() returns (graph_pred, node_pred) — graph_pred is the scalar the optimizer reads",
        ],
    )

    add_pipeline(
        slide, Emu(5278000), "How a KS value reaches the optimizer", Emu(5758000),
        ["GNN node stress", "ks_pool(ρ)", "graph_pred\n(scalar)", "Optimizer\nconstraint g(x) ≤ 0"],
        accent_idx=(0, 2),
    )


SLIDES_UPLOAD_PATH = "optimization/Kreisselmeier-Steinhauser_slides_upload.pptx"


def build_deck(use_video):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    build_slide_topopt(prs, use_video=use_video)
    build_slide_topopt_vars(prs)
    build_slide_topopt_update(prs)
    build_slide_topopt_filter(prs)
    build_slide1(prs)
    build_slide2(prs)
    return prs


def main():
    build_deck(use_video=True).save(PATH)
    print("wrote", PATH)
    # Google Slides' pptx import doesn't preserve embedded local video, and
    # the video is the bulk of the file's size -- a lighter before/after-image
    # variant is what actually survives an upload/import into Slides.
    build_deck(use_video=False).save(SLIDES_UPLOAD_PATH)
    print("wrote", SLIDES_UPLOAD_PATH)


if __name__ == "__main__":
    main()
