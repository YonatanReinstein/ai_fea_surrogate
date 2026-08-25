"""Builds Kreisselmeier-Steinhauser.pdf to match Kreisselmeier-Steinhauser.pptx
(see build_ks_deck.py) -- same three pages, same geometry (EMU/12700 -> pt).
No LibreOffice on this machine to export the pptx directly, so the PDF is
drawn independently with reportlab, using the same layout constants.
"""
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

PATH = "optimization/Kreisselmeier-Steinhauser.pdf"
CHART = "optimization/artifacts/ks_vs_max.png"

PAGE_W, PAGE_H = 960.0, 540.0
LEFT = 43.2
CONTENT_W = 871.2
TOPBAR_H = 8.64

INK_PRIMARY = HexColor("#1B1F2A")
INK_SECONDARY = HexColor("#555D6B")
INK_MUTED = HexColor("#898781")
ACCENT = HexColor("#2F6FED")
CARD_FILL = HexColor("#F4F6FA")
CARD_BORDER = HexColor("#DDE1E8")

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))


def y_top(top_pt, height_pt=0):
    return PAGE_H - top_pt - height_pt


def wrap_text(text, font, size, max_width):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(trial, font, size) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_base(c):
    c.setFillColor(white)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - TOPBAR_H, PAGE_W, TOPBAR_H, stroke=0, fill=1)


def draw_title_block(c, title, subtitle):
    c.setFillColor(INK_PRIMARY)
    c.setFont("DejaVu-Bold", 27)
    c.drawString(LEFT, y_top(25.2) - 27, title)
    c.setFillColor(INK_SECONDARY)
    c.setFont("DejaVu", 14)
    c.drawString(LEFT, y_top(89.8) - 14, subtitle)


def rounded_card(c, left, top, w, h, r=8):
    c.setFillColor(CARD_FILL)
    c.setStrokeColor(CARD_BORDER)
    c.setLineWidth(1)
    c.roundRect(left, y_top(top, h), w, h, r, stroke=1, fill=1)


def page1(c):
    draw_base(c)
    draw_title_block(c, "Kreisselmeier–Steinhauser (KS) Function",
                      "A smooth, differentiable approximation of max()")

    formula_top, formula_h = 133.1, 94.5
    rounded_card(c, LEFT, formula_top, CONTENT_W, formula_h, r=10)
    c.setFillColor(INK_PRIMARY)
    c.setFont("DejaVu-Bold", 24)
    c.drawCentredString(LEFT + CONTENT_W / 2, y_top(formula_top) - 44, "KS(g) = (1/ρ) · ln [ Σ exp(ρ · gᵢ) ]")
    c.setFillColor(INK_SECONDARY)
    c.setFont("DejaVu", 15)
    c.drawCentredString(LEFT + CONTENT_W / 2, y_top(formula_top) - 76, "As ρ → ∞,  KS(g) → max(gᵢ)")

    chart_head_top = formula_top + formula_h + 11.8
    c.setFillColor(ACCENT)
    c.setFont("DejaVu-Bold", 16.5)
    c.drawString(LEFT, y_top(chart_head_top) - 18, "KS(ρ) is a conservative bound: KS(g) ≥ max(gᵢ) for every ρ > 0")

    img_top = chart_head_top + 29.9 + 4.7
    img_h = PAGE_H - 31.5 - img_top
    img = ImageReader(CHART)
    c.drawImage(img, LEFT, y_top(img_top, img_h), width=CONTENT_W, height=img_h,
                preserveAspectRatio=True, anchor="n", mask="auto")


def draw_bullets(c, x, area_top, area_h, w, bullets, size=14.5):
    c.setFillColor(INK_PRIMARY)
    c.setFont("DejaVu", size)
    line_h = size * 1.35
    gap_h = 9
    wrapped = [wrap_text("•  " + b, "DejaVu", size, w) for b in bullets]
    total_h = sum(len(lines) for lines in wrapped) * line_h + (len(wrapped) - 1) * gap_h
    y = y_top(area_top) - (area_h - total_h) / 2 - size
    for lines in wrapped:
        for line in lines:
            c.drawString(x, y, line)
            y -= line_h
        y -= gap_h


def draw_two_column_bullets(c, col_top, col_bottom, left_heading, left_bullets, right_heading, right_bullets):
    col_gap = 23.6
    col_w = (CONTENT_W - col_gap) / 2
    col_left_x = LEFT
    col_right_x = LEFT + col_w + col_gap
    col_h = col_bottom - col_top
    pad = 17.3

    rounded_card(c, col_left_x, col_top, col_w, col_h)
    rounded_card(c, col_right_x, col_top, col_w, col_h)

    bullets_top = col_top + 48.8
    bullets_h = col_h - 48.8 - 14

    c.setFillColor(ACCENT)
    c.setFont("DejaVu-Bold", 16.5)
    c.drawString(col_left_x + pad, y_top(col_top + 12.6) - 18, left_heading)
    draw_bullets(c, col_left_x + pad, bullets_top, bullets_h, col_w - 2 * pad, left_bullets)

    c.setFillColor(ACCENT)
    c.setFont("DejaVu-Bold", 16.5)
    c.drawString(col_right_x + pad, y_top(col_top + 12.6) - 18, right_heading)
    draw_bullets(c, col_right_x + pad, bullets_top, bullets_h, col_w - 2 * pad, right_bullets)


def draw_pipeline(c, label_top, label, row_top, nodes):
    c.setFillColor(INK_PRIMARY)
    c.setFont("DejaVu-Bold", 15)
    c.drawString(LEFT, y_top(label_top) - 16, label)

    row_h = 55.1
    node_w = 181.1
    gap_w = (CONTENT_W - len(nodes) * node_w) / (len(nodes) - 1)

    x = LEFT
    centers = []
    for label_text, color in nodes:
        rounded_card(c, x, row_top, node_w, row_h)
        c.setFillColor(color)
        c.setFont("DejaVu-Bold", 11.5)
        lines = wrap_text(label_text, "DejaVu-Bold", 11.5, node_w - 16)
        total_h = len(lines) * 14
        ty = y_top(row_top + row_h / 2) + total_h / 2 - 10
        for line in lines:
            c.drawCentredString(x + node_w / 2, ty, line)
            ty -= 14
        centers.append(x + node_w / 2)
        x += node_w + gap_w

    c.setFillColor(INK_MUTED)
    c.setFont("DejaVu-Bold", 16)
    for i in range(len(nodes) - 1):
        gap_center = (centers[i] + node_w / 2 + centers[i + 1] - node_w / 2) / 2
        c.drawCentredString(gap_center, y_top(row_top + row_h / 2) - 6, "→")


FORMULA_LINES = [
    ("minimize V(x)  s.t.  G(x) = stress(x)/σ_yield − 1 ≤ 0,  lo ≤ x ≤ hi", 15, "DejaVu-Bold", INK_PRIMARY),
    ("OC update each tile e (λ tuned by bisection so G(x_new) ≈ 0):", 10.5, "DejaVu", INK_MUTED),
    ("B_e(λ) = λ·(−∂G/∂xₑ) / (∂V/∂xₑ)", 14.5, "DejaVu", INK_PRIMARY),
    ("xₑ ← clip( xₑ·B_e(λ)^η,  xₑ ± Δ,  [loₑ, hiₑ] )", 14.5, "DejaVu", INK_PRIMARY),
    ("sensitivity filter (Sigmund 1997), suppresses checkerboarding:", 10.5, "DejaVu", INK_MUTED),
    ("s̃ₑ = (1/xₑ) · Σᵢ Hₑᵢxᵢsᵢ  /  Σᵢ Hₑᵢ", 14.5, "DejaVu", INK_PRIMARY),
]


def page_topopt(c):
    draw_base(c)
    draw_title_block(c, "Topology Optimization: the OC Loop",
                      "A SIMP-style optimality-criteria search over per-tile design variables")

    col_top, col_bottom = 133.1, 400.0
    col_gap = 23.6
    col_w = (CONTENT_W - col_gap) / 2
    col_left_x = LEFT
    col_right_x = LEFT + col_w + col_gap
    col_h = col_bottom - col_top
    pad = 17.3

    rounded_card(c, col_left_x, col_top, col_w, col_h)
    avail_w = col_w - 2 * pad
    wrapped = [(wrap_text(text, font, size, avail_w), size, font, color)
               for text, size, font, color in FORMULA_LINES]
    gap_h = 10
    total_h = sum(len(lines) * size * 1.35 for lines, size, _, _ in wrapped) + gap_h * (len(wrapped) - 1)
    y = y_top(col_top) - (col_h - total_h) / 2 - FORMULA_LINES[0][1]
    for lines, size, font, color in wrapped:
        c.setFillColor(color)
        c.setFont(font, size)
        for line in lines:
            c.drawString(col_left_x + pad, y, line)
            y -= size * 1.35
        y -= gap_h

    rounded_card(c, col_right_x, col_top, col_w, col_h)
    img = ImageReader("optimization/artifacts/topopt_before_after.jpg")
    iw, ih = img.getSize()
    img_w = col_w - 20
    img_h = img_w * ih / iw
    caption_h = 26
    block_h = img_h + 8 + caption_h
    img_top = col_top + (col_h - block_h) / 2
    c.drawImage(img, col_right_x + (col_w - img_w) / 2, y_top(img_top, img_h), width=img_w, height=img_h,
                preserveAspectRatio=True, anchor="n", mask="auto")

    c.setFillColor(INK_SECONDARY)
    c.setFont("DejaVu", 10)
    c.drawCentredString(col_right_x + col_w / 2, y_top(img_top + img_h + 8) - 10,
                         "60 generations, CCX-verified: volume 705 → 419 (−41%), stress held at σ_yield")


def page1(c):
    draw_base(c)
    draw_title_block(c, "Kreisselmeier–Steinhauser (KS) Function",
                      "A smooth, differentiable approximation of max()")

    formula_top, formula_h = 133.1, 94.5
    rounded_card(c, LEFT, formula_top, CONTENT_W, formula_h, r=10)
    c.setFillColor(INK_PRIMARY)
    c.setFont("DejaVu-Bold", 24)
    c.drawCentredString(LEFT + CONTENT_W / 2, y_top(formula_top) - 44, "KS(g) = (1/ρ) · ln [ Σ exp(ρ · gᵢ) ]")
    c.setFillColor(INK_SECONDARY)
    c.setFont("DejaVu", 15)
    c.drawCentredString(LEFT + CONTENT_W / 2, y_top(formula_top) - 76, "As ρ → ∞,  KS(g) → max(gᵢ)")

    chart_head_top = formula_top + formula_h + 11.8
    c.setFillColor(ACCENT)
    c.setFont("DejaVu-Bold", 16.5)
    c.drawString(LEFT, y_top(chart_head_top) - 18, "KS(ρ) is a conservative bound: KS(g) ≥ max(gᵢ) for every ρ > 0")

    img_top = chart_head_top + 29.9 + 4.7
    img_h = PAGE_H - 31.5 - img_top
    img = ImageReader(CHART)
    c.drawImage(img, LEFT, y_top(img_top, img_h), width=CONTENT_W, height=img_h,
                preserveAspectRatio=True, anchor="n", mask="auto")


def page2(c):
    draw_base(c)
    draw_title_block(c, "Using KS in the Optimization Loop",
                      "From per-node GNN predictions to a single differentiable constraint")

    draw_two_column_bullets(
        c, 133.1, 400.0,
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

    draw_pipeline(c, 415.6, "How a KS value reaches the optimizer", 453.4, [
        ("GNN node stress", ACCENT),
        ("ks_pool(ρ)", INK_PRIMARY),
        ("graph_pred (scalar)", ACCENT),
        ("Optimizer constraint g(x) ≤ 0", INK_PRIMARY),
    ])


def main():
    c = canvas.Canvas(PATH, pagesize=(PAGE_W, PAGE_H))
    page_topopt(c)
    c.showPage()
    page1(c)
    c.showPage()
    page2(c)
    c.showPage()
    c.save()
    print("wrote", PATH)


if __name__ == "__main__":
    main()
