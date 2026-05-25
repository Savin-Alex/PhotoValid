"""Generate a PDF validation report from a summary + the uploaded photo.

Pure layout: takes the already-computed validation summary (see
main._summarize_response) and the PIL image, and renders a one/two-page PDF.
Stateless — nothing is persisted.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_STATUS_COLOR = {
    "pass": colors.HexColor("#16a34a"),
    "warning": colors.HexColor("#f59e0b"),
    "fail": colors.HexColor("#ef4444"),
    "skipped": colors.HexColor("#6b7280"),
}
_STATUS_LABEL = {"pass": "PASS", "warning": "WARN", "fail": "FAIL", "skipped": "REVIEW"}


def _find_face_box(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for check in summary.get("biometric", []):
        fb = check.get("faceBox")
        if isinstance(fb, dict):
            return fb
    return None


def _annotated_image(pil: Image.Image, summary: Dict[str, Any], max_px: int = 460) -> tuple[io.BytesIO, float]:
    """Return (PNG buffer, height/width ratio) of the photo with guide lines drawn."""
    img = pil.convert("RGB").copy()
    w, h = img.size
    fb = _find_face_box(summary)
    if fb:
        draw = ImageDraw.Draw(img)
        line_w = max(2, h // 300)
        for key, color in (("top", (37, 99, 235)), ("eyeY", (22, 163, 74)), ("bottom", (239, 68, 68))):
            y = fb.get(key)
            if isinstance(y, (int, float)) and 0 <= y < h:
                draw.line([(0, int(y)), (w, int(y))], fill=color, width=line_w)
    img.thumbnail((max_px, max_px))
    tw, th = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf, (th / tw if tw else 1.0)


def _section_table(title: str, checks: List[Dict[str, Any]], styles) -> List[Any]:
    if not checks:
        return []
    elems: List[Any] = [Spacer(1, 6), Paragraph(title, styles["section"])]
    cell = styles["cell"]
    data = [["Check", "Result", "Expected", "Status"]]
    row_colors = []
    for c in checks:
        status = c.get("status", "fail")
        data.append([
            Paragraph(str(c.get("name", "")), cell),
            Paragraph(str(c.get("value", "") or "—"), cell),
            Paragraph(str(c.get("expected", "") or "—"), cell),
            Paragraph(_STATUS_LABEL.get(status, status.upper()), cell),
        ])
        row_colors.append(_STATUS_COLOR.get(status, colors.grey))

    table = Table(data, colWidths=[42 * mm, 45 * mm, 50 * mm, 18 * mm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, color in enumerate(row_colors, start=1):
        style.append(("TEXTCOLOR", (3, i), (3, i), color))
        style.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    elems.append(table)
    return elems


def build_report_pdf(pil: Image.Image, summary: Dict[str, Any]) -> bytes:
    out = io.BytesIO()
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
        title="DV Lottery Photo Validation Report", author="PhotoValid",
    )
    base = getSampleStyleSheet()
    styles = {
        "title": base["Title"],
        "section": ParagraphStyle("section", parent=base["Heading3"], spaceBefore=4, spaceAfter=2),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8, leading=10),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8, textColor=colors.grey),
    }

    status = str(summary.get("status", "")).lower()
    score = summary.get("overall_score", 0)
    sc = _STATUS_COLOR.get(status, colors.grey)
    result_style = ParagraphStyle("result", parent=base["Heading2"], textColor=sc)

    elems: List[Any] = [
        Paragraph("DV Lottery Photo Validation Report", styles["title"]),
        Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["small"]),
        Spacer(1, 6),
        Paragraph(f"Result: {status.upper() or 'N/A'} &nbsp;•&nbsp; Score: {score}%", result_style),
    ]

    # Annotated photo
    try:
        img_buf, ratio = _annotated_image(pil, summary)
        disp_w = 60 * mm
        elems.append(Spacer(1, 4))
        elems.append(RLImage(img_buf, width=disp_w, height=disp_w * ratio))
        elems.append(Paragraph("Blue = top of head, green = eye level, red = chin (when detected).", styles["small"]))
    except Exception:
        pass

    elems += _section_table("Technical", summary.get("technical", []), styles)
    elems += _section_table("Biometric & Composition", summary.get("biometric", []), styles)
    elems += _section_table("Tamper / Integrity", summary.get("tamper", []), styles)

    manual = summary.get("manual_review") or []
    if manual:
        elems.append(Spacer(1, 6))
        elems.append(Paragraph("Manual review (not auto-verified)", styles["section"]))
        for item in manual:
            elems.append(Paragraph(f"• {item}", styles["cell"]))

    elems.append(Spacer(1, 10))
    elems.append(Paragraph(
        "Automated analysis based on publicly available DV Lottery photo requirements. "
        "Always verify against official U.S. Department of State guidance before submitting. "
        "This tool does not guarantee acceptance.",
        styles["small"],
    ))

    doc.build(elems)
    return out.getvalue()
