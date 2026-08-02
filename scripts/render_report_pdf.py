"""Render the report DOCX content to a polished PDF when LibreOffice is absent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "report" / "CIS6005_WRIT1_MEDA_Report.docx"
PDF = ROOT / "output" / "pdf" / "CIS6005_WRIT1_MEDA_Report.pdf"
FIG = ROOT / "outputs" / "figures"

FIGURES = {
    "Figure 1.": FIG / "eda_pressure_distribution.png",
    "Figure 2.": FIG / "eda_missingness.png",
    "Figure 3.": FIG / "eda_pressure_by_sol.png",
    "Figure 4.": FIG / "eda_pressure_by_lmst.png",
    "Figure 5.": FIG / "eda_correlation.png",
    "Figure 6.": FIG / "architecture_training.png",
    "Figure 7.": FIG / "architecture_inference.png",
    "Figure 8.": FIG / "evaluation_residuals.png",
    "Figure 9.": FIG / "evaluation_error_by_sol.png",
    "Figure A1.": ROOT / "outputs" / "evidence" / "kaggle_submissions.png",
}


def escaped(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5A6770"))
    canvas.drawCentredString(letter[0] / 2, letter[1] - 0.48 * inch,
                             "CIS6005 Computational Intelligence | MEDA Virtual Sensor Recovery")
    canvas.drawRightString(letter[0] - inch, 0.48 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    PDF.parent.mkdir(parents=True, exist_ok=True)
    source = Document(DOCX)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("BodyMEDA", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4,
                          leading=12.4, alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("H1MEDA", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                        leading=18, textColor=colors.HexColor("#2E74B5"), spaceBefore=13, spaceAfter=7, keepWithNext=True)
    h2 = ParagraphStyle("H2MEDA", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                        leading=15, textColor=colors.HexColor("#2E74B5"), spaceBefore=10, spaceAfter=5, keepWithNext=True)
    h3 = ParagraphStyle("H3MEDA", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10.5,
                        leading=13, textColor=colors.HexColor("#1F4D78"), spaceBefore=7, spaceAfter=4, keepWithNext=True)
    caption = ParagraphStyle("CaptionMEDA", parent=body, fontSize=8.2, leading=10, alignment=TA_CENTER,
                             textColor=colors.HexColor("#5A6770"), spaceAfter=6)
    reference = ParagraphStyle("ReferenceMEDA", parent=body, fontSize=8.1, leading=10.2, leftIndent=16,
                               firstLineIndent=-16, alignment=TA_LEFT, spaceAfter=4)
    title = ParagraphStyle("TitleMEDA", fontName="Helvetica-Bold", fontSize=25, leading=29,
                           textColor=colors.HexColor("#173346"), alignment=TA_CENTER, spaceAfter=10)
    subtitle = ParagraphStyle("SubtitleMEDA", fontName="Helvetica", fontSize=12.5, leading=16,
                              textColor=colors.HexColor("#1F4D78"), alignment=TA_CENTER, spaceAfter=15)
    kicker = ParagraphStyle("KickerMEDA", fontName="Helvetica-Bold", fontSize=9.5, leading=12,
                            textColor=colors.HexColor("#B7653D"), alignment=TA_CENTER, spaceAfter=16)
    metadata = ParagraphStyle("MetadataMEDA", fontName="Helvetica", fontSize=10, leading=15,
                              textColor=colors.HexColor("#5A6770"), alignment=TA_CENTER)

    story = [Spacer(1, 1.25 * inch)]
    in_references = False
    for paragraph in source.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if text == "COMPUTATIONAL INTELLIGENCE PROJECT":
            story.append(Paragraph(text, kicker)); continue
        if text.startswith("MEDA Atmospheric Pressure"):
            story.append(Paragraph(escaped(text).replace("\n", "<br/>"), title)); continue
        if text.startswith("Temporally honest"):
            story.append(Paragraph(escaped(text), subtitle)); continue
        if text.startswith("Student: [STUDENT NAME]"):
            story.append(Spacer(1, .7 * inch)); story.append(Paragraph(escaped(text).replace("\n", "<br/>"), metadata)); story.append(PageBreak()); continue
        style_name = paragraph.style.name if paragraph.style else "Normal"
        if style_name == "Heading 1":
            if text == "References": in_references = True
            if text.startswith("Appendix "): story.append(PageBreak())
            story.append(Paragraph(escaped(text), h1)); continue
        if style_name == "Heading 2":
            story.append(Paragraph(escaped(text), h2)); continue
        if style_name == "Heading 3":
            story.append(Paragraph(escaped(text), h3)); continue
        matched = next((key for key in FIGURES if text.startswith(key)), None)
        if matched:
            path = FIGURES[matched]
            if path.exists():
                width = 6.2 * inch
                image = Image(str(path)); ratio = image.imageHeight / image.imageWidth
                image.drawWidth = width; image.drawHeight = min(width * ratio, 6.2 * inch)
                story.extend([Spacer(1, 4), image, Spacer(1, 3)])
            story.append(Paragraph(escaped(text), caption)); continue
        if text.startswith("Table 1."):
            story.append(Paragraph(escaped(text), caption))
            comparison = pd.read_csv(ROOT / "outputs" / "model_comparison_v2.csv")
            test = comparison[comparison.split.eq("validation")].sort_values("mse")
            data = [["Model", "MSE", "RMSE", "MAE", "R2"]] + [[row.model, f"{row.mse:.3f}",
                    f"{row.rmse:.3f}", f"{row.mae:.3f}", f"{row.r2:.3f}"] for _, row in test.iterrows()]
            table = Table(data, colWidths=[1.65*inch, .85*inch, .85*inch, .85*inch, .85*inch], repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F2F4F7")),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
                ("FONTSIZE",(0,0),(-1,-1),7.5),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#AAB4BC")),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),6),
                ("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
            story.extend([table, Spacer(1, 8)]); continue
        story.append(Paragraph(escaped(text), reference if in_references else body))

    pdf = SimpleDocTemplate(str(PDF), pagesize=letter, rightMargin=inch, leftMargin=inch,
                            topMargin=.72*inch, bottomMargin=.72*inch,
                            title="MEDA Atmospheric Pressure Virtual Sensor Recovery",
                            author="[STUDENT NAME]")
    pdf.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(PDF)


if __name__ == "__main__":
    main()
