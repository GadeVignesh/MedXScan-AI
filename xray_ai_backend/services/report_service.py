from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
import os
from datetime import datetime

def generate_pdf_report(filename, predictions, confidences, heatmap_path):

    os.makedirs("reports", exist_ok=True)

    report_name = f"reports/MedXScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(report_name)

    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>MedXScan AI Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"File: {filename}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    for p, c in zip(predictions, confidences):
        elements.append(Paragraph(f"Disease: {p} | Confidence: {c}", styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))

    if heatmap_path and os.path.exists(heatmap_path):
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Image(heatmap_path, width=4 * inch, height=4 * inch))

    doc.build(elements)

    return report_name
