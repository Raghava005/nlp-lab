from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
import json

def create_pdf(result,path):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Hero",parent=styles["Title"],textColor=colors.HexColor("#8b5cf6"),alignment=TA_CENTER,spaceAfter=8))
    styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8,leading=11))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=16*mm,bottomMargin=16*mm)
    story=[Paragraph("NLP LAB — EXPERIMENT REPORT",styles["Hero"]),Paragraph(result.get("title","NLP Experiment"),styles["Heading1"]),
           Paragraph(f"Run ID: {result.get('id','')} &nbsp;&nbsp; | &nbsp;&nbsp; Created: {result.get('created_at','')}",styles["Small"]),Spacer(1,8),
           Paragraph("Summary",styles["Heading2"]),Paragraph(result.get("summary",""),styles["BodyText"]),Spacer(1,8)]
    m=result.get("metrics",{})
    if m:
        data=[["Metric","Value"],["Accuracy",f"{m.get('accuracy',0):.4f}"],["Macro precision",f"{m.get('macro_precision',0):.4f}"],["Macro recall",f"{m.get('macro_recall',0):.4f}"],["Macro F1",f"{m.get('macro_f1',0):.4f}"]]
        t=Table(data,colWidths=[70*mm,70*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#252047")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d1d5db")),("PADDING",(0,0),(-1,-1),7)])); story += [Paragraph("Evaluation metrics",styles["Heading2"]),t,Spacer(1,10)]
        if m.get("confusion_matrix"):
            labels=m.get("labels",[]); cm=m["confusion_matrix"]
            data=[["Actual \\ Predicted"]+labels]+[[labels[i]]+[str(v) for v in row] for i,row in enumerate(cm)]
            t=Table(data,repeatRows=1); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#252047")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.grey),("PADDING",(0,0),(-1,-1),5)])); story += [Paragraph("Confusion matrix",styles["Heading2"]),t,Spacer(1,10)]
    if result.get("pairs"):
        story += [Paragraph("Sentence pair similarity",styles["Heading2"])]
        for i,p in enumerate(result["pairs"][:40],1):
            story.append(Paragraph(f"{i}. {p['sentence1']} <b>↔</b> {p['sentence2']} — cosine: {p['cosine_similarity']:.4f}",styles["Small"]))
    if result.get("notes"):
        story += [Paragraph("Notes and interpretation",styles["Heading2"])]
        for n in result["notes"]: story.append(Paragraph("• "+n,styles["Small"]))
    story += [Spacer(1,12),Paragraph("Generated locally by NLP Lab. Interpret metrics in the context of dataset size, quality, label balance and split strategy.",styles["Small"])]
    doc.build(story)
