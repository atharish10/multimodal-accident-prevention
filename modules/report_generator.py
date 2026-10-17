"""
Official PDF Incident & Assessment Report Generator
Panimalar Engineering College - Department of AI & ML
Multimodal Accident Prevention & Severe Damage Assessment System
"""

import io
import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from PIL import Image as PILImage


def generate_incident_pdf(
    incident_id: str,
    modality_type: str,
    overall_risk: float,
    status_label: str,
    damage_results: dict = None,
    violation_results: dict = None,
    hazard_results: dict = None,
    annotated_image: PILImage.Image = None,
    environmental_factors: dict = None,
    notes: str = ""
) -> bytes:
    """
    Generates a formal, professional incident report in PDF format.
    Returns bytes suitable for Streamlit download buttons.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor("#0B2545")
    ACCENT_BLUE = colors.HexColor("#134074")
    TEXT_DARK = colors.HexColor("#1D2D44")
    BG_LIGHT = colors.HexColor("#EEF4F8")
    BORDER_COLOR = colors.HexColor("#8DA9C4")
    
    # Severity Colors
    if overall_risk >= 70:
        STATUS_COLOR = colors.HexColor("#C1121F")
        STATUS_BG = colors.HexColor("#FDF0ED")
    elif overall_risk >= 40:
        STATUS_COLOR = colors.HexColor("#D97706")
        STATUS_BG = colors.HexColor("#FEF3C7")
    else:
        STATUS_COLOR = colors.HexColor("#059669")
        STATUS_BG = colors.HexColor("#ECFDF5")

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=NAVY,
        alignment=0
    )
    
    sub_title_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=ACCENT_BLUE,
        alignment=0
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=NAVY,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK
    )

    body_bold = ParagraphStyle(
        'BodyBold_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK
    )

    status_style = ParagraphStyle(
        'StatusText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=STATUS_COLOR,
        alignment=1
    )

    elements = []

    # 1. Header with College Logo and Title
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "college_logo.png")
    header_cells = []
    if os.path.exists(logo_path):
        try:
            logo_img = RLImage(logo_path, width=54, height=54)
            header_cells = [[
                logo_img,
                [
                    Paragraph("PANIMALAR ENGINEERING COLLEGE", title_style),
                    Paragraph("An Autonomous Institution, Affiliated to Anna University, Chennai", sub_title_style),
                    Paragraph("Department of Artificial Intelligence & Machine Learning | D16 Review Showcase", sub_title_style),
                    Paragraph("<b>Project:</b> Multimodal Accident Prevention & Severe Damage Assessment System", sub_title_style)
                ]
            ]]
        except Exception:
            header_cells = [[
                "",
                [
                    Paragraph("PANIMALAR ENGINEERING COLLEGE", title_style),
                    Paragraph("Department of Artificial Intelligence & Machine Learning", sub_title_style),
                    Paragraph("Multimodal Accident Prevention & Severe Damage Assessment System", sub_title_style)
                ]
            ]]
    else:
        header_cells = [[
            "",
            [
                Paragraph("PANIMALAR ENGINEERING COLLEGE", title_style),
                Paragraph("Department of Artificial Intelligence & Machine Learning", sub_title_style),
                Paragraph("Multimodal Accident Prevention & Severe Damage Assessment System", sub_title_style)
            ]
        ]]

    header_table = Table(header_cells, colWidths=[65, 475])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8, spaceBefore=4))

    # 2. Metadata Box
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta_data = [
        [
            Paragraph("<b>Incident Ref ID:</b>", body_style), Paragraph(incident_id, body_bold),
            Paragraph("<b>Timestamp:</b>", body_style), Paragraph(now_str, body_style)
        ],
        [
            Paragraph("<b>Assessment Modality:</b>", body_style), Paragraph(modality_type, body_bold),
            Paragraph("<b>Edge Device / Node:</b>", body_style), Paragraph("Edge-Cluster-Node-04 (GPU Enabled)", body_style)
        ],
        [
            Paragraph("<b>Investigators:</b>", body_style), Paragraph("Harish AT & Barathraj AS (AIML)", body_style),
            Paragraph("<b>Faculty Supervisor:</b>", body_style), Paragraph("Dr. A. Arun, Assoc. Professor / AIML", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[100, 170, 100, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 8))

    # 3. Overall Triage Status Box
    status_box = [
        [
            Paragraph(f"<b>TRIAGE STATUS: {status_label.upper()}</b>", status_style),
            Paragraph(f"<b>COMPOSITE RISK INDEX: {overall_risk:.1f}%</b>", status_style)
        ]
    ]
    status_table = Table(status_box, colWidths=[270, 270])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), STATUS_BG),
        ('BOX', (0, 0), (-1, -1), 1.5, STATUS_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(status_table)
    elements.append(Spacer(1, 10))

    # 4. Modality Breakdown Table
    elements.append(Paragraph("Multimodal Telemetry & Risk Dimensions", h2_style))
    modality_rows = [
        [Paragraph("<b>Risk Modality</b>", body_bold), Paragraph("<b>Observed Metrics & Sub-Scores</b>", body_bold), Paragraph("<b>Severity</b>", body_bold)]
    ]

    # Vehicle Damage
    if damage_results:
        raw_score = damage_results.get("severity_score", 0.0)
        dmg_pct = raw_score * 100.0 if raw_score <= 1.0 else raw_score
        dmg_class = damage_results.get("damage_class", damage_results.get("severity_label", "Structural Impact"))
        area_pct = damage_results.get("damage_area_percent", 0.0)
        repair_est = damage_results.get("repair_estimate", "Standard Workshop Repair")
        desc = f"{dmg_class} | Area: {area_pct:.1f}% | Est: {repair_est}"
        modality_rows.append([
            Paragraph("Structural Damage (CNN)", body_style),
            Paragraph(desc, body_style),
            Paragraph(f"{dmg_pct:.1f}%", body_bold)
        ])
    else:
        modality_rows.append([
            Paragraph("Structural Damage", body_style),
            Paragraph("Not tested in this transaction session", body_style),
            Paragraph("N/A", body_style)
        ])

    # Traffic Violations
    if violation_results:
        raw_viol = violation_results.get("violation_score", violation_results.get("risk_score", 0.0))
        viol_score = raw_viol * 100.0 if raw_viol <= 1.0 else raw_viol
        viols = violation_results.get("violations_detected", violation_results.get("violations", []))
        viol_txt = ", ".join(viols) if viols else "No safety violations registered"
        speed_val = violation_results.get("speed_kmh")
        speed_txt = f" (Speed: {speed_val:.1f} km/h)" if speed_val is not None else ""
        modality_rows.append([
            Paragraph("Traffic Violations (YOLOv8)", body_style),
            Paragraph(f"Detected: {viol_txt}{speed_txt}", body_style),
            Paragraph(f"{viol_score:.1f}%", body_bold)
        ])
    else:
        modality_rows.append([
            Paragraph("Traffic Violations", body_style),
            Paragraph("Not tested in this transaction session", body_style),
            Paragraph("N/A", body_style)
        ])

    # Road Hazards
    if hazard_results:
        raw_haz = hazard_results.get("hazard_score", hazard_results.get("hazard_risk_score", 0.0))
        haz_score = raw_haz * 100.0 if raw_haz <= 1.0 else raw_haz
        hazards = hazard_results.get("hazards_detected", [])
        haz_list = []
        for h in hazards:
            if isinstance(h, dict):
                h_type = h.get("type", "Pothole").capitalize()
                h_sev = h.get("severity", "DEFECT")
                haz_list.append(f"{h_type} ({h_sev})")
            else:
                haz_list.append(str(h))
        haz_txt = ", ".join(haz_list) if haz_list else "Road surface clear"
        pavement = hazard_results.get("pavement_state")
        pave_txt = f" | {pavement}" if pavement else ""
        modality_rows.append([
            Paragraph("Road Hazards & Surface", body_style),
            Paragraph(f"Detected: {haz_txt}{pave_txt}", body_style),
            Paragraph(f"{haz_score:.1f}%", body_bold)
        ])
    else:
        modality_rows.append([
            Paragraph("Road Hazards", body_style),
            Paragraph("Not tested in this transaction session", body_style),
            Paragraph("N/A", body_style)
        ])

    # Environmental Factors if present
    if environmental_factors:
        env_weather = environmental_factors.get("weather", "Clear")
        env_light = environmental_factors.get("lighting", "Daylight")
        env_traffic = environmental_factors.get("traffic_density", "Moderate")
        env_desc = f"Atmosphere: {env_weather} | Lighting: {env_light} | Density: {env_traffic}"
        modality_rows.append([
            Paragraph("Environmental Factors", body_style),
            Paragraph(env_desc, body_style),
            Paragraph(f"Coeff: {environmental_factors.get('multiplier', 1.0):.2f}x", body_style)
        ])

    breakdown_table = Table(modality_rows, colWidths=[140, 320, 80])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(breakdown_table)
    elements.append(Spacer(1, 10))

    # 5. Visual Evidence (Annotated Capture)
    if annotated_image is not None:
        try:
            # Resize image to fit nicely within printable page width
            img_byte_arr = io.BytesIO()
            # Convert RGBA to RGB if necessary
            if annotated_image.mode in ("RGBA", "P"):
                annotated_image = annotated_image.convert("RGB")
            annotated_image.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            
            elements.append(Paragraph("Computer Vision Telemetry Snapshot", h2_style))
            # Fit inside 400x160 box maintaining aspect ratio
            w, h = annotated_image.size
            aspect = h / max(1, w)
            target_w = 320
            target_h = int(target_w * aspect)
            if target_h > 150:
                target_h = 150
                target_w = int(target_h / aspect)
            
            rl_img = RLImage(img_byte_arr, width=target_w, height=target_h)
            img_table = Table([[rl_img]], colWidths=[540])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
                ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(img_table)
            elements.append(Spacer(1, 10))
        except Exception:
            pass

    # 6. Actionable Triage & Emergency Response Recommendations
    elements.append(Paragraph("Automated Dispatch & Safety Action Protocol", h2_style))
    
    if overall_risk >= 70:
        protocol_items = [
            [Paragraph("<b>🚨 Emergency Medical:</b> Immediate dispatch of Tier-1 Advanced Life Support (ALS) Ambulance.", body_style)],
            [Paragraph("<b>🚓 Traffic Command:</b> Automatic highway incident geofence alert; deploy nearest sector patrol.", body_style)],
            [Paragraph("<b>🛠️ Roadway Clearance:</b> Dispatch heavy-duty towing & hazmat surface containment unit.", body_style)],
            [Paragraph("<b>🛡️ Insurance Telemetry:</b> Pre-loss cryptographic snapshot logged for accelerated surveyor audit.", body_style)],
        ]
    elif overall_risk >= 40:
        protocol_items = [
            [Paragraph("<b>⚠️ Driver Warning:</b> Multi-tone audible alert emitted; automatic speed advisory active.", body_style)],
            [Paragraph("<b>🚓 Traffic Management:</b> Moderate risk threshold flagged to regional traffic monitoring center.", body_style)],
            [Paragraph("<b>🔧 Service Advisory:</b> Schedule structural check & recalibrate active collision sensors.", body_style)],
        ]
    else:
        protocol_items = [
            [Paragraph("<b>✅ Nominal Operating State:</b> Standard continuous multi-sensor telemetry running.", body_style)],
            [Paragraph("<b>ℹ️ Routine Logging:</b> Event logged in edge-blackbox rolling buffer (24-hour cycle).", body_style)],
        ]

    action_table = Table(protocol_items, colWidths=[540])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor("#DDE5ED")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(action_table)
    elements.append(Spacer(1, 10))

    # 7. Investigator Notes & Forensic Survey Data
    if notes:
        elements.append(Paragraph("Investigator Findings & Forensic Survey Notes", h2_style))
        formatted_notes = notes.replace("\n", "<br/>")
        notes_table = Table([[Paragraph(formatted_notes, body_style)]], colWidths=[540])
        notes_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(notes_table)
        elements.append(Spacer(1, 10))

    # 8. Verification & Sign-off Footer
    elements.append(HRFlowable(width="100%", thickness=0.8, color=BORDER_COLOR, spaceAfter=6, spaceBefore=0))
    footer_text = (
        "This is an automated AI decision support report generated by the Multimodal Accident Prevention "
        "System at Panimalar Engineering College (AIML Dept). Real-time inference executed on edge nodes. "
        "Verified for academic review and operational triage simulation."
    )
    elements.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=6.5, leading=9, textColor=colors.gray, alignment=1)))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
