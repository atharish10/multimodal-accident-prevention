#!/usr/bin/env python3
"""
Compiles the publication-grade IEEE Double-Column Base Paper PDF using headless Google Chrome.
Source HTML: paper/Multimodal_Accident_Prevention_Base_Paper.html
Output PDF:  paper/Multimodal_Accident_Prevention_Base_Paper.pdf
"""

import os
import subprocess
from pathlib import Path

base_dir = Path("/home/harish/multimodal-accident-prevention")
paper_dir = base_dir / "paper"
html_path = paper_dir / "Multimodal_Accident_Prevention_Base_Paper.html"
pdf_path = paper_dir / "Multimodal_Accident_Prevention_Base_Paper.pdf"

if not html_path.exists():
    raise FileNotFoundError(f"HTML paper source not found: {html_path}")

cmd = [
    "google-chrome",
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=6000",
    f"--print-to-pdf={pdf_path}",
    str(html_path)
]

print("Compiling IEEE Base Paper PDF via headless Chrome...")
res = subprocess.run(cmd, capture_output=True, text=True)

if pdf_path.exists():
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"SUCCESS: Generated {pdf_path} ({size_mb:.2f} MB)")
else:
    print(f"ERROR: Failed to generate PDF: {res.stderr}")
    exit(1)
