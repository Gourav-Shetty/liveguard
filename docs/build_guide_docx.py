import docx
from pathlib import Path
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = Path(__file__).resolve().parent

def create_guide_docx():
    doc = docx.Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("LiveGuard-EHMS")
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Hardware, Sensor Ingestion & Edge AI Engineering Guide\nFinal Year Major Project")
    sub_run.font.size = Pt(14)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Helper function for headings
    def add_custom_heading(text, level=1):
        h = doc.add_heading(text, level=level)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        for r in h.runs:
            r.font.name = 'Calibri'
            if level == 1:
                r.font.size = Pt(16)
                r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
                r.font.bold = True
            elif level == 2:
                r.font.size = Pt(13)
                r.font.color.rgb = RGBColor(0x2E, 0x6B, 0x9E)
                r.font.bold = True
        return h

    # Section 1
    add_custom_heading("1. Project Architecture & End-to-End Workflow")
    doc.add_paragraph(
        "LiveGuard-EHMS is an Edge-AI and Federated Healthcare Monitoring System designed for real-time ECG arrhythmia "
        "detection. The system operates on a staged architecture where the Edge device captures biological signals, "
        "applies digital signal filtering, detects R-peaks using the Pan-Tompkins algorithm, and runs on-device inference "
        "using a lightweight 1D-CNN (SmallConv1DGate) model."
    )

    doc.add_paragraph(
        "Key System Stages:\n"
        "1. Real-time Ingestion: 360 Hz ECG sampling (AD8232 via Arduino USB / Raspberry Pi MCP3008 SPI).\n"
        "2. Digital Signal Processing: 0.5–40 Hz Butterworth Bandpass + 50 Hz IIR Notch filter.\n"
        "3. Beat Segmentation: Pan-Tompkins QRS detector extracting normalized 180-sample beat windows.\n"
        "4. Edge Inference: PyTorch Stage 1 CNN classifying Normal vs Abnormal with < 4ms latency.\n"
        "5. Telemetry Broadcasting: WebSocket streaming of raw/filtered waveforms and telemetry to the Web Dashboard.\n"
        "6. Federated Transfer Learning: Local fine-tuning and periodic model weight synchronization with Flower FedAvg server."
    )

    # Section 2
    add_custom_heading("2. Hardware Wiring & Pinout Specifications")
    add_custom_heading("2.1 Arduino Uno / Nano Bridge Wiring (Immediate Setup)", level=2)
    
    table1 = doc.add_table(rows=1, cols=4)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table1.rows[0].cells
    headers = ["AD8232 Sensor Pin", "Arduino Pin", "Wire Color", "Function"]
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="E8EEF5"/>')
        hdr_cells[i]._tc.get_or_add_tcPr().append(shading)

    wiring_data = [
        ["3.3V", "3.3V", "Red", "Power supply (Do NOT connect to 5V)"],
        ["GND", "GND", "Black", "System ground"],
        ["OUTPUT", "Pin A0", "Yellow / White", "Analog ECG signal output"],
        ["LO+", "Digital Pin 2", "Blue", "Leads-Off Detection Positive"],
        ["LO-", "Digital Pin 3", "Green", "Leads-Off Detection Negative"],
        ["SDN", "Unconnected", "—", "Shutdown pin (Not needed)"]
    ]

    for row in wiring_data:
        r_cells = table1.add_row().cells
        for i, val in enumerate(row):
            r_cells[i].text = val
            r_cells[i].paragraphs[0].runs[0].font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_custom_heading("2.2 Raspberry Pi 4 (1GB) + MCP3008 ADC Wiring (When Pi Arrives)", level=2)
    table2 = doc.add_table(rows=1, cols=3)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells2 = table2.rows[0].cells
    headers2 = ["Component Pin", "Raspberry Pi 4 Pin", "Function / SPI Channel"]
    for i, h in enumerate(headers2):
        hdr_cells2[i].text = h
        hdr_cells2[i].paragraphs[0].runs[0].font.bold = True
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="E8EEF5"/>')
        hdr_cells2[i]._tc.get_or_add_tcPr().append(shading)

    rpi_data = [
        ["MCP3008 VDD & VREF", "Pin 1 (3.3V)", "Power & ADC reference voltage"],
        ["MCP3008 AGND & DGND", "Pin 6 (GND)", "Analog & digital ground"],
        ["MCP3008 CLK", "Pin 23 (GPIO 11 / SPI0_SCLK)", "SPI Clock"],
        ["MCP3008 DOUT", "Pin 21 (GPIO 9 / SPI0_MISO)", "Master In Slave Out"],
        ["MCP3008 DIN", "Pin 19 (GPIO 10 / SPI0_MOSI)", "Master Out Slave In"],
        ["MCP3008 CS/SHDN", "Pin 24 (GPIO 8 / SPI0_CE0)", "Chip Select 0"],
        ["MCP3008 CH0", "AD8232 OUTPUT", "Analog ECG channel input"],
        ["AD8232 3.3V & GND", "Pin 17 (3.3V) & Pin 14 (GND)", "Sensor power"],
        ["AD8232 LO+ & LO-", "Pin 16 (GPIO 23) & Pin 18 (GPIO 24)", "Leads-off detection"]
    ]

    for row in rpi_data:
        r_cells = table2.add_row().cells
        for i, val in enumerate(row):
            r_cells[i].text = val
            r_cells[i].paragraphs[0].runs[0].font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Section 3
    add_custom_heading("3. 3-Lead ECG Electrode Placement on the Body")
    doc.add_paragraph(
        "For optimal signal-to-noise ratio (SNR) and clear R-peak detection, use Einthoven's Triangle configuration:\n"
        "• RA (Right Arm / Red Electrode): Place on the right infraclavicular fossa (just below right collarbone) or right inner wrist.\n"
        "• LA (Left Arm / Yellow Electrode): Place on the left infraclavicular fossa (just below left collarbone) or left inner wrist.\n"
        "• RL (Right Leg / Green Electrode): Place on the lower right abdominal region. This serves as the reference ground."
    )
    doc.add_paragraph(
        "Practical Tip for Clean Signal: Clean application areas with alcohol wipes to remove skin oils. "
        "Keep the subject relaxed and seated during acquisition to prevent electromyographic (EMG) muscle artifacts."
    )

    # Section 4
    add_custom_heading("4. Flashing Arduino Firmware & Running the Pipeline")
    doc.add_paragraph(
        "Step 1: Open Arduino IDE and load: backend/arduino_bridge/liveguard_sensor_bridge.ino\n"
        "Step 2: Connect Arduino via USB, select Board (Arduino Uno/Nano) and COM Port (e.g. COM3).\n"
        "Step 3: Click Upload (Right Arrow button).\n"
        "Step 4: Install Python dependencies on your laptop:\n"
        "         pip install pyserial websockets torch scipy numpy\n"
        "Step 5: Run the LiveGuard edge engine:\n"
        "         python -m backend.run_edge --source ARDUINO --port COM3"
    )

    # Section 5
    add_custom_heading("5. Operating Modes Summary")
    doc.add_paragraph(
        "1. ARDUINO Mode (--source ARDUINO):\n"
        "   Reads live sensor stream from Arduino USB COM port at 360 Hz.\n\n"
        "2. MOCK Mode (--source MOCK):\n"
        "   Simulates continuous 360 Hz patient data using synthetic P-Q-R-S-T generator or MIT-BIH recordings. "
        "Ideal for developing without physical hardware.\n\n"
        "3. RPI_SPI Mode (--source RPI_SPI):\n"
        "   Direct hardware execution on Raspberry Pi 4 using spidev and MCP3008 ADC."
    )

    # Section 6
    add_custom_heading("6. Division of Work & Responsibilities")
    doc.add_paragraph(
        "Hardware & Edge Lead (You):\n"
        "• Sensor interfacing (AD8232 + MAX30102 + Arduino/MCP3008).\n"
        "• Digital signal processing (Bandpass, Notch filter, Pan-Tompkins QRS detection).\n"
        "• Edge inference pipeline & WebSocket telemetry server.\n"
        "• Web Dashboard frontend connection.\n\n"
        "Machine Learning & Federated Learning Lead (You):\n"
        "• Flower Federated Learning ServerApp & ClientApp.\n"
        "• Centralized baseline benchmarking & non-IID patient partitioning.\n"
        "• Stage 1 threshold calibration (Recall >= 0.95).\n"
        "• Stage 2 multiclass arrhythmia classifier (AAMI N, S, V, F, Q categories)."
    )

    output_path = str(BASE_DIR / "LiveGuard_Hardware_Guide.docx")
    doc.save(output_path)
    print(f"Successfully generated Word document at: {output_path}")

if __name__ == "__main__":
    create_guide_docx()
