📡 4G LTE Drive Test Worst Cell Analyzer
A robust, production-grade Streamlit application designed for RF Optimization Engineers and Telecommunications Data Analysts. This tool automates the post-processing of TEMS Discovery Drive Test (DT) exports, correlates them with the 4G Physical Database and VoLTE exports, and generates comprehensive Worst Cell analysis reports.
✨ Core Features
Multi-Domain Analysis: Evaluates network performance across four critical domains:
📶 RF Analysis (PS + CS Data)
📊 Throughput Analysis (PS Data Only)
🗣️ MOS / Voice Quality Analysis (CS Data Only)
📞 VoLTE Analysis (Dedicated VoLTE Export)
Cell-Centric Aggregation: Eliminates statistical noise and duplicate cell entries by aggregating raw time-bin/ping data into unique logical cells (eNB_ID + Cell_ID) using sample-weighted averages.
Strict Domain Isolation: Ensures CS voice data never contaminates Throughput metrics, and PS data is strictly isolated for data KPI evaluations.
Automated Root Cause Classification: Identifies Overshooters, Non-Dominant Servers, Poor Coverage, and Poor Quality cells based on configurable RF thresholds.
Multi-Cluster Support: Seamlessly processes single or multiple cluster/region exports.
🔄 Data Flow & Architecture
The application follows a strict Cell-Centric Data Model to ensure KPI accuracy:
text

12345
Ingestion & Isolation: Data streams are separated into PS, CS, and VoLTE domains.
Aggregation: Raw rows (which may contain multiple time-bins or pings per cell) are grouped by eNB_Part and Cell_Part. Averages (RSRP, SINR, Throughput) are calculated using the count column as a weight.
Database Join: Aggregated cell data is left-joined with the 4G Physical Database for site metadata (Azimuth, Height, PCI, etc.).
Classification: Cells are evaluated against Worst Cell definitions (e.g., RSRP ≤ -95dBm AND SINR ≤ 0dB).
📥 Required Inputs
To generate the full report, the application requires the following Excel (.xlsx) files:
Input File
Description
Used For
TEMS PS Export
Packet-Switched Drive Test data (FTP, Ping, Web Browsing)
RF (Combined), Throughput
TEMS CS Export
Circuit-Switched Voice Drive Test data (RxQual, MOS)
RF (Combined), MOS
4G Database
Physical site database (eNB_ID, Cell_ID, PCI, Azimuth, etc.)
Metadata & Site Info
VoLTE Export
VoLTE specific KPI export (RTP Loss, VoLTE MOS, Drop Rate)
VoLTE Analysis
🛠️ Installation & Setup
Prerequisites
Python 3.8 or higher
pip package manager
1. Clone / Download the Repository
bash

12
2. Install Dependencies
Create a virtual environment (recommended) and install the required packages:
bash

1
(Or use pip install -r requirements.txt if a requirements file is provided).
3. Run the Application
Launch the Streamlit web interface:
bash

1
The application will automatically open in your default web browser (usually at http://localhost:8501).
🖥️ How to Use
Upload Files: Use the sidebar or main dashboard to upload the TEMS PS, TEMS CS, 4G Database, and VoLTE Export files.
Configure Thresholds: Adjust the Worst Cell definitions if needed (e.g., change Poor Coverage RSRP threshold from -95 to -100).
Process Data: Click the "Generate Worst Cell Report" button.
Download: Once processing is complete, download the generated .xlsx report containing 8 distinct tabs.
📑 Output Structure
The generated Excel file (WorstCells_Output.xlsx) contains 8 sheets, divided by domain and classification:
📶 RF Analysis (PS + CS Combined)
RF Worst Cells: Cells failing RF thresholds (Poor RSRP/SINR).
RF All Cells: Baseline of all unique tested cells.
📊 Throughput Analysis (PS Only)
Throughput Worst Cells: Cells with poor RSRP/SINR AND DL Throughput < 30 Mbps.
Throughput All Cells: Baseline of all unique PS-tested cells.
🗣️ MOS Analysis (CS Only)
MOS Worst Cells: Cells with poor RF AND MOS ≤ 3.5.
MOS All Cells: Baseline of all unique CS-tested cells.
📞 VoLTE Analysis (VoLTE Export)
VoLTE Worst Cells: Cells failing VoLTE specific KPIs (e.g., Poor RTP Loss / Status != "RF is good").
VoLTE All Cells: Baseline of all VoLTE-tested cells.
⚠️ Important Engineering Notes
Cell Identity: The tool uses eNB_Part + Cell_Part as the primary key. This prevents PCI collisions and ensures multi-layer sites (e.g., L08, L18, L21) are treated as distinct logical cells.
Sample Weighting: A cell tested for 5 minutes with 300 samples will have a higher statistical weight on the aggregated RSRP/SINR than a cell tested for 10 seconds with 5 samples. This prevents "ping-pong" or momentary drive-by cells from skewing cluster averages.
Minimum Sample Threshold: To avoid statistical anomalies, cells with extremely low sample counts (e.g., < 5 samples) may be excluded from the "Worst Cell" classification but will remain in the "All Cells" baseline for visibility.
🚀 Future Enhancements (Roadmap)
Integration with OSS PM counters for capacity/traffic correlation.
Automated MapInfo/KML generation for Worst Cell plotting.
Support for 5G NR (NSA/SA) Drive Test exports.
Cluster-level summary dashboard with trend comparisons (Week-over-Week).
