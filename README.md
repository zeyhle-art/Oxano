# Oxano Portfolio Risk Monitor — Live Demo

## What this is
A working Streamlit dashboard for the portfolio early-warning monitoring
system. Toggle between three simulated market environments (Stable /
Realistic Baseline / Stressed-Crisis) live and watch the risk table,
feature-importance chart, cash-runway simulation, and detection track
record update in real time.

All data is synthetic, generated to mirror realistic PE/impact-investor
portfolio monitoring data. Structure is documented in the master dataset
workbook delivered alongside this app.

## How to run it locally

1. Install Python 3.10+ if you don't have it.
2. In this folder, install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py
4. It opens automatically in your browser at http://localhost:8501

## How to run it in the room (no laptop setup needed on their side)

Easiest free option: deploy to **Streamlit Community Cloud**
(share.streamlit.io) — connect this folder as a GitHub repo, click deploy,
and you get a public URL you can open on any device. Takes about 10 minutes
the first time, free tier is enough for a demo.

## Folder structure
    app.py              -> the dashboard
    requirements.txt    -> dependencies
    data/                -> the 12 CSVs the app reads (must stay in this
                            relative path, or edit DATA_DIR in app.py)

## If something looks off in the room
The false-positive rate and AUC shown per environment are real outputs of
the model trained live on that environment's data — not hardcoded. If you
re-run the generator scripts with a different random seed, these numbers
will shift slightly. That's expected and honest; say so if asked.
