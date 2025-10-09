# Unauthorised Access Detection (Flask)

A Flask web app to detect/flag **unauthorised access attempts** using classic Machine Learning (scikit-learn).

> **Important:** We **do not** upload datasets to GitHub. See **Datasets** (links only).  
> This repo contains **code only**. Data, archives, reports, local DBs, coverage, and `node_modules/` are ignored via `.gitignore`.

---

## 1) Project Structure (key files)

- `app.py` — Flask entrypoint (UI + prediction/retrain routes)  
- `templates/`, `static/` — web UI  
- `models/` — trained model files (kept local or small artifacts only)  
- `security/`, `tests/` — auxiliary materials (optional)  
- `requirements.txt` — Python dependencies  
- `.env.example` — sample env; copy to `.env` and edit locally

> Local path we use: `.../Unauthorised Access Detection/app/sam/`

---

## 2) Quickstart — Run the Project (step-by-step)

### Prerequisites
- Python **3.10+**
- Git
- (Windows) PowerShell or Git Bash

### Setup & Run
```bash
# 1) Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Environment variables
# Copy sample env and edit values (e.g., SECRET_KEY)
cp .env.example .env

# 4) Run the app
python app.py
# App runs at http://127.0.0.1:8000

Training & Testing — Commands / Steps Used

We keep data files outside the repo. Replace any paths locally as needed.
Target column is typically label (or class). Models are saved under models/.

Option A — In-App (no CLI scripts)

Start the app: python app.py

Log in as Super Admin → open Retrain Model

Upload your training CSV (set target column, e.g., label)

Wait for training to complete; note the metrics (shown in UI/logs)

Use the Predict page to test with sample rows (CSV or form)

Datasets 


IoT Data-Driven Defense (Kaggle)
https://www.kaggle.com/datasets/charleswheelus/iotdatadrivendefense

Download datasets locally (e.g., C:\data\capstone\...) and keep them out of the repo.
For demos, use a tiny synthetic CSV with non-sensitive values.