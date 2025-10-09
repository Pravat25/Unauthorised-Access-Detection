cat > README.md << 'EOF'
# Unauthorised Access Detection (Flask)

A Flask web app to detect/flag unauthorised access attempts using classic ML (scikit-learn).

> **Important:** We **do not** upload datasets to GitHub. See **Datasets** for links only.

---

## 1) Project Structure (key files)

- `app.py` — Flask entrypoint (UI + prediction/retrain routes)
- `templates/`, `static/` — web UI
- `models/` — trained model files (large/binary ignored by Git)
- `security/`, `tests/` — auxiliary materials
- `requirements.txt` — Python dependencies
- `.env.example` — copy to `.env` and fill secrets/config

> Datasets, archives, reports, local DBs, coverage, and `node_modules/` are ignored via `.gitignore`.

---

## 2) Quickstart — Run the App

### Prerequisites
- Python **3.10+**, Git

### Setup & Run
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate

pip install -r requirements.txt
cp .env.example .env   # edit values if needed

python app.py
# http://127.0.0.1:8000
