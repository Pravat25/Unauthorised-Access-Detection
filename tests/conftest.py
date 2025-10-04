# tests/conftest.py
import os
import sys
import tempfile
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

# Ensure project root (…/sam) is on sys.path so "import app" works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import User, app as flask_app, db  # noqa: E402


@pytest.fixture
def client():
    flask_app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
        SQLALCHEMY_DATABASE_URI="sqlite://",
        UPLOAD_FOLDER=os.path.join(tempfile.gettempdir(), "uploads"),
        REPORT_FOLDER=os.path.join(tempfile.gettempdir(), "reports"),
        MODEL_FOLDER=os.path.join(tempfile.gettempdir(), "models"),
        SECRET_KEY="test-secret",
    )

    # Make sure temp dirs exist
    for key in ("UPLOAD_FOLDER", "REPORT_FOLDER", "MODEL_FOLDER"):
        os.makedirs(flask_app.config[key], exist_ok=True)

    with flask_app.app_context():
        db.drop_all()
        db.create_all()

        # Ensure superadmin exists
        if not User.query.filter_by(email="admin@localhost").first():
            u = User(
                fullname="Super Administrator",
                email="admin@localhost",
                password_hash=generate_password_hash("admin123"),
                role="superadmin",
                approved=True,
            )
            db.session.add(u)
            db.session.commit()

    with flask_app.test_client() as c:
        yield c
