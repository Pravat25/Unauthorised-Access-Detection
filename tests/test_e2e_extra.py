# tests/test_e2e_extra.py
import io
import os

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from werkzeug.security import generate_password_hash

from app import User, app, db


def ensure_superadmin():
    """Make sure superadmin exists for tests that log in directly."""
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email="admin@localhost").first():
            su = User(
                fullname="Super Administrator",
                email="admin@localhost",
                password_hash=generate_password_hash("admin123"),
                role="superadmin",
                approved=True,
            )
            db.session.add(su)
            db.session.commit()


def login_superadmin(client):
    ensure_superadmin()
    return client.post(
        "/login",
        data={"email": "admin@localhost", "password": "admin123"},
        follow_redirects=True,
    )


def make_dummy_model(filename_base: str, columns=("f1", "f2"), constant=0):
    """
    Create a tiny pickled model with meta['columns'] so /detect can load it
    and align features. Uses DummyClassifier for speed and determinism.
    """
    with app.app_context():
        model_dir = app.config["MODEL_FOLDER"]
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{filename_base}.pkl")

        X = pd.DataFrame({columns[0]: [0, 1], columns[1]: [1, 0]})
        y = [0, 1]
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(X, y)
        joblib.dump({"model": clf, "meta": {"columns": list(X.columns)}}, path)
        return path


def test_home_redirects_when_logged_in(client):
    login_superadmin(client)
    rv = client.get("/", follow_redirects=False)
    # Should redirect straight to dashboard for authenticated users
    assert rv.status_code in (301, 302)
    assert "/dashboard" in rv.headers["Location"]


def test_admin_models_page_renders(client):
    login_superadmin(client)
    rv = client.get("/admin/models")
    assert rv.status_code == 200


def test_detect_unlabeled_flow(client):
    # prepare model
    make_dummy_model("dummy-unlabeled", columns=("f1", "f2"), constant=0)

    login_superadmin(client)

    # CSV WITHOUT label column -> unlabeled inference branch
    csv = "f1,f2\n0,1\n1,0\n"
    data = {
        "model_name": "dummy-unlabeled",
        "testfile": (io.BytesIO(csv.encode("utf-8")), "unlabeled.csv"),
    }
    rv = client.post("/detect", data=data, content_type="multipart/form-data", follow_redirects=True)
    # Should flash "Predictions generated" and render results_unlabeled
    assert b"Predictions generated" in rv.data
    assert rv.status_code == 200


def test_detect_labeled_flow(client):
    # prepare model that always predicts 0
    make_dummy_model("dummy-labeled", columns=("a", "b"), constant=0)

    login_superadmin(client)

    # CSV WITH label column -> supervised metrics branch
    # labels are all 0 to match DummyClassifier constant prediction
    csv = "a,b,label\n0,0,0\n1,1,0\n"
    data = {
        "model_name": "dummy-labeled",
        "testfile": (io.BytesIO(csv.encode("utf-8")), "labeled.csv"),
    }
    rv = client.post("/detect", data=data, content_type="multipart/form-data", follow_redirects=True)
    # Page should include an "Accuracy" readout / report
    assert b"Accuracy" in rv.data
    assert rv.status_code == 200


def test_download_report_route(client, tmp_path):
    # create a small report file and download it
    with app.app_context():
        report_dir = app.config["REPORT_FOLDER"]
        os.makedirs(report_dir, exist_ok=True)
        name = "test_report.csv"
        path = os.path.join(report_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("x,y\n1,2\n")

    rv = client.get("/reports/test_report.csv", follow_redirects=False)
    assert rv.status_code == 200
    assert b"x,y" in rv.data


def test_delete_model_not_found_branch(client):
    login_superadmin(client)
    # Hitting delete for a model record that doesn't exist -> "not found" branch
    rv = client.get("/admin/delete_model/nope-no-model", follow_redirects=True)
    assert rv.status_code == 200
    assert b"not found" in rv.data.lower()


def test_train_page_get_and_logout(client):
    # GET train page as superadmin (allowed)
    login_superadmin(client)
    rv = client.get("/admin/train")
    assert rv.status_code == 200

    # Also cover logout
    rv2 = client.get("/logout", follow_redirects=False)
    assert rv2.status_code in (301, 302)
