# tests/test_more_coverage.py
import os

from app import ModelRecord, User, app, db


def login_as_superadmin(client):
    return client.post("/login", data={"email": "admin@localhost", "password": "admin123"}, follow_redirects=True)


def test_home_redirects_when_logged_in(client):
    login_as_superadmin(client)
    rv = client.get("/", follow_redirects=False)
    assert rv.status_code in (302, 303)
    assert "/dashboard" in rv.headers.get("Location", "")


def test_invalid_login_message(client):
    rv = client.post("/login", data={"email": "no@ex.com", "password": "nope"}, follow_redirects=True)
    assert b"Invalid credentials." in rv.data


def test_register_duplicate_email(client):
    client.post(
        "/register",
        data={"fullname": "U1", "email": "dup@ex.com", "password": "Aa@123", "role": "user"},
        follow_redirects=True,
    )
    rv = client.post(
        "/register",
        data={"fullname": "U2", "email": "dup@ex.com", "password": "Aa@123", "role": "user"},
        follow_redirects=True,
    )
    assert b"Email already registered." in rv.data


def test_admin_models_requires_admin_or_superadmin(client):
    # Not logged in: should redirect to login
    rv = client.get("/admin/models", follow_redirects=False)
    assert rv.status_code in (302, 303)


def test_admin_models_ok_for_superadmin(client):
    login_as_superadmin(client)
    rv = client.get("/admin/models", follow_redirects=True)
    assert rv.status_code == 200


def test_admin_train_requires_admin_or_superadmin(client):
    # Not logged in: redirect
    rv = client.get("/admin/train", follow_redirects=False)
    assert rv.status_code in (302, 303)


def test_detect_requires_login(client):
    rv = client.get("/detect", follow_redirects=False)
    assert rv.status_code in (302, 303)


def test_reports_download_route(client, tmp_path):
    # create a dummy report file
    with app.app_context():
        fname = "report_dummy.csv"
        fpath = os.path.join(app.config["REPORT_FOLDER"], fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("a,b\n1,2\n")
    rv = client.get(f"/reports/{fname}", follow_redirects=False)
    assert rv.status_code == 200


def test_models_download_and_delete_model(client):
    with app.app_context():
        # Make a model record and a dummy pickle in the real models folder
        mname = "dummy-model"
        rec = ModelRecord(name=mname, accuracy=0.5)
        db.session.add(rec)
        db.session.commit()

        model_path = os.path.join(app.config["MODEL_FOLDER"], f"{mname}.pkl")
        with open(model_path, "wb") as f:
            f.write(b"\x80\x04N.")  # tiny pickle: None

    # Can download the raw file
    rv = client.get(f"/models/{mname}.pkl", follow_redirects=False)
    assert rv.status_code == 200

    # Deletion requires admin/superadmin -> log in as superadmin
    login_as_superadmin(client)
    rv = client.get(f"/admin/delete_model/{mname}", follow_redirects=True)
    assert b"deleted" in rv.data.lower()

    # Ensure DB row gone
    with app.app_context():
        assert ModelRecord.query.filter_by(name=mname).first() is None


def test_superadmin_approve_and_reject(client):
    # Register a user (unapproved)
    client.post(
        "/register",
        data={"fullname": "P", "email": "p@ex.com", "password": "Aa@123", "role": "user"},
        follow_redirects=True,
    )
    with app.app_context():
        u = User.query.filter_by(email="p@ex.com").first()
        uid = u.id

    # Superadmin approves
    login_as_superadmin(client)
    rv = client.get(f"/admin/approve/{uid}", follow_redirects=True)
    assert b"approved" in rv.data.lower()
    with app.app_context():
        # CHANGED: use SQLAlchemy 2.x style
        assert db.session.get(User, uid).approved is True

    # Create another and reject it
    client.get("/logout", follow_redirects=True)
    client.post(
        "/register",
        data={"fullname": "Q", "email": "q@ex.com", "password": "Aa@123", "role": "user"},
        follow_redirects=True,
    )
    with app.app_context():
        q = User.query.filter_by(email="q@ex.com").first()
        qid = q.id
    login_as_superadmin(client)
    rv = client.get(f"/admin/reject/{qid}", follow_redirects=True)
    assert b"removed" in rv.data.lower()
    with app.app_context():
        # CHANGED: use SQLAlchemy 2.x style
        assert db.session.get(User, qid) is None
