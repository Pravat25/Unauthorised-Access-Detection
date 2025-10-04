def test_admin_users_requires_superadmin(client):
    # normal user approved
    from app import User, db, generate_password_hash

    with client.application.app_context():
        u = User(
            fullname="N",
            email="n@ex.com",
            password_hash=generate_password_hash("Aa@123"),
            role="user",
            approved=True,
        )
        db.session.add(u)
        db.session.commit()
    client.post("/login", data={"email": "n@ex.com", "password": "Aa@123"}, follow_redirects=True)
    rv = client.get("/admin/users", follow_redirects=True)
    assert b"Superadmin access required" in rv.data
