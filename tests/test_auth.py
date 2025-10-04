def login(client, email, pw):
    return client.post("/login", data={"email": email, "password": pw}, follow_redirects=True)


def test_login_requires_approval(client):
    # register an unapproved user
    client.post(
        "/register",
        data={"fullname": "U", "email": "u@ex.com", "password": "Aa@123", "role": "user"},
        follow_redirects=True,
    )
    rv = client.post("/login", data={"email": "u@ex.com", "password": "Aa@123"}, follow_redirects=True)
    assert b"awaiting superadmin approval" in rv.data


def test_superadmin_can_login(client):
    rv = login(client, "admin@localhost", "admin123")
    assert b"Logged in successfully" in rv.data
