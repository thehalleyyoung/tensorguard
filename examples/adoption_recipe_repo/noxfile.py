import nox


@nox.session
def tensorguard(session):
    session.install("tensorguard")
    session.run("python", "-m", "pytest", "--tensorguard")
