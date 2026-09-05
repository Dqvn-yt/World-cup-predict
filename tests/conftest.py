import pytest

from app import create_app


@pytest.fixture(scope="session")
def app():
    return create_app(testing=True)


@pytest.fixture()
def client(app):
    return app.test_client()
