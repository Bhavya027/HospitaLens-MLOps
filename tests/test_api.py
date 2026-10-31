
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def valid_payload():
    return {
        "current_occupancy_beds": 200,
        "previous_occupancy_beds": 195,
        "two_weeks_ago_occupancy_beds": 190,
        "total_beds_7_day_avg": 250,
        "current_occupancy_rate": 0.8,
    }


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "features_required" in response.get_json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_valid_prediction(client):
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    result = response.get_json()
    assert result["model"] == "Random Forest"
    assert result["forecast_horizon"] == "7 days"
    assert result["predicted_occupancy_beds_next_week"] >= 0


def test_missing_feature(client):
    payload = valid_payload()
    payload.pop("current_occupancy_beds")

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "missing_fields" in response.get_json()


def test_invalid_capacity(client):
    payload = valid_payload()
    payload["total_beds_7_day_avg"] = 0

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_numeric_input(client):
    payload = valid_payload()
    payload["current_occupancy_beds"] = "two hundred"

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
