import os
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI(title="EatWise API")
MOCK_MODE = os.getenv("EATWISE_MOCK", "1") == "1"

# Mock scores represent expected consumed enrichment, not choice probabilities.
MOCK_RECOMMENDATIONS = [
    {
        "recipe_id": "r_456",
        "title": "Chicken and veggie wrap",
        "score": 0.7368,
        "p_choose": 0.78,
        "enrichment": 0.86,
        "temptation": 0.70,
    },
    {
        "recipe_id": "r_457",
        "title": "Chickpea and roasted vegetable bowl",
        "score": 0.65,
        "p_choose": 0.70,
        "enrichment": 0.80,
        "temptation": 0.63,
    },
    {
        "recipe_id": "r_458",
        "title": "Lemon and lentil soup",
        "score": 0.531,
        "p_choose": 0.55,
        "enrichment": 0.72,
        "temptation": 0.54,
    },
]


class InitialRating(BaseModel):
    recipe_id: str
    rating: int = Field(ge=1, le=5)


class OnboardRequest(BaseModel):
    diet_goal: Literal["weight_loss", "muscle_gain", "low_sodium", "balanced"]
    cheat_meal_type: str
    initial_ratings: list[InitialRating] = Field(min_length=10, max_length=20)


class InteractionRequest(BaseModel):
    user_id: str
    session_id: str
    shown_recipe_ids: list[str] = Field(min_length=1)
    chosen_recipe_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cheat_meal_type: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    context: dict | None = None


def require_mock_mode() -> None:
    if not MOCK_MODE:
        raise HTTPException(status_code=503, detail="Live services are not connected yet.")


@app.post("/onboard")
def onboard(profile: OnboardRequest) -> dict:
    require_mock_mode()
    return {"user_id": uuid4().hex, "diet_goal": profile.diet_goal, "mock": True, "stored": False}


@app.get("/recommend")
def recommend(user_id: str = Query(min_length=1), n: int = Query(10, ge=1)) -> dict:
    require_mock_mode()
    return {"user_id": user_id, "mock": True, "recommendations": MOCK_RECOMMENDATIONS[:n]}


@app.post("/log_interaction")
def log_interaction(event: InteractionRequest) -> dict:
    require_mock_mode()
    return {"session_id": event.session_id, "mock": True, "stored": False}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mock": MOCK_MODE,
        "database": "not_connected",
        "model": "mock" if MOCK_MODE else "not_connected",
    }
