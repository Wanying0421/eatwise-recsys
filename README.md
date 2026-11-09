# EatWise

**A temptation-aware healthy-eating recommender.**
EatWise recommends home-cooked recipes that are healthier than what you would otherwise eat, *and* tempting enough that you will actually cook them instead of ordering a pizza.

It implements the model from:

> M. S. Anwar, P. S. Dhillon, G. Schoenebeck. **Recommendation and Temptation.** RecSys ’25, 2025. [doi:10.1145/3705328.3748063](https://doi.org/10.1145/3705328.3748063)

Instead of optimizing clicks or ratings, EatWise optimizes **the nutrition the user actually eats**:

- each recipe has an **enrichment** score (health value, anchored to its nutrition) and a **temptation** score (instant appeal, learned from behaviour);
- the user always has an **outside option**: their usual cheat meal (takeaway, snacks…);
- recipes are ranked by the **expected enrichment consumed** (Theorem 1 of the paper):

```
E[u_j(i | o)] = Φ((C_j(i) − μ) / σ) · u_j(i) + (1 − Φ((C_j(i) − μ) / σ)) · E[u_j(o)]
C_j(i)        = λC_j · u_j(i) + (1 − λC_j) · v_j(i)
```

---

## Team

| Role | Member |
|---|---|
| Project Lead (evaluation, repo, schedule) | Wanying XIONG |
| Data Engineer (data pipeline, database, `/log_interaction`) | Alban VILLAIN |
| Lead ML Engineer (model, Algorithm 1, greedy rule) | Hicham RKAIBI |
| ML Engineer (training loop, click simulation, `predict()`) | Nouha SANTIR |
| Systems Engineer (FastAPI, Streamlit, deployment) | Laroche NGWA NGOH |

---

## Project status

| Week | Milestone | Status |
|---|---|---|
| 1 | Project plan, repo skeleton, interfaces agreed |  Done |
| 2 | Data pipeline on full dataset · model forward pass · eval harness on mock predictions · API returns mock recommendations |  In progress |
| 3 | Trained model · beats random & most-popular · ablation · live end-to-end system |  To do |
| 4 | Showcase: demo, poster, pitch |  To do |

> Some commands below refer to modules that are still being written. They describe the agreed structure; this README is updated as each part lands.

---

## Repository structure

```
eatwise-recsys/
├── README.md
├── requirements.txt
├── docs/                     # project plan, status reports, figures
├── data/                     # created locally, not committed
│   ├── raw/                  # Food.com files from Kaggle
│   └── processed/            # cleaned recipes, ratings, consumption rounds
├── models/                   # trained artifacts, not committed
└── src/
    ├── data/                 # [DE]  cleaning, health score, SQLite database
    │   ├── preprocess.py
    │   └── db.py
    ├── model/                # [MLE] enrichment/temptation model, Algorithm 1, greedy rule
    │   ├── model.py
    │   ├── simulate_clicks.py
    │   ├── train.py
    │   └── predict.py
    ├── eval/                 # [PL]  splits, NDCG@10, Recall@20, baselines, ablation, simulation
    │   ├── run_eval.py
    │   └── run_simulation.py
    └── app/                  # [SE]  FastAPI backend + Streamlit frontend
        ├── api.py
        └── streamlit_app.py
```

---

## Run the system locally

Requirements: **Python 3.11** and Git.

**1. Clone the repository**

```bash
git clone https://github.com/Wanying0421/eatwise-recsys.git
cd eatwise-recsys
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. (Optional) Mock mode** — run the API without dataset or trained model:

```bash
export EATWISE_MOCK=1              # Windows PowerShell: $env:EATWISE_MOCK=1
```

**5. Start the backend** (terminal 1)

```bash
uvicorn src.app.api:app --reload --port 8000
```

Check http://localhost:8000/health and the API docs at http://localhost:8000/docs.

**6. Start the frontend** (terminal 2, same virtual environment)

```bash
streamlit run src/app/streamlit_app.py
```

Open http://localhost:8501.

To let another machine on the same Wi-Fi use the demo, run uvicorn with `--host 0.0.0.0` and open `http://<your-local-ip>:8501` from the other machine.

---

## Data

V1 dataset: [Food.com – Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) (Kaggle).

1. Download the dataset and put `recipes.csv` and `reviews.csv` (or the `.parquet` versions) in `data/raw/`.
2. Build the processed data:

```bash
python -m src.data.preprocess         # clean recipes, compute health score, keep users with ≥ 25 ratings
python -m src.model.simulate_clicks   # build consumption rounds with 15-item shown sets
```

The health score (enrichment anchor) is a simplified Nutri-Score in [0, 1]: points for fiber and protein, penalties for calories, saturated fat, sugar and sodium, min-max scaled.

---

## (Optional) Re-train the model

```bash
python -m src.model.train --dim 32 --lr 0.001 --batch-size 256 --epochs 50 --seed 0
```

Adam optimizer with early stopping on validation loss; we report results over seeds 0–4. Artifacts are written to `models/`; the API loads them at start-up when mock mode is off.

---

## Evaluation

```bash
python -m src.eval.run_eval         # NDCG@10, Recall@20 vs random and most-popular
python -m src.eval.run_simulation   # overall individual enrichment, baselines and ablation (5 seeds)
```

- **Split:** temporal 70 / 15 / 15 (train / validation / test). Test data is never used for training, embeddings or tuning.
- **Ranking metrics:** NDCG@10 and Recall@20 on test-period ratings ≥ 4.
- **Paper metric:** overall individual enrichment over 50 simulated rounds after a 25-round random warm-up.
- **Baselines:** random, most-popular, enrichment-only, temptation-only, ratings-based, click-based.
- **Ablation:** temptation switched off (λC = 1) and outside option ignored.

---

## Interfaces (agreed in Week 1)

**Model**

```python
predict(user_id: str,
        candidate_recipe_ids: list[str],
        context: dict | None = None,
        top_n: int = 10) -> list[dict]
# each item: {"recipe_id", "score", "p_choose", "enrichment", "temptation"}, sorted by score
```

**API**

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/onboard` | Create a user: diet goal, usual cheat meal, 10–20 initial recipe ratings |
| `GET` | `/recommend?user_id=...&n=10` | Ranked recipes (mock results in mock mode) |
| `POST` | `/log_interaction` | Log shown recipes, the choice (a recipe or `outside_option`) and an optional rating |
| `GET` | `/health` | API, database and model status, including mock mode |

Example `/recommend` response:

```json
{
  "user_id": "u_123",
  "mock": false,
  "recommendations": [
    {"recipe_id": "r_456", "title": "Chicken and veggie wrap", "score": 0.71,
     "p_choose": 0.64, "enrichment": 0.78, "temptation": 0.59}
  ]
}
```

**Events table**

```
events(user_id, session_id, ts, shown_recipe_ids, chosen_recipe_id | 'outside_option',
       cheat_meal_type, rating, context)
```

---

## Contributing

- Work on a branch, open a pull request, and get one review before merging into `main`.
- Run `pytest` before pushing.
- Never commit raw data or large model files (`data/`, `models/` are in `.gitignore`).
