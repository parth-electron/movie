# Reel Match — Content-Based Movie Recommendation System

A Flask web app that recommends movies similar to one you already like, using **content-based filtering** (TF-IDF + cosine similarity over genres and plot text). Fully self-contained — no external dataset download, API key, or database required — and ready to deploy on [Render](https://render.com).

## How the recommendation engine works

1. Each movie's **genres** and **overview** (plot summary) are combined into one text blob (genres are repeated to weight them more heavily than the overview).
2. **TF-IDF vectorization** (`scikit-learn`'s `TfidfVectorizer`) turns every movie's text blob into a numeric vector, downweighting common words and upweighting distinctive ones.
3. **Cosine similarity** is computed between every pair of movies, producing a similarity matrix.
4. To recommend movies for a chosen title, the app looks up that movie's row in the similarity matrix and returns the highest-scoring other movies.

This approach needs no user rating history (unlike collaborative filtering) — it only needs the movies' own genre/plot metadata — so it works well as a cold-start-friendly demo.

## Project structure

```
movie_recommender/
├── app.py                     # Flask application (routes, HTML + JSON API)
├── recommender.py             # TF-IDF + cosine similarity recommendation engine
├── movies_data.py             # Embedded catalog of ~65 movies (title, genres, overview)
├── requirements.txt           # Python dependencies
├── Procfile                   # Tells Render/gunicorn how to start the app
├── runtime.txt                # Pins the Python version
├── render.yaml                # Optional one-click Render Blueprint config
├── templates/
│   ├── base.html               # Shared layout (marquee header/footer, fonts)
│   ├── index.html              # Search page + full catalog browser
│   └── recommendations.html    # Results page for a chosen movie
└── static/
    └── style.css                # Cinema/marquee-themed design system
```

## Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Search page with autocomplete + browsable catalog |
| `/recommend?title=<name>` | GET | HTML results page with the top 10 similar movies |
| `/api/recommend?title=<name>&n=<count>` | GET | JSON API returning recommendations |
| `/api/search?q=<query>` | GET | JSON search suggestions (substring match on titles) |
| `/healthz` | GET | Health check endpoint (used by Render) |

Example API call:
```
GET /api/recommend?title=Inception&n=5
```
```json
{
  "query_title": "Inception",
  "recommendations": [
    {"title": "Guardians of the Galaxy", "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"], "overview": "...", "similarity": 0.195},
    ...
  ]
}
```

## Running locally

```bash
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000` in your browser. (`app.py` reads the `PORT` environment variable if set, defaulting to 5000.)

## Deploying to Render

Render can deploy this either automatically from a `render.yaml` Blueprint, or manually through the dashboard. Both options are described below.

### Option A — One-click Blueprint (uses the included `render.yaml`)

1. Push this project to a GitHub (or GitLab) repository.
2. Go to the [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect your repository. Render will detect `render.yaml` automatically and pre-fill the service configuration (build command, start command, health check path).
4. Click **Apply** to create and deploy the service.

### Option B — Manual Web Service setup

1. Push this project to a GitHub (or GitLab) repository.
2. Go to the [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect your repository and select it.
4. Configure the service:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** `Free` (fine for this demo; upgrade for production traffic)
5. Click **Create Web Service**. Render will install dependencies, build, and deploy automatically.
6. Once deployed, Render gives you a public URL like `https://movie-recommender.onrender.com` — open it to use the app.

### Notes on the config files

- **`Procfile`** — tells Render (and any other Heroku-style platform) to run `gunicorn app:app`, binding to the `$PORT` environment variable Render provides at runtime. Flask's built-in dev server (`app.run(...)`) is only used for local development.
- **`runtime.txt`** — pins the Python version Render uses to build the environment, keeping builds reproducible.
- **`render.yaml`** — optional infrastructure-as-code file; lets you (re)create the exact same service configuration with one click, and is handy if you ever need to redeploy from scratch or share the setup with teammates.
- **`/healthz`** — a lightweight endpoint Render can poll to confirm the service is alive; already wired into `render.yaml`'s `healthCheckPath`.

### Free tier notes

Render's free web services spin down after a period of inactivity and take ~30–60 seconds to "wake up" on the next request. This is expected behavior — for an always-on service, upgrade to a paid instance type.

## Extending the project

- **Bigger catalog:** swap `movies_data.py` for a full dataset like [MovieLens](https://grouplens.org/datasets/movielens/) or [TMDB](https://www.themoviedb.org/documentation/api) — load it into a CSV/DataFrame and adjust `recommender.py`'s `MovieRecommender.__init__` accordingly.
- **Posters/images:** integrate the TMDB API to fetch poster images per title (requires a free API key).
- **Collaborative filtering:** if you add user ratings data, you could blend in a matrix-factorization-based recommender (e.g. `surprise` or `implicit`) alongside the content-based one for a hybrid system.
- **Persistence:** precompute and cache the TF-IDF/similarity matrix to disk (e.g. with `pickle` or `joblib`) if the catalog grows large enough that recomputing it on every app boot becomes slow.
