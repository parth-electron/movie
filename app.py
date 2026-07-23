"""
Flask web app for the movie recommendation system.

Routes:
    GET  /                 -> search / home page
    GET  /recommend        -> HTML results page (?title=Inception)
    GET  /api/recommend    -> JSON API (?title=Inception&n=10)
    GET  /api/search       -> JSON search suggestions (?q=incep)
"""

import os

from flask import Flask, render_template, request, jsonify

from recommender import recommender

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", titles=recommender.all_titles())


@app.route("/recommend")
def recommend_page():
    title = request.args.get("title", "").strip()

    if not title:
        return render_template(
            "index.html",
            titles=recommender.all_titles(),
            error="Please enter or select a movie title.",
        )

    results = recommender.recommend(title, top_n=10)

    if results is None:
        return render_template(
            "index.html",
            titles=recommender.all_titles(),
            error=f'"{title}" was not found in our catalog. Try selecting one from the list.',
        )

    source_movie = recommender.get_movie(recommender.lower_title_to_title[title.lower()])

    return render_template(
        "recommendations.html",
        source_movie=source_movie,
        results=results,
    )


@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    n = request.args.get("n", default=10, type=int)

    if not title:
        return jsonify({"error": "Missing required query param 'title'"}), 400

    results = recommender.recommend(title, top_n=n)

    if results is None:
        return jsonify({"error": f'Movie "{title}" not found'}), 404

    return jsonify({
        "query_title": title,
        "recommendations": results,
    })


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    matches = recommender.search(q, limit=10)
    return jsonify({"query": q, "matches": matches})


@app.route("/healthz")
def healthz():
    """Simple health check endpoint, useful for Render's health checks."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # Local development only. In production (Render), gunicorn runs the app
    # directly via the Procfile, so this block is not used.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
