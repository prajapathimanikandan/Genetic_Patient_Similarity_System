from flask import Flask, request, jsonify
import pandas as pd
from src.analysis import run_analysis

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "genetic-data-compression-api"})

@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True)
    if not payload or "records" not in payload:
        return jsonify({"error": "Send JSON with a 'records' array of row objects."}), 400
    try:
        df = pd.DataFrame(payload["records"])
        target = payload.get("target_column")
        result = run_analysis(
            df,
            target_col=target,
            n_components=int(payload.get("n_components", 2)),
            variance_threshold=float(payload.get("variance_threshold", 0.95))
        )
        return jsonify({
            "original_feature_count": result["original_feature_count"],
            "pca_component_count": result["pca_component_count"],
            "variance_retained": result["variance_retained"],
            "recommended_components": result["recommended_components"],
            "explained_variance": result["variance_df"].to_dict(orient="records"),
            "pca_coordinates": result["pca_plot"].to_dict(orient="records"),
            "lda_coordinates": None if result["lda_plot"] is None else result["lda_plot"].to_dict(orient="records"),
            "lda_message": result["lda_message"]
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
