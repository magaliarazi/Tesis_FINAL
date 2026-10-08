"""
app.py
------
Servidor Flask para la app de predicción de atomtype y charge parcial.

Uso:
    pip install flask pandas numpy scikit-learn joblib rdkit
    python app.py

La app corre en http://localhost:5000
"""

import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from predictor import predict_molecule

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "mol2file" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    file = request.files["mol2file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400
    if not file.filename.endswith(".mol2"):
        return jsonify({"error": "El archivo debe tener extensión .mol2"}), 400

    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(suffix=".mol2", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = predict_molecule(tmp_path, original_name=file.filename)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
