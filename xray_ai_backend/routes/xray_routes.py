from flask import Blueprint, request, jsonify
import os
from xray_ai_backend.services.inference_service import predict_xray

xray_bp = Blueprint("xray", __name__)

@xray_bp.route("/predict-xray", methods=["POST"])
def predict_xray_route():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(BASE_DIR, "..", "uploads")
    upload_dir = os.path.abspath(upload_dir)

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    result = predict_xray(file_path, file.filename)

    return jsonify(result)