from flask import Blueprint, request, jsonify
import os
from services.inference_service import predict_xray

xray_bp = Blueprint("xray", __name__)

@xray_bp.route("/predict-xray", methods=["POST"])
def predict_xray_route():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    result = predict_xray(file_path, file.filename)
    return jsonify(result)
