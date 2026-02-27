from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

from xray_ai_backend.routes.xray_routes import xray_bp
from xray_ai_backend.routes.chatbot_routes import chatbot_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(xray_bp)
app.register_blueprint(chatbot_bp)

@app.route("/")
def home():
    return jsonify({
        "status": "MedXScan AI Backend Running",
        "model": "TorchXRayVision DenseNet-121",
        "endpoints": ["/predict-xray", "/chat"]
    })

@app.route("/reports/<path:filename>")
def download_report(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(BASE_DIR, "reports")

    return send_from_directory(
        reports_dir,
        filename,
        as_attachment=True
    )

if __name__ == "__main__":
    app.run()