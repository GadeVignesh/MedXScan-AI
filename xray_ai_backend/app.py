from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify
from routes.xray_routes import xray_bp
from routes.chatbot_routes import chatbot_bp
from flask_cors import CORS
from flask import send_from_directory
import os

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
    reports_dir = os.path.join(os.getcwd(), "reports")

    return send_from_directory(
        reports_dir,
        filename,
        as_attachment=True 
    )



if __name__ == "__main__":
    app.run(debug=False)
