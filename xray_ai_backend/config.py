import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

    SECRET_KEY = "medxscan-secret-key"
    DEBUG = True
