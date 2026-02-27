import torch
import numpy as np

from xray_ai_backend.models.xray_model import get_model
from xray_ai_backend.utils.image_utils import preprocess_image
from xray_ai_backend.utils.gradcam import generate_gradcam_multi
from xray_ai_backend.services.report_service import generate_pdf_report


model = get_model()
DEVICE = next(model.parameters()).device

DISEASE_THRESHOLDS = {
    "Effusion": 0.60,
    "Lung Opacity": 0.60,
    "Pneumonia": 0.65,
    "Atelectasis": 0.60,
    "Edema": 0.65,
    "Consolidation": 0.60,
    "Cardiomegaly": 0.65,
    "Infiltration": 0.60,
    "Fracture": 0.70,
}

DEFAULT_THRESHOLD = 0.65
TOP_K = 3

_last_prediction = {"prediction": [], "confidence": []}


def get_last_prediction() -> dict:
    return _last_prediction


def predict_xray(image_path: str, filename: str) -> dict:
    global _last_prediction

    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        inference_tensor = image_tensor.clone().to(DEVICE)
        outputs = model(inference_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    labels = model.pathologies
    sorted_idx = np.argsort(probs)[::-1]

    predictions = []
    confidences = []

    for idx in sorted_idx[:TOP_K]:
        disease = labels[idx]
        prob = probs[idx]
        threshold = DISEASE_THRESHOLDS.get(disease, DEFAULT_THRESHOLD)

        if prob >= threshold:
            predictions.append(disease)
            confidences.append(round(float(prob), 4))

    if not predictions:
        max_prob = float(np.max(probs))
        if max_prob < 0.55:
            predictions = ["Normal"]
            confidences = [round(max_prob, 4)]
        else:
            top_idx = sorted_idx[0]
            predictions = [labels[top_idx]]
            confidences = [round(float(probs[top_idx]), 4)]

    _last_prediction = {"prediction": predictions, "confidence": confidences}

    heatmap_path = None
    disease_pairs = [
        (disease, labels.index(disease))
        for disease in predictions
        if disease != "Normal"
    ]

    if disease_pairs:
        print(f"[Inference] GradCAM for: {[d for d, _ in disease_pairs]}")

        heatmap_path = generate_gradcam_multi(
            model=model,
            image_tensor=image_tensor,
            image_path=image_path,
            filename=filename,
            diseases=disease_pairs,
        )

        if heatmap_path is None:
            print("[Inference] WARNING: GradCAM returned None")

    report_path = generate_pdf_report(
        filename, predictions, confidences, heatmap_path
    )

    return {
        "prediction": predictions,
        "confidence": confidences,
        "heatmap_path": heatmap_path,
        "report_path": report_path,
    }