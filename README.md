# 🩺 MedXScan AI
**Explainable Chest X-Ray Disease Detection with Medical RAG Chatbot**

MedXScan AI is a full-stack medical AI system that detects thoracic diseases from chest X-rays using a NIH-trained DenseNet-121 model, visualizes predictions with Grad-CAM heatmaps, generates PDF diagnostic reports, and answers medical questions through a RAG-powered chatbot.

---

## 🚀 Features

- 🧠 Multi-label thoracic disease detection
- 🔥 Grad-CAM heatmap visualization with disease labels
- 📄 Automated PDF diagnostic reports
- 🤖 RAG medical chatbot powered by Llama-3.3-70B
- ⚡ Production-ready Flask REST API
- 🌐 React + TypeScript frontend

---

## 🛠️ Tech Stack

**Backend:** Python, Flask, PyTorch, TorchXRayVision, OpenCV, FAISS, Groq SDK  
**Frontend:** React, TypeScript, Tailwind CSS, Framer Motion

---

## ▶️ Setup

### Backend
```bash
cd xray_ai_backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

```bash
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict-xray` | Analyze chest X-ray |
| `GET` | `/reports/<filename>` | Download PDF report |
| `POST` | `/chat` | Ask medical question |
| `GET` | `/chat/status` | Check chatbot status |
