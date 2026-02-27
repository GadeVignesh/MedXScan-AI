import os
import re
import time
from collections import deque

from groq import Groq
from xray_ai_backend.services.rag_retriever import retrieve_with_scores

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 350
TEMPERATURE  = 0.2
MIN_SCORE    = 0.25
MAX_HISTORY  = 6

if not GROQ_API_KEY:
    print("[RAG] WARNING: GROQ_API_KEY not found in environment.")
    print("[RAG] Make sure your .env file contains: GROQ_API_KEY=gsk_...")
    _groq_client = None
else:
    _groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"[RAG] Groq client initialized — model: {MODEL_NAME}")

_chat_history: deque = deque(maxlen=MAX_HISTORY)

INTENT_PATTERNS = {
    "definition":  r"\b(what is|define|definition|meaning of|explain what|tell me about)\b",
    "symptoms":    r"\b(symptoms?|signs?|feel|feeling|how does it feel|present with|what do i feel)\b",
    "causes":      r"\b(causes?|caused by|why|reason|risk factors?|etiology|who gets|how do you get)\b",
    "treatment":   r"\b(treat|treatment|cure|medication|medicine|manage|therapy|how to treat|drugs?|prescribed)\b",
    "xray":        r"\b(x.?ray|xray|scan|radiograph|imaging|finding|look like|appear on|show on|radiology)\b",
    "emergency":   r"\b(emergency|urgent|serious|dangerous|life.?threaten|fatal|critical|when to go|seek care|call doctor)\b",
    "prognosis":   r"\b(prognosis|outlook|survive|recovery|how long|life expectancy|chance of|will i)\b",
    "prevention":  r"\b(prevent|prevention|avoid|reduce risk|vaccine|protect|stop getting)\b",
    "diagnosis":   r"\b(diagnos|test|confirm|detect|how do doctors|how is it found|blood test|biopsy)\b",
    "comparison":  r"\b(difference|vs\.?|versus|compare|distinguish|similar to|same as|unlike)\b",
    "scan_result": r"\b(my scan|my result|my x.?ray|my report|what did|what does my|medxscan|detected in me|found in my|my diagnosis)\b",
    "normal":      r"\b(normal|healthy|clear|no disease|nothing found|all clear|no findings)\b",
}

SYSTEM_PROMPTS = {
    "definition": (
        "You are a medical education assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Give a clear, accurate definition in 2-3 sentences using simple language a patient can understand. "
        "Do not add any information not present in the context."
    ),
    "symptoms": (
        "You are a clinical medical assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Describe the key symptoms in 3-4 sentences. "
        "Clearly distinguish between common symptoms and serious warning signs. "
        "Do not add any information not present in the context."
    ),
    "causes": (
        "You are a medical education assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Explain the main causes and risk factors in 2-3 clear sentences. "
        "Do not add any information not present in the context."
    ),
    "treatment": (
        "You are a clinical medical assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Explain the standard treatment approach in 3-5 sentences. "
        "Include both medical treatments and relevant lifestyle changes. "
        "Always remind the patient to follow their doctor's specific advice. "
        "Never recommend specific medication doses. "
        "Do not add any information not present in the context."
    ),
    "xray": (
        "You are a radiology education assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Describe the typical X-ray and imaging findings in 2-3 clear sentences. "
        "Use language a non-radiologist patient can understand. "
        "Do not add any information not present in the context."
    ),
    "emergency": (
        "You are an urgent care medical assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Clearly and directly state which symptoms require IMMEDIATE emergency care. "
        "Be direct — this is safety-critical information. "
        "Always advise calling emergency services or going to the ER for these symptoms. "
        "Do not add any information not present in the context."
    ),
    "prognosis": (
        "You are a medical education assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Explain the typical prognosis and recovery outlook in 2-3 sentences. "
        "Be honest but compassionate. "
        "Remind the patient that outcomes vary and they should discuss with their doctor. "
        "Do not add any information not present in the context."
    ),
    "prevention": (
        "You are a preventive medicine assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Explain prevention strategies clearly in 2-3 sentences. "
        "Do not add any information not present in the context."
    ),
    "diagnosis": (
        "You are a medical education assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Explain how this condition is diagnosed — what tests and procedures doctors use. "
        "Keep the explanation accessible for a non-medical audience. "
        "Do not add any information not present in the context."
    ),
    "comparison": (
        "You are a medical education assistant for the MedXScan chest X-ray system. "
        "Answer using ONLY the provided medical context. "
        "Clearly explain the key differences between the conditions being compared. "
        "Do not add any information not present in the context."
    ),
    "scan_result": (
        "You are the MedXScan AI medical assistant. "
        "A patient has just received their AI chest X-ray analysis result. "
        "Answer using ONLY the provided medical context. "
        "Explain the detected finding(s) in clear, empathetic, and honest language. "
        "Cover: what the condition is, what it means, and recommended next steps. "
        "Always clearly state that MedXScan is an AI screening tool — NOT a replacement for a doctor. "
        "The patient must consult a qualified physician for proper diagnosis and treatment. "
        "Do not add any information not present in the context."
    ),
    "normal": (
        "You are the MedXScan AI medical assistant. "
        "Answer using ONLY the provided medical context. "
        "Explain what a normal or clear chest X-ray result means in simple, reassuring terms. "
        "Note that X-rays have limitations and cannot detect everything. "
        "Advise regular check-ups and consulting a doctor if any symptoms persist. "
        "Do not add any information not present in the context."
    ),
    "default": (
        "You are a professional medical assistant specializing in chest and lung conditions for MedXScan. "
        "Answer using ONLY the provided medical context. "
        "Give a clear, accurate answer in 3-4 sentences using language a patient can understand. "
        "If the context does not contain enough information, say so honestly. "
        "Do not add any information not present in the context."
    ),
}

DISCLAIMER = (
    "\n\n⚕️ *This information is for educational purposes only. "
    "Always consult a qualified healthcare professional for medical "
    "advice, diagnosis, or treatment.*"
)
DISCLAIMER_INTENTS = {"treatment", "emergency", "prognosis", "scan_result", "diagnosis"}

def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            return intent
    return "default"


def is_followup(question: str) -> bool:
    patterns = [
        r"\b(it|this|that|these|those|the condition|the disease|the finding)\b",
        r"\b(tell me more|elaborate|more about|what else|anything else)\b",
        r"\b(also|what about|and what|how about)\b",
        r"\b(you mentioned|you said|earlier)\b",
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def build_history_block(history: deque, max_turns: int = 2) -> str:
    if not history:
        return ""
    recent = list(history)[-max_turns:]
    lines  = ["[Recent Conversation]"]
    for turn in recent:
        lines.append(f"User: {turn['question']}")
        ans = turn["answer"][:200] + "..." if len(turn["answer"]) > 200 else turn["answer"]
        lines.append(f"Assistant: {ans}")
    return "\n".join(lines)


def clean_response(text: str) -> str:
    text      = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if sentences and not re.search(r"[.!?]$", sentences[-1].strip()):
        sentences = sentences[:-1]

    seen, unique = set(), []
    for s in sentences:
        key = s.strip().lower()
        if key not in seen and len(key) > 8:
            seen.add(key)
            unique.append(s.strip())

    return " ".join(unique).strip()


def _call_groq(system_prompt: str, user_message: str) -> str:
    response = _groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.1,
        stream=False
    )
    return response.choices[0].message.content.strip()


def generate_rag_answer(
    question: str,
    prediction_context: str = "",
    session_id: str = "default"
) -> str:
    """
    Main entry point — generates accurate medical answer using RAG + Llama3.

    Args:
        question           : User's question
        prediction_context : Detected disease(s) from X-ray e.g. "Atelectasis"
        session_id         : Reserved for future per-user session isolation
    Returns:
        str : Clean medical answer
    """
    question = question.strip()
    if not question:
        return "Please enter a question."

    if not _groq_client:
        return (
            "The AI model is not configured. "
            "Please make sure your .env file contains GROQ_API_KEY=gsk_... "
            "and restart the server."
        )

    intent = detect_intent(question)
    if prediction_context and intent in {"default", "definition", "symptoms"}:
        if re.search(
            r"\b(my scan|my result|detected|found|medxscan|report|x.?ray)\b",
            question.lower()
        ):
            intent = "scan_result"

    print(f"[RAG] Intent='{intent}' | Q='{question[:60]}'")

    retrieval_query = question
    if is_followup(question) and _chat_history:
        retrieval_query = f"{_chat_history[-1]['question']} {question}"

    scored_chunks = retrieve_with_scores(retrieval_query, top_k=3)

    if not scored_chunks:
        fallback = (
            "I don't have specific information about that in my knowledge base. "
            "For accurate medical guidance please consult a qualified healthcare "
            "professional or your doctor directly."
        )
        _chat_history.append({"question": question, "answer": fallback})
        return fallback

    top_score    = scored_chunks[0]["score"]
    context_text = "\n\n---\n\n".join(c["chunk"] for c in scored_chunks)

    user_parts = []

    history_block = build_history_block(_chat_history)
    if history_block:
        user_parts.append(history_block)

    if prediction_context:
        user_parts.append(
            f"[X-ray Analysis Result]\n"
            f"MedXScan AI detected: {prediction_context}"
        )

    user_parts.append(f"[Medical Knowledge Base]\n{context_text}")
    user_parts.append(f"[Patient Question]\n{question}")
    user_parts.append("Answer based only on the medical knowledge provided above.")

    user_message  = "\n\n".join(user_parts)
    system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["default"])

    start = time.time()
    try:
        raw_answer = _call_groq(system_prompt, user_message)
    except Exception as e:
        err = str(e).lower()
        print(f"[RAG] Groq error: {e}")
        if "invalid_api_key" in err or "authentication" in err:
            return "Invalid Groq API key. Please check your .env file."
        if "rate_limit" in err:
            return "The AI model is temporarily busy. Please wait a moment and try again."
        if "connection" in err or "timeout" in err:
            return "Cannot connect to the AI service. Please check your internet connection."
        return "An error occurred generating the response. Please try again."

    elapsed = round(time.time() - start, 2)
    answer  = clean_response(raw_answer)

    if len(answer.split()) < 10:
        print(f"[RAG] Short answer — using context fallback")
        sentences = re.split(r"(?<=[.!?])\s+", scored_chunks[0]["chunk"])
        answer    = " ".join(sentences[:3])

    if top_score < 0.35:
        answer += (
            " Please note that my knowledge base may not have complete information "
            "on this topic — consult a healthcare professional for accurate guidance."
        )

    if intent in DISCLAIMER_INTENTS:
        answer += DISCLAIMER

    _chat_history.append({"question": question, "answer": answer})

    print(f"[RAG] Done in {elapsed}s | {len(answer.split())} words | score={top_score:.3f}")
    return answer


def clear_history():
    _chat_history.clear()
    print("[RAG] History cleared")


def get_history() -> list:
    return list(_chat_history)


def get_model_status() -> dict:
    if not GROQ_API_KEY:
        return {
            "status":  "error",
            "message": "GROQ_API_KEY not set. Add it to your .env file."
        }
    if not _groq_client:
        return {
            "status":  "error",
            "message": "Groq client failed to initialize. Check your API key."
        }
    try:
        _groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        return {
            "status":  "ready",
            "model":   MODEL_NAME,
            "message": "Llama3-70B ready via Groq"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}