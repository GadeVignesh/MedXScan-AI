from flask import Blueprint, request, jsonify, Response, stream_with_context

from xray_ai_backend.services.rag_service import (
    generate_rag_answer,
    clear_history,
    get_history,
    get_model_status,
    detect_intent,
    build_history_block,
    is_followup,
    SYSTEM_PROMPTS,
    DISCLAIMER,
    DISCLAIMER_INTENTS,
    _chat_history,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    _groq_client
)

from xray_ai_backend.services.inference_service import get_last_prediction

chatbot_bp = Blueprint("chatbot", __name__)

MAX_QUESTION_LEN = 500
MIN_QUESTION_LEN = 3

def _get_prediction_context() -> str:
    try:
        last = get_last_prediction()
        if last and last.get("prediction"):
            diseases = [p for p in last["prediction"] if p != "Normal"]
            return ", ".join(diseases) if diseases else ""
    except Exception:
        pass
    return ""


def _validate_question(data):
    if not data:
        return None, (jsonify({"error": "Request body must be JSON"}), 400)

    question = data.get("question", "").strip()

    if not question:
        return None, (jsonify({"error": "No question provided"}), 400)

    if len(question) < MIN_QUESTION_LEN:
        return None, (jsonify({"error": "Question too short"}), 400)

    if len(question) > MAX_QUESTION_LEN:
        return None, (
            jsonify({"error": f"Question exceeds {MAX_QUESTION_LEN} character limit"}),
            400,
        )

    return question, None

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    question, err = _validate_question(data)
    if err:
        return err

    prediction_context = _get_prediction_context()

    try:
        answer = generate_rag_answer(
            question=question,
            prediction_context=prediction_context,
        )
    except Exception as e:
        print(f"[Chat] Error: {e}")
        return jsonify({"error": "Failed to generate response."}), 500

    response = {
        "answer": answer,
        "question": question,
    }

    if prediction_context:
        response["prediction_context"] = prediction_context

    return jsonify(response), 200

@chatbot_bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True)
    question, err = _validate_question(data)
    if err:
        return err

    prediction_context = _get_prediction_context()

    def generate_stream():
        try:
            import re
            from xray_ai_backend.services.rag_retriever import retrieve_with_scores

            if not _groq_client:
                yield "data: Groq API key not configured.\n\n"
                yield "data: [DONE]\n\n"
                return

            intent = detect_intent(question)

            if prediction_context and intent in {"default", "definition", "symptoms"}:
                if re.search(
                    r"\b(my scan|my result|detected|found|medxscan|report|x.?ray)\b",
                    question.lower(),
                ):
                    intent = "scan_result"

            retrieval_q = question

            if is_followup(question) and _chat_history:
                retrieval_q = f"{_chat_history[-1]['question']} {question}"

            scored_chunks = retrieve_with_scores(retrieval_q, top_k=3)

            if not scored_chunks:
                yield "data: I don't have specific information about that.\n\n"
                yield "data: [DONE]\n\n"
                return

            context_text = "\n\n---\n\n".join(c["chunk"] for c in scored_chunks)
            system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["default"])

            user_parts = []

            history_block = build_history_block(_chat_history)
            if history_block:
                user_parts.append(history_block)

            if prediction_context:
                user_parts.append(
                    f"[X-ray Analysis Result]\nMedXScan AI detected: {prediction_context}"
                )

            user_parts.append(f"[Medical Knowledge Base]\n{context_text}")
            user_parts.append(f"[Patient Question]\n{question}")
            user_parts.append(
                "Answer based only on the medical knowledge provided above."
            )

            user_message = "\n\n".join(user_parts)

            full_response = ""

            stream = _groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    safe_delta = delta.replace("\n", "\\n")
                    yield f"data: {safe_delta}\n\n"

            if intent in DISCLAIMER_INTENTS:
                safe_disclaimer = DISCLAIMER.replace("\n", "\\n")
                yield f"data: {safe_disclaimer}\n\n"

            _chat_history.append(
                {"question": question, "answer": full_response.strip()}
            )

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"[Stream] Error: {e}")
            yield "data: Error generating response.\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )

@chatbot_bp.route("/chat/clear", methods=["POST"])
def clear_chat():
    try:
        clear_history()
        return jsonify({"message": "Conversation history cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chatbot_bp.route("/chat/history", methods=["GET"])
def chat_history():
    try:
        history = get_history()
        return jsonify({"history": history, "count": len(history)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chatbot_bp.route("/chat/status", methods=["GET"])
def model_status():
    try:
        status = get_model_status()
        code = 200 if status["status"] == "ready" else 503
        return jsonify(status), code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503