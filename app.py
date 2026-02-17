
#pip -q install flask flask-cors pyngrok faiss-cpu sentence-transformers openai

#cd /content/drive/MyDrive/projects/chu_chat_bot

#   pip install ngrok

import os
os.environ["ngrok"] ="key"
os.environ["OPENAI_API_KEY"] = "ur key "

# !pip -q install flask flask-cors pyngrok faiss-cpu sentence-transformers openai

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from pathlib import Path
import numpy as np
import faiss
import json
import os
import re
import time
import logging

# NGROK (google colab)
USE_NGROK = True
if USE_NGROK:
    try:
        from pyngrok import ngrok
    except ImportError:
        os.system("pip -q install pyngrok")
        from pyngrok import ngrok

# OpenAI
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o-mini"
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except Exception:
    _openai_client = None
    USE_OPENAI = False

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr-rag-api")

NGROK_TOKEN = os.environ.get("ngrok", None)
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_RETRIEVE = 10
TOP_K_CONTEXT = 4
SIM_THRESHOLD = 0.50
MAX_CHUNKS_PER_SOURCE = 2

INDEX_DIR = Path("data/index")
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH  = INDEX_DIR / "chunks_metadata.json"

PERSONAL_PATTERNS = [
    r"\bmon solde\b", r"\bmes cong[ée]s\b", r"\bmes cong[ée]s restants\b",
    r"\bmon salaire\b", r"\bma paie\b", r"\bmon contrat\b", r"\bmon dossier\b",
    r"\bmon planning\b", r"\bmes heures\b", r"\bmes bulletins\b",
    r"\bpeux[- ]?tu vérifier\b", r"\bpeux[- ]?tu consulter\b"
]

def is_personal_request(q: str) -> bool:
    ql = q.lower()
    return any(re.search(p, ql) for p in PERSONAL_PATTERNS)

SYSTEM_RULES = """Tu es un assistant RH interne.
Règles:
- Réponds UNIQUEMENT à partir des extraits de contexte fournis.
- Si le contexte ne contient pas la réponse, dis que tu n'as pas assez d'informations.
- N'invente pas de règles, dates, chiffres ou procédures.
- N'accède pas aux données personnelles (soldes, paie, dossier, contrat, planning).
- Réponse claire, en français, sous forme de points quand utile.
Ne crée pas de section "Sources:" dans ta réponse.
"""

if not INDEX_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError(
        f"Missing artifacts:\n- {INDEX_PATH}\n- {META_PATH}\n"
        "Check your paths / generated artifacts."
    )

logger.info("Loading FAISS index: %s", INDEX_PATH)
index = faiss.read_index(str(INDEX_PATH))

logger.info("Loading metadata: %s", META_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

logger.info("Loading embedder: %s", EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

logger.info("Loaded: index=%d vectors | chunks=%d", index.ntotal, len(chunks))

def retrieve(query: str, k: int = TOP_K_RETRIEVE):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    scores, idxs = index.search(q_emb, k)

    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        out.append({"score": float(score), "chunk": chunks[idx]})
    return out

def select_context(results):
    filtered = [r for r in results if r["score"] >= SIM_THRESHOLD]
    if not filtered:
        return [], 0.0

    selected = []
    per_source = {}

    for r in sorted(filtered, key=lambda x: x["score"], reverse=True):
        src = r["chunk"]["source"]
        per_source[src] = per_source.get(src, 0)
        if per_source[src] >= MAX_CHUNKS_PER_SOURCE:
            continue
        selected.append(r)
        per_source[src] += 1
        if len(selected) >= TOP_K_CONTEXT:
            break

    confidence = selected[0]["score"] if selected else 0.0
    return selected, confidence

def build_context_block(selected):
    lines = []
    for r in selected:
        c = r["chunk"]
        page = c.get("page_start", c.get("page", None))
        text = c["text"].strip().replace("\n", " ")
        lines.append(f"- Source: {c['source']} (page {page})\n  Extrait: {text}")
    return "\n".join(lines)

def build_prompt(question: str, context_block: str) -> str:
    return f"""{SYSTEM_RULES}

Extraits de contexte:
{context_block}

Question: {question}
Réponse:
"""

def call_openai(prompt: str) -> str:
    if not _openai_client:
        return "OpenAI client not configured (set OPENAI_API_KEY)."
    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Tu es un assistant RH prudent et factuel."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assistant RH</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #fafafa; color: #1a1a1a; min-height: 100vh;
            display: flex; align-items: center; justify-content: center; padding: 2rem;
        }
        .container { width: 100%; max-width: 800px; }
        .header { text-align: center; margin-bottom: 3rem; }
        h1 { font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; }
        .subtitle { color: #666; font-size: 0.95rem; }
        .chat-container { background: white; border-radius: 16px; box-shadow: 0 2px 20px rgba(0, 0, 0, 0.08); overflow: hidden; }
        #chat-messages { height: 500px; overflow-y: auto; padding: 2rem; }
        .empty-state { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #999; text-align: center; padding: 2rem; }
        .empty-state-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.3; }
        .message { margin-bottom: 1.5rem; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message-user { display: flex; justify-content: flex-end; }
        .message-assistant { display: flex; justify-content: flex-start; }
        .message-error { display: flex; justify-content: flex-start; } /* FIX */
        .message-content { max-width: 75%; padding: 1rem 1.25rem; border-radius: 18px; line-height: 1.5; }
        .message-user .message-content { background: #1a1a1a; color: white; }
        .message-assistant .message-content { background: #f5f5f5; color: #1a1a1a; }
        .message-error .message-content { background: #fff0f0; color: #d32f2f; border: 1px solid #ffcdd2; }
        .sources { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0, 0, 0, 0.06); font-size: 0.85rem; }
        .sources-title { color: #666; margin-bottom: 0.5rem; font-weight: 500; }
        .source-item { color: #888; padding: 0.5rem 0; line-height: 1.4; }
        .input-area { padding: 1.5rem; border-top: 1px solid #f0f0f0; background: #fafafa; }
        .input-group { display: flex; gap: 0.75rem; align-items: center; }
        input[type="text"] {
            flex: 1; padding: 1rem 1.25rem; border: 1px solid #e0e0e0; background: white; color: #1a1a1a;
            border-radius: 12px; font-size: 0.95rem; transition: all 0.2s;
        }
        input[type="text"]:focus { outline: none; border-color: #1a1a1a; }
        input[type="text"]::placeholder { color: #999; }
        button {
            padding: 1rem 1.75rem; background: #1a1a1a; color: white; border: none; border-radius: 12px;
            font-size: 0.95rem; font-weight: 500; cursor: pointer; transition: all 0.2s; white-space: nowrap;
        }
        button:hover { background: #333; }
        button:active { transform: scale(0.98); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .loading {
            display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 640px) {
            body { padding: 1rem; }
            h1 { font-size: 1.5rem; }
            #chat-messages { height: 400px; padding: 1.5rem; }
            .message-content { max-width: 85%; }
            .input-group { flex-direction: column; }
            button { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Assistant RH</h1>
            <p class="subtitle">Posez vos questions sur les politiques RH</p>
        </div>

        <div class="chat-container">
            <div id="chat-messages">
                <div class="empty-state">
                    <div class="empty-state-icon">💬</div>
                    <p>Comment puis-je vous aider aujourd'hui ?</p>
                </div>
            </div>

            <div class="input-area">
                <div class="input-group">
                    <input type="text" id="question-input" placeholder="Posez votre question..." autocomplete="off" />
                    <button id="send-btn" type="button">Envoyer</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chat-messages');
        const questionInput = document.getElementById('question-input');
        const sendBtn = document.getElementById('send-btn');

        function addMessage(role, content, sources = [], confidence = null, refused = false) {
            const emptyState = chatMessages.querySelector('.empty-state');
            if (emptyState) emptyState.remove();

            const messageDiv = document.createElement('div');

            // FIX: normalize role -> correct class name
            const roleClass = (role === 'user') ? 'user' : (role === 'error' ? 'error' : 'assistant');
            messageDiv.className = `message message-${roleClass}`;

            let html = `<div class="message-content">`;

            if (roleClass === 'assistant' && !refused) {
                html += (content || '').replace(/\\n/g, '<br>');

                if (sources && sources.length > 0) {
                    html += '<div class="sources"><div class="sources-title">Sources</div>';
                    sources.forEach(src => {
                        html += `<div class="source-item">${src.source}, p.${src.page}</div>`;
                    });
                    html += '</div>';
                }
            } else {
                html += (content || '');
            }

            html += '</div>';
            messageDiv.innerHTML = html;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;

            addMessage('user', question);
            questionInput.value = '';
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="loading"></span>';

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                // FIX: handle non-JSON responses cleanly
                const text = await res.text();
                let data = null;
                try { data = JSON.parse(text); } catch(e) {}

                if (!res.ok) {
                    addMessage('error', data?.error || `HTTP ${res.status}: ${text.slice(0, 200)}`);
                } else if (!data) {
                    addMessage('error', 'Invalid JSON response from server.');
                } else if (data.error) {
                    addMessage('error', data.error);
                } else {
                    addMessage('assistant', data.answer, data.sources, data.confidence, data.refused);
                }
            } catch (err) {
                addMessage('error', `Erreur: ${err.message}`);
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Envoyer';
                questionInput.focus();
            }
        }

        // FIX: make sure listeners always fire
        sendBtn.addEventListener('click', () => sendQuestion());

        // FIX: keydown is more reliable than keypress
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendQuestion();
            }
        });

        questionInput.focus();
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "index_size": int(index.ntotal),
        "chunks": len(chunks),
        "embed_model": EMBED_MODEL_NAME,
        "openai_enabled": bool(_openai_client and USE_OPENAI),
        "llm_model": OPENAI_MODEL if (_openai_client and USE_OPENAI) else None
    })

@app.route("/api/ask", methods=["POST"])
def ask_api():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    if is_personal_request(question):
        return jsonify({
            "answer": "Je ne peux pas accéder à vos informations personnelles (solde, paie, dossier, planning). "
                      "Veuillez consulter le SIRH / l’intranet RH ou contacter votre service RH.",
            "sources": [],
            "confidence": 0.0,
            "refused": True
        })

    t0 = time.time()
    results = retrieve(question, TOP_K_RETRIEVE)
    selected, confidence = select_context(results)

    if not selected:
        return jsonify({
            "answer": "Je n’ai pas assez d’informations dans les documents disponibles pour répondre de façon fiable. "
                      "Pouvez-vous préciser votre question (type de congé, statut, contexte) ?",
            "sources": [],
            "confidence": float(confidence),
            "refused": True
        })

    context_block = build_context_block(selected)
    prompt = build_prompt(question, context_block)

    if _openai_client and USE_OPENAI:
        answer = call_openai(prompt)
    else:
        answer = "LLM désactivé. Extraits retrouvés:\n\n" + context_block

    sources = []
    for r in selected:
        c = r["chunk"]
        sources.append({
            "source": c["source"],
            "page": c.get("page_start", c.get("page", None)),
            "score": r["score"],
        })

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "confidence": float(confidence),
        "refused": False,
        "latency_ms": elapsed_ms
    })

if __name__ == "__main__":
    if USE_NGROK:
        if NGROK_TOKEN:
            ngrok.set_auth_token(NGROK_TOKEN)
        try:
            ngrok.kill()
        except Exception:
            pass
        tunnel = ngrok.connect(5000, "http")
        print("Public URL:", tunnel.public_url)

    app.run(host="0.0.0.0", port=5000, debug=False)