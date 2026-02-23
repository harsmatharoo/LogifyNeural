"""
LogifyNeural - app.py (Final Version)
Features:
1. Message history (last 5 predictions)
2. Live character/word counter
3. Confidence label
4. Top spam words highlight
5. Session stats counter
6. Logistic Regression Sigmoid Curve at /sigmoid
7. Current message dot on sigmoid curve (inline + full page)
"""

import os
import io
import re
import json
import math
import base64
import joblib
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # no GUI, runs in background
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template_string, redirect, url_for

MODEL_FILE    = "model.pkl"
FEEDBACK_FILE = "user_data.jsonl"

app = Flask(__name__)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found. Run: python train.py")

model = joblib.load(MODEL_FILE)

# ── in-memory state ──────────────────────────────────────────────
session_stats   = {"checked": 0, "spam": 0, "ham": 0}
message_history = []       # last 5 predictions
last_prob       = [None]   # mutable so /sigmoid route can read it


# ════════════════════════════════════════════════════════════════
#  HTML TEMPLATE
# ════════════════════════════════════════════════════════════════
HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LogifyNeural</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #080c14;
      --card: rgba(255,255,255,0.055);
      --card2: rgba(255,255,255,0.03);
      --text: rgba(255,255,255,0.93);
      --muted: rgba(255,255,255,0.55);
      --border: rgba(255,255,255,0.10);
      --shadow: 0 8px 32px rgba(0,0,0,0.4);
      --green: #22d3a5;
      --red: #f4536a;
      --yellow: #f5c542;
      --blue: #3b9eff;
      --btn: rgba(255,255,255,0.07);
      --btnHover: rgba(255,255,255,0.13);
    }
    * { box-sizing:border-box; margin:0; padding:0; }
    body {
      font-family:'DM Sans',sans-serif;
      background:
        radial-gradient(ellipse 900px 500px at 10% 0%,  rgba(34,211,165,0.07) 0%,transparent 60%),
        radial-gradient(ellipse 700px 500px at 90% 20%, rgba(59,158,255,0.08) 0%,transparent 55%),
        radial-gradient(ellipse 600px 400px at 50% 100%,rgba(244,83,106,0.06) 0%,transparent 55%),
        var(--bg);
      color:var(--text); min-height:100vh; padding:32px 24px;
    }
    .header {
      display:flex; align-items:center; justify-content:space-between;
      margin-bottom:28px; flex-wrap:wrap; gap:14px;
      max-width:1100px; margin-left:auto; margin-right:auto;
    }
    .logo { display:flex; align-items:center; gap:12px; }
    .logo-icon {
      width:42px; height:42px;
      background:linear-gradient(135deg,var(--green),var(--blue));
      border-radius:12px; display:flex; align-items:center;
      justify-content:center; font-size:22px;
    }
    .logo-text { font-family:'Space Mono',monospace; font-size:22px; font-weight:700; }
    .logo-text span { color:var(--green); }
    .stats-bar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .stat-pill {
      padding:6px 14px; border-radius:999px;
      border:1px solid var(--border); background:var(--card2);
      font-size:12.5px; color:var(--muted); font-family:'Space Mono',monospace;
    }
    .stat-pill b { color:var(--text); }
    .stat-spam b { color:var(--red); }
    .stat-ham  b { color:var(--green); }
    .sigmoid-btn {
      padding:6px 14px; border-radius:999px;
      border:1px solid rgba(59,158,255,0.35);
      background:rgba(59,158,255,0.10);
      font-size:12.5px; color:var(--blue);
      font-family:'Space Mono',monospace;
      text-decoration:none; transition:background 0.15s;
    }
    .sigmoid-btn:hover { background:rgba(59,158,255,0.18); }
    .grid { display:grid; grid-template-columns:1.2fr 0.8fr; gap:16px; max-width:1100px; margin:0 auto; }
    @media(max-width:860px){ .grid{ grid-template-columns:1fr; } }
    .card {
      background:var(--card); border:1px solid var(--border);
      border-radius:20px; padding:20px;
      backdrop-filter:blur(12px); box-shadow:var(--shadow);
    }
    .card-title {
      font-size:13px; font-family:'Space Mono',monospace;
      color:var(--muted); text-transform:uppercase;
      letter-spacing:1px; margin-bottom:14px;
    }
    textarea {
      width:100%; resize:vertical; min-height:130px;
      padding:14px; border-radius:14px; border:1px solid var(--border);
      background:rgba(0,0,0,0.3); color:var(--text); outline:none;
      font-size:14px; font-family:'DM Sans',sans-serif; line-height:1.5;
      transition:border-color 0.2s;
    }
    textarea:focus { border-color:rgba(34,211,165,0.4); }
    textarea::placeholder { color:rgba(255,255,255,0.3); }
    .counter {
      font-size:11.5px; color:var(--muted);
      font-family:'Space Mono',monospace;
      margin-top:6px; text-align:right;
    }
    .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
    .btn {
      border:1px solid var(--border); background:var(--btn); color:var(--text);
      padding:9px 14px; border-radius:10px; cursor:pointer;
      font-size:13px; font-family:'DM Sans',sans-serif;
      transition:background 0.15s,transform 0.05s;
    }
    .btn:hover { background:var(--btnHover); }
    .btn:active { transform:translateY(1px); }
    .btn-primary { background:rgba(34,211,165,0.15); border-color:rgba(34,211,165,0.35); font-weight:600; }
    .btn-primary:hover { background:rgba(34,211,165,0.22); }
    .btn-danger { background:rgba(244,83,106,0.13); border-color:rgba(244,83,106,0.35); }
    .btn-danger:hover { background:rgba(244,83,106,0.20); }
    .result-box { margin-top:16px; }
    .verdict {
      display:flex; align-items:center; gap:10px;
      padding:14px 16px; border-radius:14px;
      margin-bottom:12px; font-size:15px; font-weight:600;
    }
    .verdict-spam { background:rgba(244,83,106,0.12); border:1px solid rgba(244,83,106,0.35); color:var(--red); }
    .verdict-ham  { background:rgba(34,211,165,0.10);  border:1px solid rgba(34,211,165,0.30);  color:var(--green); }
    .verdict-icon { font-size:22px; }
    .confidence { font-size:12px; font-family:'Space Mono',monospace; opacity:0.8; font-weight:400; margin-left:4px; }
    .meta-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
    .badge {
      padding:5px 11px; border-radius:999px;
      border:1px solid var(--border); background:rgba(255,255,255,0.05);
      font-size:12px; color:var(--muted); font-family:'Space Mono',monospace;
    }
    .badge b { color:var(--text); }
    .bar-wrap { height:8px; border-radius:999px; background:rgba(255,255,255,0.07); overflow:hidden; margin-bottom:14px; }
    .bar { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--green),var(--yellow),var(--red)); transition:width 0.4s ease; }
    .spam-words {
      margin-top:10px; padding:10px 14px; border-radius:12px;
      background:rgba(244,83,106,0.07); border:1px solid rgba(244,83,106,0.2);
      font-size:12.5px; color:var(--muted);
    }
    .spam-words b { color:var(--text); display:block; margin-bottom:6px; }
    .spam-tag {
      display:inline-block; padding:3px 9px; border-radius:6px;
      background:rgba(244,83,106,0.18); border:1px solid rgba(244,83,106,0.3);
      color:var(--red); font-family:'Space Mono',monospace;
      font-size:11px; margin:3px 3px 0 0;
    }
    .sigmoid-preview { margin-top:14px; border-radius:14px; overflow:hidden; border:1px solid var(--border); }
    .sigmoid-preview img { width:100%; display:block; }
    .sigmoid-link-row {
      display:flex; align-items:center; justify-content:space-between;
      padding:8px 12px; background:rgba(59,158,255,0.06); border-top:1px solid var(--border);
      font-size:12px; color:var(--muted);
    }
    .sigmoid-link-row a { color:var(--blue); text-decoration:none; font-family:'Space Mono',monospace; font-size:11.5px; }
    .sigmoid-link-row a:hover { text-decoration:underline; }
    .feedback-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:10px; }
    .feedback-label { font-size:12px; color:var(--muted); font-family:'Space Mono',monospace; }
    .toast {
      margin-top:12px; padding:10px 14px; border-radius:12px;
      border:1px solid rgba(34,211,165,0.3); background:rgba(34,211,165,0.08);
      font-size:13px; color:var(--text);
    }
    .error-box {
      margin-top:12px; padding:10px 14px; border-radius:12px;
      border:1px solid rgba(244,83,106,0.3); background:rgba(244,83,106,0.08);
      font-size:13px; color:var(--text);
    }
    .history-table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }
    .history-table th {
      text-align:left; padding:6px 8px; color:var(--muted);
      font-family:'Space Mono',monospace; font-size:11px;
      text-transform:uppercase; border-bottom:1px solid var(--border); font-weight:400;
    }
    .history-table td {
      padding:8px; border-bottom:1px solid rgba(255,255,255,0.04);
      color:var(--muted); vertical-align:middle;
    }
    .history-table td:first-child { color:var(--text); max-width:160px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .tag-spam { padding:2px 8px; border-radius:999px; background:rgba(244,83,106,0.15); color:var(--red);   font-family:'Space Mono',monospace; font-size:10.5px; }
    .tag-ham  { padding:2px 8px; border-radius:999px; background:rgba(34,211,165,0.12); color:var(--green); font-family:'Space Mono',monospace; font-size:10.5px; }
    .no-history { color:var(--muted); font-size:12.5px; text-align:center; padding:20px 0; font-family:'Space Mono',monospace; }
    .info-list { list-style:none; padding:0; }
    .info-list li {
      padding:8px 0; border-bottom:1px solid var(--border);
      font-size:13px; color:var(--muted); display:flex; gap:10px; align-items:flex-start;
    }
    .info-list li:last-child { border-bottom:none; }
    .step {
      min-width:22px; height:22px;
      background:rgba(34,211,165,0.15); border:1px solid rgba(34,211,165,0.3);
      border-radius:6px; display:flex; align-items:center; justify-content:center;
      font-size:11px; color:var(--green); font-family:'Space Mono',monospace; font-weight:700;
    }
    code {
      background:rgba(255,255,255,0.07); padding:1px 6px; border-radius:5px;
      font-family:'Space Mono',monospace; font-size:11.5px; color:rgba(255,255,255,0.85);
    }
    .reason-box {
      margin-top:8px; padding:8px 12px; border-radius:10px;
      background:rgba(245,197,66,0.08); border:1px solid rgba(245,197,66,0.25);
      font-size:12.5px; color:rgba(245,197,66,0.9);
    }
  </style>
</head>
<body>

  <div class="header">
    <div class="logo">
      <div class="logo-icon">📡</div>
      <div>
        <div class="logo-text">Logify<span>Neural</span></div>
        <div style="font-size:12px;color:var(--muted);margin-top:2px;">AI-powered message classifier</div>
      </div>
    </div>
    <div class="stats-bar">
      <div class="stat-pill">Checked: <b>{{ stats.checked }}</b></div>
      <div class="stat-pill stat-spam">Spam: <b>{{ stats.spam }}</b></div>
      <div class="stat-pill stat-ham">Clean: <b>{{ stats.ham }}</b></div>
      <a class="sigmoid-btn" href="/sigmoid" target="_blank">📈 Sigmoid Curve</a>
    </div>
  </div>

  <div class="grid">

    <!-- LEFT -->
    <div>
      <div class="card">
        <div class="card-title">Analyze Message</div>
        <form method="post" action="/">
          <textarea name="text" id="msgInput"
            placeholder="Paste any message… e.g. WIN a FREE iPhone now! Click the link to claim your prize."
            oninput="updateCounter(this)">{{ text }}</textarea>
          <div class="counter" id="counter">0 characters · 0 words</div>
          <div class="row">
            <button class="btn btn-primary" type="submit"> Analyze</button>
            <button class="btn" type="button" onclick="fillExample(1)">Spam example</button>
            <button class="btn" type="button" onclick="fillExample(0)">Normal example</button>
            <button class="btn" type="button" onclick="clearBox()">Clear</button>
          </div>
        </form>

        {% if error %}<div class="error-box">{{ error }}</div>{% endif %}
        {% if saved %}<div class="toast"> Label saved to <code>user_data.jsonl</code>. Retrain with <code>python train.py</code>.</div>{% endif %}

        {% if result %}
        <div class="result-box">
          <div class="verdict {% if result.pred == 1 %}verdict-spam{% else %}verdict-ham{% endif %}">
            <span class="verdict-icon">{% if result.pred == 1 %}{% else %}{% endif %}</span>
            <span>
              {% if result.pred == 1 %}SPAM{% else %}NOT SPAM{% endif %}
              <span class="confidence">— {{ result.confidence }}</span>
            </span>
          </div>

          <div class="meta-row">
            <span class="badge">Probability: <b>{{ result.prob }}</b></span>
            <span class="badge">Threshold: <b>{{ result.threshold }}</b></span>
          </div>

          <div class="bar-wrap">
            <div class="bar" style="width:{{ result.prob_pct }}%;"></div>
          </div>

          {% if result.spam_words %}
          <div class="spam-words">
            <b> Top spam signals:</b>
            {% for w in result.spam_words %}<span class="spam-tag">{{ w }}</span>{% endfor %}
          </div>
          {% endif %}

          {% if result.reason %}<div class="reason-box"> {{ result.reason }}</div>{% endif %}

          <!-- Inline sigmoid with dot -->
          {% if result.sigmoid_img %}
          <div class="sigmoid-preview">
            <img src="data:image/png;base64,{{ result.sigmoid_img }}" alt="Sigmoid curve">
            <div class="sigmoid-link-row">
              <span> White dot = your message on the sigmoid curve</span>
              <a href="/sigmoid" target="_blank">Open full chart →</a>
            </div>
          </div>
          {% endif %}

          <div class="feedback-row">
            <span class="feedback-label">Was this correct?</span>
            <form method="post" action="/feedback" style="display:inline;">
              <input type="hidden" name="text" value="{{ text|e }}">
              <input type="hidden" name="label" value="1">
              <button class="btn btn-danger" type="submit">🚫 It's spam</button>
            </form>
            <form method="post" action="/feedback" style="display:inline;">
              <input type="hidden" name="text" value="{{ text|e }}">
              <input type="hidden" name="label" value="0">
              <button class="btn" type="submit"> Not spam</button>
            </form>
          </div>
        </div>
        {% endif %}
      </div>

      <!-- HISTORY -->
      <div class="card" style="margin-top:16px;">
        <div class="card-title">Recent Checks</div>
        {% if history %}
        <table class="history-table">
          <thead><tr><th>Message</th><th>Result</th><th>Probability</th><th>Time</th></tr></thead>
          <tbody>
            {% for item in history %}
            <tr>
              <td title="{{ item.text }}">{{ item.text[:45] }}{% if item.text|length > 45 %}…{% endif %}</td>
              <td>{% if item.pred == 1 %}<span class="tag-spam">SPAM</span>{% else %}<span class="tag-ham">CLEAN</span>{% endif %}</td>
              <td style="font-family:'Space Mono',monospace;font-size:11.5px;">{{ item.prob }}</td>
              <td style="font-size:11px;">{{ item.time }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% else %}
        <div class="no-history">No messages analyzed yet.</div>
        {% endif %}
      </div>
    </div>

    <!-- RIGHT -->
    <div class="card" style="align-self:start;">
      <div class="card-title">How It Works</div>
      <ul class="info-list">
        <li><span class="step">1</span><span>Train once: <code>python train.py</code> → downloads 5,000+ real SMS messages → saves <code>model.pkl</code>.</span></li>
        <li><span class="step">2</span><span>TF-IDF converts words to numbers. Words like "FREE", "WIN", "CLICK" score high.</span></li>
        <li><span class="step">3</span><span>Logistic Regression passes the score through the <b>sigmoid function</b> → probability 0 to 1. Above 0.5 = spam.</span></li>
        <li><span class="step">4</span><span>Label messages to improve the model. Labels save to <code>user_data.jsonl</code>.</span></li>
        <li><span class="step">5</span><span>JSON API: <code>POST /predict</code> with <code>{"text":"..."}</code></span></li>
        <li>
          <span class="step">→</span>
          <span><a href="/sigmoid" target="_blank" style="color:var(--blue);">View Sigmoid Curve</a> — see how raw scores become probabilities.</span>
        </li>
      </ul>

      <div style="margin-top:20px;">
        <div class="card-title">Confidence Levels</div>
        <ul class="info-list">
          <li><span class="step" style="background:rgba(244,83,106,0.2);border-color:rgba(244,83,106,0.4);color:var(--red);">!!</span><span style="color:var(--red);">Very likely spam</span> — prob &gt; 0.85</li>
          <li><span class="step" style="background:rgba(244,83,106,0.1);border-color:rgba(244,83,106,0.25);color:var(--red);">!</span><span>Probably spam</span> — prob 0.65–0.85</li>
          <li><span class="step" style="background:rgba(245,197,66,0.15);border-color:rgba(245,197,66,0.35);color:var(--yellow);">~</span><span>Borderline</span> — prob 0.50–0.65</li>
          <li><span class="step">✓</span><span style="color:var(--green);">Looks clean</span> — prob &lt; 0.50</li>
        </ul>
      </div>
    </div>

  </div>

<script>
  function updateCounter(ta) {
    const chars = ta.value.length;
    const words = ta.value.trim() === "" ? 0 : ta.value.trim().split(/\s+/).length;
    document.getElementById("counter").textContent =
      chars + " character" + (chars !== 1 ? "s" : "") +
      " · " + words + " word" + (words !== 1 ? "s" : "");
  }
  window.addEventListener("DOMContentLoaded", () => {
    const ta = document.getElementById("msgInput");
    if (ta) updateCounter(ta);
  });
  function fillExample(isSpam) {
    const ta = document.getElementById("msgInput");
    ta.value = isSpam
      ? "URGENT! You have WON a FREE prize worth $1000. Click the link NOW to claim before it expires!"
      : "Hey, are we still on for lunch at 1pm today?";
    updateCounter(ta); ta.focus();
  }
  function clearBox() {
    const ta = document.getElementById("msgInput");
    ta.value = ""; updateCounter(ta); ta.focus();
  }
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════
#  SIGMOID CHART GENERATOR
# ════════════════════════════════════════════════════════════════

def generate_sigmoid_chart(current_prob=None, inline=False):
    """
    Returns base64 PNG of the sigmoid curve.
    current_prob: plots a colored dot showing where the message lands.
    inline: smaller size for embedding inside the result card.
    """
    figsize = (5.5, 2.8) if inline else (7, 4)
    x = np.linspace(-8, 8, 400)
    y = 1 / (1 + np.exp(-x))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0d1120")

    # S curve
    ax.plot(x, y, color="#22d3a5", linewidth=2.5, zorder=3)

    # Decision boundary lines
    ax.axhline(y=0.5, color="#f5c542", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(x=0.0, color="#f5c542", linestyle="--", linewidth=1.2, alpha=0.7)

    # Shaded zones
    ax.fill_between(x, y, 0.5, where=(y > 0.5), color="#f4536a", alpha=0.10)
    ax.fill_between(x, y, 0.5, where=(y < 0.5), color="#22d3a5", alpha=0.10)

    # Zone labels
    ax.text( 4.5, 0.06, "HAM",  color="#22d3a5", fontsize=10, fontweight="bold", alpha=0.8)
    ax.text(-7.0, 0.90, "SPAM", color="#f4536a", fontsize=10, fontweight="bold", alpha=0.8)
    ax.text( 0.2, 0.52, "threshold = 0.5", color="#f5c542", fontsize=7.5, alpha=0.75)

    # Dot for current message
    if current_prob is not None:
        p = max(min(current_prob, 0.9999), 0.0001)
        log_odds  = math.log(p / (1 - p))
        dot_color = "#f4536a" if current_prob >= 0.5 else "#22d3a5"

        ax.scatter([log_odds], [current_prob],
                   color=dot_color, edgecolors="white",
                   linewidths=1.5, s=90, zorder=5)

        x_offset = 0.25 if log_odds < 5 else -3.8
        ax.annotate(
            f"  your message\n  p = {current_prob:.3f}",
            xy=(log_odds, current_prob),
            xytext=(log_odds + x_offset, current_prob + 0.10),
            color="white", fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
        )

    # Styling
    ax.set_xlabel("Log-odds score (raw model output)", color="#666", fontsize=8)
    ax.set_ylabel("Spam probability", color="#666", fontsize=8)
    ax.set_title("Logistic Regression — Sigmoid Curve", color="white",
                 fontsize=10 if inline else 12, pad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#222")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def looks_like_gibberish(text: str) -> bool:
    t = text.strip()
    if len(t) < 5:
        return False

    words = t.split()

    # Check if MOST words look like gibberish
    gibberish_words = 0
    for word in words:
        letters = re.sub(r"[^A-Za-z]", "", word)
        if not letters or len(letters) < 3:
            continue
        vowels = sum(ch.lower() in "aeiou" for ch in letters)
        vowel_ratio = vowels / len(letters)
        if vowel_ratio < 0.20:
            gibberish_words += 1

    # If more than half the words are gibberish → flag it
    return gibberish_words >= len(words) * 0.5


def save_feedback(text: str, label: int):
    record = {"text": text, "label": int(label), "ts": datetime.utcnow().isoformat()}
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_confidence_label(prob: float, pred: int) -> str:
    if pred == 1:
        if prob > 0.87:   return "Very likely spam"
        elif prob > 0.65: return "Probably spam"
        else:             return "Borderline"
    else:
        if prob < 0.15:   return "Definitely clean"
        elif prob < 0.35: return "Looks clean"
        else:             return "Borderline"


def get_top_spam_words(text: str, top_n: int = 6):
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]
        tfidf_mat  = vectorizer.transform([text])
        feat_names = np.array(vectorizer.get_feature_names_out())
        coefs      = classifier.coef_[0]
        nz         = tfidf_mat.nonzero()[1]
        if len(nz) == 0: return []
        scores = [(feat_names[i], float(tfidf_mat[0, i]) * coefs[i]) for i in nz]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [w for w, s in scores[:top_n] if s > 0]
    except Exception:
        return []


def predict_message(text: str, threshold: float = 0.50):
    if looks_like_gibberish(text):
        return 1, 0.99, "Looks like random keyboard-smash (gibberish rule).", []
    prob_spam  = float(model.predict_proba([text])[0][1])
    pred       = 1 if prob_spam >= threshold else 0
    spam_words = get_top_spam_words(text) if pred == 1 else []
    return pred, prob_spam, None, spam_words


# ════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def home():
    text   = ""
    result = None
    error  = None
    saved  = (request.args.get("saved") == "1")

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        if not text:
            error = "Paste a message first."
        else:
            threshold = 0.50
            pred, prob, reason, spam_words = predict_message(text, threshold)

            # update session stats
            session_stats["checked"] += 1
            session_stats["spam" if pred == 1 else "ham"] += 1

            # store for /sigmoid route
            last_prob[0] = prob

            # generate inline sigmoid with dot
            sigmoid_img = generate_sigmoid_chart(current_prob=prob, inline=True)

            result = {
                "pred":        pred,
                "prob":        f"{prob:.4f}",
                "prob_pct":    int(round(prob * 100)),
                "threshold":   f"{threshold:.2f}",
                "reason":      reason,
                "confidence":  get_confidence_label(prob, pred),
                "spam_words":  spam_words,
                "sigmoid_img": sigmoid_img,
            }

            # update history (newest first, max 5)
            message_history.insert(0, {
                "text": text,
                "pred": pred,
                "prob": f"{prob:.3f}",
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            if len(message_history) > 5:
                message_history.pop()

    return render_template_string(
        HTML,
        text=text, result=result, saved=saved,
        error=error, history=message_history, stats=session_stats,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    text      = (request.form.get("text") or "").strip()
    label_raw = (request.form.get("label") or "").strip()
    if not text or label_raw not in ("0", "1"):
        return redirect(url_for("home"))
    save_feedback(text, int(label_raw))
    return redirect(url_for("home", saved="1"))


@app.route("/sigmoid")
def sigmoid_page():
    """Full-page sigmoid — dot shows last analyzed message if available."""
    prob = last_prob[0]
    img  = generate_sigmoid_chart(current_prob=prob, inline=False)
    note = f"Showing position for last message &nbsp;(p = {prob:.4f})" \
           if prob is not None else "Analyze a message first to see your dot on the curve."
    return f"""
    <!doctype html><html>
    <head>
      <meta charset="utf-8"/>
      <title>LogifyNeural — Sigmoid</title>
      <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
      <style>
        body {{
          background:#080c14; color:rgba(255,255,255,0.85);
          font-family:'DM Sans',sans-serif;
          display:flex; flex-direction:column; align-items:center;
          justify-content:center; min-height:100vh; padding:40px 20px; gap:18px;
        }}
        .title {{ font-family:'Space Mono',monospace; font-size:18px; color:#22d3a5; }}
        .note  {{ font-size:13px; color:rgba(255,255,255,0.45); font-family:'Space Mono',monospace; }}
        img {{
          border-radius:18px; max-width:760px; width:100%;
          border:1px solid rgba(255,255,255,0.08);
          box-shadow:0 8px 40px rgba(0,0,0,0.5);
        }}
        .explain {{
          max-width:680px; text-align:center;
          font-size:13px; color:rgba(255,255,255,0.45); line-height:1.7;
        }}
        a {{ color:#3b9eff; font-size:13px; font-family:'Space Mono',monospace; text-decoration:none; }}
        a:hover {{ text-decoration:underline; }}
      </style>
    </head>
    <body>
      <div class="title">📈 Logistic Regression — Sigmoid Curve</div>
      <div class="note">{note}</div>
      <img src="data:image/png;base64,{img}" alt="Sigmoid">
      <div class="explain">
        The sigmoid maps a raw log-odds score to a probability between 0 and 1.
        <span style="color:#f5c542;">Yellow dashed lines</span> = 0.5 decision boundary.
        <span style="color:#f4536a;">Red zone</span> = spam.
        <span style="color:#22d3a5;">Green zone</span> = ham.
        The dot shows exactly where your last message landed.
      </div>
      <a href="/">← Back to LogifyNeural</a>
    </body>
    </html>
    """


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    pred, prob, reason, spam_words = predict_message(text, threshold=0.50)
    return jsonify({
        "label":            "SPAM" if pred == 1 else "NOT_SPAM",
        "spam_probability": prob,
        "confidence":       get_confidence_label(prob, pred),
        "spam_words":       spam_words,
        "reason":           reason,
    })


if __name__ == "__main__":

    app.run(debug=True)
