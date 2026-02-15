# LogifyNeural

![Python](https://img.shields.io/badge/Python-3.8+-3572A5?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-web-e8603c?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f89939?style=flat&logo=scikit-learn&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-charts-11557c?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-130654?style=flat&logo=pandas&logoColor=white)
![Accuracy](https://img.shields.io/badge/accuracy-96%25-2ecc71?style=flat)
![License](https://img.shields.io/badge/license-MIT-aaaaaa?style=flat)

A machine learning web app that detects whether a message is spam or not. Paste any message, get an instant prediction with a confidence score, and see exactly where it lands on the model's decision curve.

Built with Python, Flask, and scikit-learn. No cloud APIs, no subscriptions — everything runs on your own machine.

---

## Why the name LogifyNeural?

The name has two hidden layers to it — just like the project itself.

**"Logi"** is a direct reference to **Logistic Regression** — the machine learning algorithm powering every prediction under the hood. Not a neural network, not deep learning, just classic logistic regression done properly. The name is honest about what's inside.

**"-fy"** means *to make* or *to turn into* — like Spotify turns music into a service, LogifyNeural turns your message into a spam verdict.

**"Neural"** represents the intelligent, modern feel of the app — the live charts, confidence scores, feedback loop, and explainability features that make it behave smarter than a basic spam filter.

Put it together: **LogifyNeural** = *turning messages into verdicts using logistic regression, intelligently.*

Most people will think it's a neural network. It's not. That's the joke.

---

## What it does

You paste a message. It tells you if it's spam.

But it also shows you *why* — which words triggered the spam flag, how confident the model is, and a live graph showing exactly where your message sits on the logistic regression curve. It's not just a yes/no answer, it actually explains its reasoning.

---

## How it looks

- Dark themed web interface, runs in your browser at `localhost:5000`
- After each prediction you see the spam probability, a confidence label, the top words that pushed it toward spam, and a mini sigmoid curve with a dot showing your message's position
- A table at the bottom keeps track of the last 5 messages you checked
- A stats bar at the top shows how many messages you've checked this session and how many were spam vs clean

---

## The tech behind it

The model uses two things working together:

**TF-IDF** converts your message into numbers. It figures out which words appear a lot in spam messages and gives them high scores. Words like "FREE", "WIN", "URGENT", "CLICK" score very high. "hey see you at 3pm" scores very low.

**Logistic Regression** takes those numbers and produces a probability between 0 and 1. That probability passes through a sigmoid function (the S-shaped curve you see in the chart) and if it comes out above 0.5, it's spam.

The model was trained on 5,574 real SMS messages from a public dataset — a mix of genuine texts and actual spam messages.

---

## What is Logistic Regression?

Despite the name having "regression" in it, it's actually a **classification** algorithm — it answers yes/no questions like "is this spam or not".

Here's the idea in plain English. The model looks at your message and computes a single score based on the words in it. Spam words push the score up. Normal words push it down. That raw score then gets converted into a probability using the **sigmoid function**.

The logistic regression equation is:

$$
P(\text{spam}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n)}}
$$

Breaking that down in human words:

```
x1, x2, x3 ... xn   →  the TF-IDF score of each word in your message
β1, β2, β3 ... βn   →  the weights the model learned during training
                         (how spammy each word is)
β0                  →  the baseline score before any words are considered
e                   →  Euler's number (2.718), just a math constant
```

So in practice it looks like this:

```
score = (weight of "free"  × how often "free" appears)
      + (weight of "win"   × how often "win" appears)
      + (weight of "click" × how often "click" appears)
      + ... every other word ...
      + baseline

probability = 1 / (1 + e^(-score))
```

If `probability > 0.5` → **SPAM**.
If `probability < 0.5` → **NOT SPAM**.

The model learned the weights by reading 5,574 labeled messages during training. The sigmoid at the end just squashes whatever number comes out into a clean 0-to-1 range.

---

## What even is a sigmoid function?

The sigmoid takes any number and squashes it between 0 and 1. That's its entire job.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$


<img width="696" height="500" alt="image" src="https://github.com/user-attachments/assets/b91584e7-e66e-4548-808c-ace2035acd08" />


**Reading the diagram:**

- The green S-curve is the sigmoid function
- Left side (negative scores) = ham territory — the curve is near 0
- Right side (positive scores) = spam territory — the curve is near 1
- Crossing decision boundary would lead to spam

**Scores to probabilities quick reference:**

| Raw score | Probability | What it means |
|---|---|---|
| -8 | 0.0003 | Basically 0% spam |
| -4 | 0.018  | Very likely clean |
| 0  | 0.5    | 50/50, right on the line |
| +4 | 0.982  | Very likely spam |
| +8 | 0.9997 | Basically 100% spam |

---

## What even is "training" a model?

Training is just the model reading a lot of labeled examples and figuring out patterns.

Imagine you're trying to learn what spam looks like. Someone shows you 5,000 messages and tells you which are spam and which aren't. After enough examples you start noticing patterns — spam says "FREE" a lot, it's urgent, it has random prize claims. Normal messages say "see you at 3" or "can you pick up milk".

That's exactly what training does. After seeing enough examples it learns things like:

```
"free"    →  strong spam signal  (weight = +3.2)
"win"     →  strong spam signal  (weight = +2.8)
"urgent"  →  strong spam signal  (weight = +2.5)
"hey"     →  ham signal          (weight = -1.4)
"lunch"   →  ham signal          (weight = -2.1)
```

Those weights get saved into `model.pkl`. That file IS the trained model.

**When you run train.py:**
```
1. Downloads 5,574 labeled SMS messages from the internet
2. Splits them — 80% for learning, 20% held back for testing
3. Figures out word weights using logistic regression
4. Tests itself on the 20% it hasn't seen before
5. Reports accuracy  (ours = 96%)
6. Saves everything to model.pkl
```

**When you run app.py after that:**
```
1. Loads model.pkl
2. You paste a message
3. Scores every word using learned weights
4. Adds scores up → one final number
5. Sigmoid converts it → probability between 0 and 1
6. Above 0.5 = spam. Done.
```

The model never thinks. It never understands language. It just does math on word frequencies. The magic is that this simple math, done on enough real examples, turns out to be surprisingly good at catching spam.

---

## How the code works

1. `train.py` downloads 5,574 labeled SMS messages, converts every word into numbers using TF-IDF, trains a Logistic Regression model on those numbers, and saves the result to `model.pkl`.

2. `app.py` loads that saved model and starts a Flask web server. When you paste a message and click Analyze, it scores every word using the weights the model learned, adds them up, passes the total through the sigmoid function to get a probability between 0 and 1, and if that number is above 0.5 it calls it spam.

3. Every prediction also generates a live sigmoid chart showing exactly where your message landed on the curve, pulls out the top words that triggered the spam flag, and logs it to the session history.

4. When you click the feedback buttons, your label gets saved to `user_data.jsonl`. Next time you run `train.py` those examples are included and the model gets a little smarter.

---

## Getting started

You need Python installed. Then install the dependencies:

```
pip install flask scikit-learn joblib pandas requests matplotlib numpy
```

**Step 1 — Train the model** (only need to do this once):
```
python train.py
```
This downloads the dataset, trains the model, and saves it as `model.pkl`. Takes about 10–20 seconds.

**Step 2 — Run the app:**
```
python app.py
```

**Step 3 — Open your browser:**
```
http://127.0.0.1:5000
```

That's it. You're running LogifyNeural.

---

## Files in this project

```
📁 logifyneural/
   ├── train.py              → downloads data, trains model, saves model.pkl
   ├── app.py                → the web app
   ├── model.pkl             → saved trained model (created by train.py)
   ├── user_data.jsonl       → your feedback labels (created automatically)
   ├── sigmoid_diagram.png   → diagram used in this README
   ├── requirements.txt      → all dependencies in one file
   └── .gitignore            → tells git to ignore model.pkl and cache files
```

---

## Teach it to be smarter

Every time you click "It's spam" or "Not spam" after a prediction, your label gets saved to `user_data.jsonl`. When you retrain, those examples get included and the model gets a little bit better.

To retrain:
```
python train.py
```
Then restart `app.py`.

---

## The sigmoid chart

Visit `http://127.0.0.1:5000/sigmoid` after making a prediction. You'll see the full S-curve with a colored dot showing exactly where your last message landed. Green dot = clean, red dot = spam. The closer the dot is to the edges, the more confident the model is.

---

## JSON API

```
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{"text": "Congratulations! You have won a free prize!"}
```

Response:
```json
{
  "label": "SPAM",
  "spam_probability": 0.9731,
  "confidence": "Very likely spam",
  "spam_words": ["congratulations", "free", "won", "prize"],
  "reason": null
}
```

---

## Confidence levels

| Label | Probability | What it means |
|---|---|---|
| Very likely spam | > 0.85 | Model is very sure |
| Probably spam | 0.65 – 0.85 | Strong spam signals |
| Borderline | 0.50 – 0.65 | Could go either way |
| Looks clean | 0.35 – 0.50 | Probably normal |
| Definitely clean | < 0.15 | Model is very sure it's fine |

---

## Known limitations

- Trained on SMS messages so works best on short texts
- Can miss sneaky spam that avoids obvious trigger words
- Gibberish like "rfvwgsedfsw efwdwqefd" is caught by a vowel-ratio rule, not the ML model
- Session stats reset every time you restart the server
- 96% accurate means it will occasionally get things wrong

---

## What I learned building this

- How TF-IDF converts text into numbers a model can actually use
- How logistic regression learns word weights from labeled examples
- What the sigmoid function actually does and why it's used
- How to embed matplotlib charts directly into Flask without saving image files
- How to build a feedback loop where user labels improve the model over time

---

## Requirements

```
pip install flask scikit-learn joblib pandas requests matplotlib numpy
```
