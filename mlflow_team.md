# 🚀 MLflow Team Tracking Server — Now Live!

Hey team! 👋

Our **MLflow Experiment Tracking Server** is up and running. From now on, all experiments, model runs, metrics, and artifacts should be logged here so we can track progress, compare models, and collaborate efficiently.

---

## 🔗 Server URL

**http://10.24.105.220:5001**

Open it in your browser — no login required, just click and explore.

---

## ⚡ Quick Setup (5 minutes)

### Step 1 — Install MLflow
```bash
pip install mlflow
```

### Step 2 — Add 2 lines at the top of your script
```python
import mlflow

mlflow.set_tracking_uri("http://10.24.105.220:5001")
mlflow.set_experiment("YourProjectName")   # use your own project name
```

That's it. You're connected. ✅

---

## 📊 Example 1 — Classic ML (Scikit-Learn)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- Connect to server ---
mlflow.set_tracking_uri("http://10.24.105.220:5001")
mlflow.set_experiment("Wine_Classification")

# --- Load data ---
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Run experiment ---
with mlflow.start_run(run_name="RandomForest_v1"):

    # Params
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset", "wine")

    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Metrics
    preds = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))

    # Save model
    mlflow.sklearn.log_model(model, "model", registered_model_name="WineClassifier")

print("✅ Run logged! Check: http://10.24.105.220:5001")
```

---

## 🤖 Example 2 — Gen AI / LLM Prompt Tracking (Gemini)

Track every LLM call: prompts, responses, latency, token usage, and scores.

```python
import mlflow
import time
import google.generativeai as genai

# --- Connect to server ---
mlflow.set_tracking_uri("http://10.24.105.220:5001")
mlflow.set_experiment("LLM_Prompt_Engineering")

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Define prompts to test ---
prompts = [
    "Summarize the benefits of MLflow in 3 bullet points.",
    "You are an expert data scientist. Summarize the benefits of MLflow in 3 bullet points.",
    "Explain MLflow to a non-technical manager in simple terms.",
]

for i, prompt in enumerate(prompts):
    with mlflow.start_run(run_name=f"prompt_variant_{i+1}"):

        # Log prompt details
        mlflow.log_param("model", "gemini-1.5-flash")
        mlflow.log_param("prompt_version", i + 1)
        mlflow.log_param("temperature", 0.2)
        mlflow.log_text(prompt, "prompt.txt")

        # Call LLM
        start = time.time()
        response = model.generate_content(prompt)
        latency = time.time() - start

        output = response.text

        # Log response and metrics
        mlflow.log_text(output, "response.txt")
        mlflow.log_metric("latency_seconds", round(latency, 3))
        mlflow.log_metric("response_length_chars", len(output))
        mlflow.log_metric("response_length_words", len(output.split()))

        # Optional: log your own quality score (1–5)
        quality_score = 4  # replace with your evaluation
        mlflow.log_metric("quality_score", quality_score)

        print(f"✅ Prompt {i+1} logged | Latency: {latency:.2f}s")

print("\n🔗 Compare results at: http://10.24.105.220:5001")
```

---

## 🧠 Example 3 — Gen AI with Auto MLflow Tracing (One-liner!)

MLflow 2.15+ supports **automatic tracing** for LLM calls — no manual logging needed:

```python
import mlflow
import google.generativeai as genai

mlflow.set_tracking_uri("http://10.24.105.220:5001")
mlflow.set_experiment("AutoTrace_LLM")

# Enable automatic tracing — just one line!
mlflow.gemini.autolog()

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# Every call is now automatically logged
response = model.generate_content("What is the best way to evaluate an LLM?")
print(response.text)

# Check traces at: http://10.24.105.220:5001  (go to "Traces" tab)
```

> Supports: `mlflow.openai.autolog()`, `mlflow.langchain.autolog()`, `mlflow.gemini.autolog()`

---

## 📁 Example 4 — Log Files & Artifacts (Plots, Reports, Datasets)

```python
import mlflow
import matplotlib.pyplot as plt
import pandas as pd

mlflow.set_tracking_uri("http://10.24.105.220:5001")
mlflow.set_experiment("Model_Reporting")

with mlflow.start_run(run_name="analysis_run"):

    # Log a plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.80, 0.88, 0.95], marker="o", label="Accuracy")
    ax.set_title("Training Progress")
    ax.legend()
    fig.savefig("training_curve.png")
    mlflow.log_artifact("training_curve.png")   # shows up in UI under Artifacts

    # Log a CSV report
    df = pd.DataFrame({"epoch": [1, 2, 3], "accuracy": [0.80, 0.88, 0.95]})
    df.to_csv("results.csv", index=False)
    mlflow.log_artifact("results.csv")

    # Log a text note
    mlflow.log_text("Best run so far — increased depth helped.", "notes.txt")

    mlflow.log_metric("final_accuracy", 0.95)

print("✅ Artifacts logged! Check: http://10.24.105.220:5001")
```

---

## 💡 Best Practices for the Team

| ✅ Do | ❌ Don't |
|-------|---------|
| Use descriptive `experiment` names per project | Use the default experiment |
| Use meaningful `run_name` values | Leave run names as auto-generated UUIDs |
| Log ALL hyperparameters, not just the good ones | Only log successful runs |
| Log prompts AND responses for LLM runs | Log only the final metric |
| Compare runs in the UI before promoting a model | Promote without comparison |

---

## 🔧 Quick Reference

```python
mlflow.set_tracking_uri("http://10.24.105.220:5001")  # connect
mlflow.set_experiment("MyProject")                     # set experiment

with mlflow.start_run(run_name="my_run"):
    mlflow.log_param("key", value)          # hyperparameters
    mlflow.log_metric("accuracy", 0.95)     # performance metrics
    mlflow.log_text("some text", "file.txt")# text artifacts
    mlflow.log_artifact("path/to/file")     # any file (plot, csv, etc.)
    mlflow.sklearn.log_model(model, "model")# save sklearn model
    mlflow.pyfunc.log_model(...)            # save any custom model
```

---

## 🙋 Need Help?

- 📖 **MLflow Docs:** https://mlflow.org/docs/latest/index.html
- 🔗 **Our Server:** http://10.24.105.220:5001
- 💬 **Ping the team lead** for access issues or questions

Happy experimenting! 🧪🚀
