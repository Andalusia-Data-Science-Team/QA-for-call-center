import mlflow
import google.generativeai as genai


TRACKING_URI = "http://10.24.105.220:5001"
EXPERIMENT_NAME = "QA for call center"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable automatic tracing for the Gemini calls
mlflow.gemini.autolog()

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# Every call is automatically logged to the experiment above
response = model.generate_content("What is the best way to evaluate a call center QA workflow?")
print(response.text)

print(f"🔗 Check traces at: {TRACKING_URI} (look under the '{EXPERIMENT_NAME}' experiment)")
