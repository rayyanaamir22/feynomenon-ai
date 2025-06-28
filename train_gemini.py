# train_gemini_tutor.py (or run directly in a Vertex AI Workbench Notebook)

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import TuningConfig
from google.cloud import storage
import os

# --- Configuration ---
PROJECT_ID = "your-gcp-project-id"  # Replace with your GCP project ID
REGION = "us-central1"               # Or another region where Gemini 1.5 Flash is available for tuning
BUCKET_NAME = "your-gcs-bucket-name" # Replace with your GCS bucket name (e.g., "my-feynman-tutor-data")
DATASET_FILE_NAME = "feynman_tutor_dataset.jsonl"
TUNED_MODEL_DISPLAY_NAME = "gemini-feynman-tutor"
BASE_MODEL = "gemini-1.5-flash-001" # Or "gemini-1.5-flash-002" if available and preferred

# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=REGION)

# --- Upload Dataset to Cloud Storage ---
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")

# Create a dummy dataset file for demonstration
# In a real scenario, you'd load your actual feynman_tutor_dataset.jsonl
dummy_dataset_content = """
{"messages": [{"role": "user", "content": "Explain gravity to me using the Feynman technique."}, {"role": "model", "content": "Imagine the Earth is like a big bowling ball on a stretchy trampoline. When you put the bowling ball there, it makes a dip. Now, if you roll a little marble (which is like the moon) near the bowling ball, it doesn't just go straight; it rolls towards the bowling ball because of the dip! That dip is what we call gravity – it's like a pull caused by massive things bending space-time.\n\nDid that trampoline analogy help, or would you like me to simplify how 'space-time' gets bent?"}]}
{"messages": [{"role": "user", "content": "Explain photosynthesis very simply. Then elaborate."}, {"role": "model", "content": "Photosynthesis is how plants make their food using sunlight. They take in water and a gas from the air, and sunlight helps them turn it into sugar (food) and oxygen. \n\nTo elaborate: This process happens in special parts of plant cells called chloroplasts, which contain chlorophyll, the green pigment. Chlorophyll absorbs sunlight. This light energy is then used to convert carbon dioxide (CO2) from the atmosphere and water (H2O) from the soil into glucose (C6H12O6), which is a sugar that serves as the plant's energy source, and oxygen (O2) as a byproduct. The chemical equation is: 6CO2 + 6H2O + Light Energy → C6H12O6 + 6O2. Are you tracking with the chemical equation, or would you like to review the inputs and outputs again?"}]}
{"messages": [{"role": "user", "content": "Tell me about quantum entanglement for a high school student, then for a university student."}, {"role": "model", "content": "For a high school student: Imagine you have two magic coins. When you flip one, it always lands on heads, and at the exact same moment, the other coin, no matter how far away it is, instantly lands on tails. They're connected in a spooky way, even if they're light-years apart. That's quantum entanglement – two particles are linked, and measuring one instantly affects the other.\n\nFor a university student: Quantum entanglement is a phenomenon where two or more particles become linked in such a way that the quantum state of each particle cannot be described independently of the others, even when the particles are separated by large distances. A measurement on one entangled particle instantaneously influences the state of the other(s), as described by the collapse of the wavefunction. This correlation is stronger than any classical correlation and has implications for quantum computing and cryptography. Do you want to discuss Bell's inequalities or the EPR paradox next?"}]}
"""
with open(DATASET_FILE_NAME, "w") as f:
    f.write(dummy_dataset_content.strip())

# Upload the dataset file to your GCS bucket
GCS_DATASET_URI = f"gs://{BUCKET_NAME}/{DATASET_FILE_NAME}"
upload_blob(BUCKET_NAME, DATASET_FILE_NAME, DATASET_FILE_NAME)

# --- Define Tuning Parameters (Optional, but recommended for control) ---
# These are just examples. You'll need to experiment with these.
# Vertex AI automatically handles hardware allocation for tuning.
tuning_config = TuningConfig(
    learning_rate_multiplier=1.0, # Default: 1.0. Higher means faster but potentially less stable.
    epochs=10,                    # Number of full passes over the training data. Start low and increase.
    batch_size=16                 # Number of examples processed in one training iteration.
)

# --- Start the Fine-tuning Job ---
print(f"Starting fine-tuning job for {BASE_MODEL} with dataset: {GCS_DATASET_URI}")
print(f"Tuned model will be named: {TUNED_MODEL_DISPLAY_NAME}")

tuned_model = GenerativeModel(BASE_MODEL).tune(
    training_data=GCS_DATASET_URI,
    tuning_config=tuning_config,
    tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
)

print(f"Fine-tuning job started. Model ID: {tuned_model.name}")
print(f"Check job status at: https://console.cloud.google.com/vertex-ai/models/tuned-models/{tuned_model.name.split('/')[-1]}?project={PROJECT_ID}&region={REGION}")

# The job runs asynchronously. You can monitor its status via the URL above.
# Once it's complete, you can use the tuned_model object to get details or deploy it.

# --- Example of how to use the tuned model for inference (after job completes) ---
# You'd typically do this in a separate script or after the tuning job finishes.

# # To load the tuned model after the job is complete:
# from vertexai.generative_models import GenerativeModel
# tuned_model_id = tuned_model.name # Or get this from the Vertex AI console

# print(f"\nOnce tuning is complete, you can load and use the tuned model:")
# try:
#     loaded_tuned_model = GenerativeModel(tuned_model_id)
#     print(f"Loaded tuned model: {loaded_tuned_model.name}")

#     # Example inference
#     prompt = "Explain quantum physics to a high schooler, then ask a clarifying question."
#     response = loaded_tuned_model.generate_content(prompt)
#     print("\n--- Tuned Model Response ---")
#     print(response.text)
# except Exception as e:
#     print(f"Could not load tuned model yet (likely still training or failed): {e}")