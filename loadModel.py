import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import torch.nn.functional as F

# Load the model
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('./results')

# Load the LabelEncoder
label_encoder = joblib.load('./results/label_encoder.joblib')

# New data to classify
new_responses = [
    "Ensure the work area is clean and free of obstacles",
    "Check for any falling objects that might be hazardous",
    "Check for other crews in area",
    "Adapt to changing conditions daily",
    "Safety first",
    "Always hazards best case",
    "I am nonsensical",
    "Never have I ever",
    "Dookie two shoes",
    "Possible spark zones and distance from heat",
    "Fire",
    "Material storage",
    "Truck unsecured loads access",
    "Tie off places for anchoring",
    "always look up",
    "Looks for hazardous crew mates",
    "Prevent sharp objects from exposure",
    "Changes",
    # Add more responses here...
]

# Tokenize the new data
new_encodings = tokenizer(new_responses, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Put the model in evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    outputs = model(**new_encodings)
    logits = outputs.logits

# Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=1)

# Get the predicted class indices
predicted_class_indices = torch.argmax(probabilities, dim=1).tolist()

# Get the confidence levels
confidence_levels = probabilities.max(dim=1).values

# Decode the class indices to original labels
predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

# Print the predictions with confidence levels
for response, label, confidence in zip(new_responses, predicted_labels, confidence_levels):
    print(f"Response: {response}")
    print(f"Predicted Category: {label}")
    print(f"Confidence Level: {confidence.item() * 100:.2f}%")