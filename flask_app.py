
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('civil_unrest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('text', '')
    if not data:
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize the input data
    vectorized_data = vectorizer.transform([data])
    
    # Predict using the model
    prediction = model.predict(vectorized_data)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
