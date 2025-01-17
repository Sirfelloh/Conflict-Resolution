# File: app.py
from flask import Flask, render_template, request, jsonify
import pickle
import folium
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)


# Load the model and vectorizer
with open('civil_unrest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Sample data for graphs and map
conflict_data = {
    'regions': ['Nairobi', 'Mombasa', 'Kisumu', 'Eldoret'],
    'levels': [80, 45, 60, 50],
    'coords': [
        {'lat': -1.286389, 'lon': 36.817223, 'conflict_level': 80},  # Nairobi
        {'lat': -4.043477, 'lon': 39.668206, 'conflict_level': 45},  # Mombasa
        {'lat': -0.091702, 'lon': 34.767956, 'conflict_level': 60},  # Kisumu
        {'lat': 0.514277, 'lon': 35.269780, 'conflict_level': 50},   # Eldoret
    ]
}

@app.route('/')
def index():
    # Create a heatmap of Kenya
    kenya_map = folium.Map(location=[0.0236, 37.9062], zoom_start=6)
    for point in conflict_data['coords']:
        folium.CircleMarker(
            location=[point['lat'], point['lon']],
            radius=point['conflict_level'] / 10,
            color='red',
            fill=True,
            fill_color='red',
        ).add_to(kenya_map)

    # Save map as HTML
    kenya_map.save('templates/map.html')

    # Create graphs
    bar_chart = go.Figure([go.Bar(x=conflict_data['regions'], y=conflict_data['levels'], marker_color='crimson')])
    line_chart = go.Figure([go.Scatter(x=conflict_data['regions'], y=conflict_data['levels'], mode='lines+markers')])

    # Serialize graphs using JSON
    bar_chart_json = json.dumps(bar_chart, cls=PlotlyJSONEncoder)
    line_chart_json = json.dumps(line_chart, cls=PlotlyJSONEncoder)

    # Pass serialized JSON objects to the template
    return render_template('index.html', bar_chart=bar_chart_json, line_chart=line_chart_json)


@app.route('/predict', methods=['POST'])
def predict():
    link = request.form.get('link', '')
    if not link:
        return jsonify({'error': 'No link provided'}), 400

    # Fetch the content from the link using requests
    try:
        response = requests.get(link)
        response.raise_for_status()  # Raise an exception for bad responses (4xx, 5xx)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error fetching the link: {str(e)}'}), 400

    # Use BeautifulSoup to parse the HTML content and extract the main text
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])  # Extract all paragraph text

    if not text:
        return jsonify({'error': 'No text found on the provided link'}), 400

    # Vectorize the extracted text
    vectorized_data = vectorizer.transform([text])

    # Predict whether the content indicates civil unrest or not
    prediction = model.predict(vectorized_data)[0]

    # Return the result as a JSON response
    if prediction == 1:
        result = "Civil Unrest"
    else:
        result = "No Civil Unrest"

    return jsonify({'prediction': result})


if __name__ == "__main__":
    app.run(debug=True)
