from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load models and data
df = pickle.load(open('model.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

app = Flask(__name__)

def find_index(text):
    text = text.lower()
    t_list = text.split(' ')
    c = 0
    for i in df['Description']:
        for element in t_list:
            if element not in i:
                c = c + 1
                break
            else:
                return c + 1
    return []

def recommend(med):
    id = find_index(med)
    dis = similarity[id]
    med_list = sorted(list(enumerate(dis)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_medis = []
    for i in med_list:
        recommended_medis.append(df.iloc[i[0]].Drug_Name)
    return recommended_medis

@app.route('/')
def index():
    return render_template('symptoms.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    recommendations = recommend(symptoms)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)

