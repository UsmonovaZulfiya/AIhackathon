import re
import string

from flask import Flask, request, render_template
import pandas as pd
import openpyxl

app = Flask(__name__)

# Load your datasets here
dataset1 = pd.read_excel('medical_descriptions_conclusions.xlsx')
dataset2 = pd.read_excel('dataset2.xlsx')

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

dataset1['medical_condition'] = dataset1['medical_condition'].apply(clean_text)
dataset2['medical_condition'] = dataset2['medical_condition'].apply(clean_text)
dataset1['conclusion'] = dataset1['conclusion'].apply(clean_text)

conclusions = dataset1['conclusion']
medical_conditions = dataset1['medical_condition']

def get_drug_names_for_conclusion(conclusion, dataset1, dataset2):
    medical_conditions = dataset1[dataset1['conclusion'].str.lower() == conclusion.lower()][
        'medical_condition'].tolist()
    if not medical_conditions:
        return "No matching medical condition found for the given conclusion."

    drugs = dataset2[dataset2['medical_condition'].isin(medical_conditions)]['drug_name'].tolist()

    return drugs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        conclusion = request.form['conclusion']
        drugs = get_drug_names_for_conclusion(conclusion, dataset1, dataset2)
        return render_template('results.html', drugs=drugs)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
