import re
import string
from flask import Flask, request, render_template
import pandas as pd
import openpyxl
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import scipy.sparse

app = Flask(__name__)

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

# def predict_model(input_text):
#     onehot_encoder = OneHotEncoder()
#     conclusions_reshaped = conclusions.values.reshape(-1, 1)
#     encoded_features = onehot_encoder.fit_transform(conclusions_reshaped)
#
#     label_encoder = LabelEncoder()
#     encoded_labels = label_encoder.fit_transform(medical_conditions)
#
#     model = MultinomialNB()
#     model.fit(encoded_features, encoded_labels)
#     new_conclusion = 'familial'
#
#     new_conclusion = clean_text(new_conclusion)
#
#     new_conclusion_encoded = onehot_encoder.transform([[new_conclusion]])
#     predicted_condition_label = model.predict(new_conclusion_encoded)
#
#     predicted_condition = label_encoder.inverse_transform(predicted_condition_label)
#
#     suggested_medications = dataset2[dataset2['medical_condition'] == predicted_condition[0]]['drug_name']
#     suggested_medications_list = suggested_medications.tolist()
#     if len(suggested_medications_list)==0:
#         return "No medicine found"
#     else:
#         return suggested_medications_list

onehot_encoder = OneHotEncoder()

conclusions_reshaped = conclusions.values.reshape(-1, 1)
encoded_features_sparse = onehot_encoder.fit_transform(conclusions_reshaped)

encoded_features = encoded_features_sparse.toarray()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(medical_conditions)

model = MultinomialNB()
model.fit(encoded_features, encoded_labels)


def predict_model(input_text):
    cleaned_input = clean_text(input_text)
    input_encoded = onehot_encoder.transform([[cleaned_input]])

    predicted_condition_label = model.predict(input_encoded)
    predicted_condition = label_encoder.inverse_transform(predicted_condition_label)

    suggested_medications = dataset2[dataset2['medical_condition'] == predicted_condition[0]]['drug_name']
    suggested_medications_list = suggested_medications.tolist()

    return suggested_medications_list

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
        drugs = predict_model(conclusion)
        return render_template('results.html', drugs=drugs)
    return render_template('index.html')
    # if request.method == 'POST':
    #     conclusion = request.form['conclusion']
    #     drugs = get_drug_names_for_conclusion(conclusion, dataset1, dataset2)
    #     return render_template('results.html', drugs=drugs)
    # return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
