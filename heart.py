import streamlit as st

# Set page title and icon
st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:")

# Set layout width
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Image URL
image_url = "https://img.freepik.com/premium-photo/human-heart-illustration-glowing-design-3d-effect-with-isolated-background_800563-503.jpg"
# Center the image
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="{image_url}" width="400">
    </div>
    """,
    unsafe_allow_html=True
)


background_css = """
<style>
body {
    background: linear-gradient(135deg, #ff7e5f, #feb47b); /* Gradient background with two colors */
    font-family: Arial, sans-serif; /* Choose a readable font */
    margin: 0; /* Remove default body margin */
    padding: 0; /* Remove default body padding */
}

.stTextInput>div>div>input {
    background-color: rgba(255, 255, 255, 0.8); /* White input field with some transparency */
    border: 1px solid #ccc; /* Light gray border */
    border-radius: 5px; /* Rounded corners */
    padding: 8px 12px; /* Padding inside the input field */
    font-size: 16px; /* Larger font size */
    color: #333; /* Text color */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Add a subtle shadow to the input field */
}

/* Add more custom styles as needed */
</style>
"""

# Apply custom CSS
st.markdown(background_css, unsafe_allow_html=True)


# Display information about heart diseases
# Define custom CSS for the title
title_css = """
<style>
.title {
    color: #ff6347; /* Red color for the title */
    font-size: 36px; /* Larger font size */
    font-weight: bold; /* Bold font weight */
    text-align: center; /* Center-align the title */
    text-transform: uppercase; /* Uppercase text */
    margin-bottom: 30px; /* Add some bottom margin */
}
</style>
"""

# Apply custom CSS
st.markdown(title_css, unsafe_allow_html=True)

# Display visually appealing title
st.markdown('<h1 class="title">Heart Diseases Prediction</h1>', unsafe_allow_html=True)
import streamlit as st

# Define custom CSS for the Did You Know section
did_you_know_css = """
<style>
.did-you-know {
    background-color: #f0f0f0; /* Light gray background */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Shadow effect */
}

.did-you-know h2 {
    color: #ff6347; /* Red color for the heading */
}

.did-you-know p {
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 10px;
}

.did-you-know ul {
    margin-left: 20px;
    list-style-type: square;
}

.did-you-know ul li {
    margin-bottom: 5px;
}

</style>
"""

# Apply custom CSS
st.markdown(did_you_know_css, unsafe_allow_html=True)

# Display Did You Know section
with st.expander("Did You Know?"):
    st.markdown("<h2>Heart Diseases</h2>", unsafe_allow_html=True)
    st.markdown("<p>Heart diseases are cardiovascular diseases, encompassing a range of conditions that affect the heart and blood vessels. Some common types include coronary artery disease, heart failure, and arrhythmias.</p>", unsafe_allow_html=True)
    st.markdown("<h3>Symptoms of Heart Diseases:</h3>", unsafe_allow_html=True)
    st.markdown("<ul>", unsafe_allow_html=True)
    st.markdown("<li>Chest Discomfort</li>", unsafe_allow_html=True)
    st.markdown("<li>Shortness of Breath</li>", unsafe_allow_html=True)
    st.markdown("<li>Fatigue</li>", unsafe_allow_html=True)
    st.markdown("<li>Rapid or Irregular Heartbeat</li>", unsafe_allow_html=True)
    st.markdown("<li>Dizziness or Fainting</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("<h3>Precautions for Heart Health:</h3>", unsafe_allow_html=True)
    st.markdown("<ul>", unsafe_allow_html=True)
    st.markdown("<li>Maintaining a Healthy Diet</li>", unsafe_allow_html=True)
    st.markdown("<li>Regular Exercise</li>", unsafe_allow_html=True)
    st.markdown("<li>Healthy Weight</li>", unsafe_allow_html=True)
    st.markdown("<li>Quitting Smoking</li>", unsafe_allow_html=True)
    st.markdown("<li>Limiting Alcohol Intake</li>", unsafe_allow_html=True)
    st.markdown("<li>Managing Stress</li>", unsafe_allow_html=True)
    st.markdown("<li>Regular Health Checkups</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

# Display descriptions of attributes
with st.expander("Descriptions of Attributes"):
    descriptions = {
        'Thalach': 'Maximum heart rate achieved during exercise, essential for assessing cardiovascular fitness and heart disease risk.',
        'Old Peak': 'Measure of ST depression during exercise relative to rest, indicating potential myocardial ischemia and coronary artery disease.',
        'CP (Chest Pain Type)': 'Classification system for characterizing chest pain symptoms, aiding in diagnosis and differentiation of cardiac and non-cardiac causes.',
        'CA (Number of Major Vessels Colored by Fluoroscopy)': 'Quantity of major coronary arteries exhibiting blockages or abnormalities, crucial for determining severity and management of coronary artery disease. A value of 0 indicates normalcy, 1 indicates mild disease, and 2 indicates moderate disease.',
        'Exang (Exercise Induced Angina)': 'Presence of exercise-induced angina (chest pain) during physical activity.',
        'Chol (Serum Cholesterol Level)': 'Serum cholesterol level in mg/dl.',
        'Age': 'Age of the patient in years.',
        'Trestbps (Resting Blood Pressure)': 'Resting blood pressure (in mm Hg) at the time of admission to the hospital.',
        'Slope': 'Slope of the peak exercise ST segment, reflecting heart rate response to exercise.',
        'Sex': 'Gender of the patient (0 for Female, 1 for Male).'
    }
    for attribute, description in descriptions.items():
        st.write(f"**{attribute}**: {description}")

# Define numeric range for attributes
numeric_range = {
    'Thalach': '71 - 202 bpm',
    'Old Peak': '0 - 6.2',
    'CA': '0 - 3',
    'CP': '0 - 3',
    'Exang': '0 - 1',
    'Chol': '126 mg/dl - 564 mg/dl',
    'Age': '0 - 100',
    'Trestbps': '94 - 210 mg/dl',
    'Slope': '0 - 2',
    'Sex': '0 - 1'
}

# Display numeric range table within an expander
with st.expander("Numeric Range for Attributes"):
    df_range = pd.DataFrame(numeric_range.items(), columns=['Attribute', 'Range'])
    st.table(df_range)
# Load dataset
df = pd.read_csv('h.csv')

# Select features and target variable
selected_features = ['thalach', 'oldpeak', 'cp', 'ca', 'exang', 'chol', 'age', 'trestbps', 'slope', 'sex']
X_selected = df[selected_features]
y = df['target']

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Perform MinMax scaling on the DataFrame with selected features
scaled_df = pd.DataFrame(scaler.fit_transform(X_selected), columns=X_selected.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

# Initialize models
logreg = LogisticRegression()
svm_model = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()

# Fit models
logreg.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_nb = nb.predict(X_test)

# Evaluate models
models = {'Logistic Regression': y_pred_logreg, 'SVM': y_pred_svm, 'Decision Tree': y_pred_dt,
          'Random Forest': y_pred_rf, 'Naive Bayes': y_pred_nb}
for model_name, y_pred in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


# Evaluate models using cross-validation
for model in [logreg, svm_model, dt, rf, nb]:
    accuracy_scores = cross_val_score(model, scaled_df, y, cv=5, scoring='accuracy')
    precision_scores = cross_val_score(model, scaled_df, y, cv=5, scoring='precision')
    recall_scores = cross_val_score(model, scaled_df, y, cv=5, scoring='recall')
    f1_scores = cross_val_score(model, scaled_df, y, cv=5, scoring='f1')


# Ensemble model
base_models = [('Logistic Regression', logreg), ('SVM', svm_model), ('Decision Tree', dt), ('Random Forest', rf), ('Naive Bayes', nb)]
ensemble_model = VotingClassifier(estimators=base_models, voting='hard')
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)


# Cross-validation for ensemble model
accuracy_scores = cross_val_score(ensemble_model, scaled_df, y, cv=5, scoring='accuracy')
precision_scores = cross_val_score(ensemble_model, scaled_df, y, cv=5, scoring='precision')
recall_scores = cross_val_score(ensemble_model, scaled_df, y, cv=5, scoring='recall')
f1_scores = cross_val_score(ensemble_model, scaled_df, y, cv=5, scoring='f1')

# Function to make prediction
def make_prediction(user_input):
    scaled_user_input = scaler.transform(user_input)
    prediction = ensemble_model.predict(scaled_user_input)
    return prediction

# Define custom CSS for the input form
input_form_css = """
<style>
/* Style the form header */
.user-input-header {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Style the input fields */
.user-input-field {
    margin-bottom: 15px;
}

/* Style the submit button */
.user-input-submit {
    margin-top: 20px;
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 18px;
}

/* Change submit button color on hover */
.user-input-submit:hover {
    background-color: #45a049;
}

/* Style the form footer */
.user-input-footer {
    margin-top: 20px;
    font-size: 14px;
    color: #888;
}
</style>
"""

# Apply custom CSS
st.markdown(input_form_css, unsafe_allow_html=True)

# Display the input form
with st.form("user_input_form"):
    st.markdown('<div class="user-input-header">Enter Patient Information for Prediction</div>', unsafe_allow_html=True)

    thalach = st.number_input("Thalach (Max heart rate during exercise):", min_value=0.0, key='thalach', value=0.0, format="%.2f", help="bpm")
    oldpeak = st.number_input("Old Peak (ST depression during exercise):", min_value=0.0, key='oldpeak', value=0.0, format="%.2f")
    cp = st.number_input("CP (Chest Pain Type):", min_value=0, key='cp', value=0)
    exang = st.number_input("Exang (Exercise Induced Angina):", min_value=0, key='exang', value=0)
    ca = st.number_input("CA (Major vessels colored by Fluoroscopy):", min_value=0.0, key='ca', value=0.0, format="%.2f")
    chol = st.number_input("Chol (Serum Cholesterol Level):", min_value=0.0, key='chol', value=0.0, format="%.2f")
    age = st.number_input("Age:", min_value=0, key='age', value=0)
    trestbps = st.number_input("Trestbps (Resting Blood Pressure):", min_value=0.0, key='trestbps', value=0.0, format="%.2f")
    slope = st.number_input("Slope of peak exercise ST segment:", min_value=0, key='slope', value=0)
    sex = st.text_input("Sex (0 for Female, 1 for Male):", key='sex', value="")

    submit_button = st.form_submit_button("Submit")

if submit_button:
    valid_inputs = None not in [thalach, oldpeak, cp, ca, exang, chol, age, trestbps, slope, sex] and sex in ['0', '1']
    if valid_inputs:
        user_input = pd.DataFrame({'thalach': [thalach], 'oldpeak': [oldpeak], 'cp': [cp], 'ca': [ca], 'exang': [exang],
                                   'chol': [chol], 'age': [age], 'trestbps': [trestbps], 'slope': [slope], 'sex': [int(sex)]})
        prediction = make_prediction(user_input)
        prediction_text = "**The model predicts that the patient is likely to have heart disease.**" if prediction[0] == 1 \
            else "**The model predicts that the patient is unlikely to have heart disease.**"
        
        # Define color based on prediction
        color = 'red' if prediction[0] == 1 else 'green'
        
        # Write prediction text with specified color
        st.markdown(f'<font color="{color}">{prediction_text}</font>', unsafe_allow_html=True)

css = """
@keyframes slidein {
    from {
        margin-left: 100%;
        width: 300%; /* Adjusted width for the container */
    }

    to {
        margin-left: 0%;
        width: 600%; /* Adjusted width for the container */
    }
}

.slide-in {
    animation: slidein 10s infinite linear; /* Adjusted animation duration */
}
"""

# Display the CSS style
st.write("<style>{}</style>".format(css), unsafe_allow_html=True)

# Display the sliding message
st.markdown("<div class='slide-in'>Please note that the predictions provided by this app are based on machine learning algorithms and should be used for informational purposes only. "
            "Consult a healthcare professional for accurate diagnosis and medical advice.</div>", unsafe_allow_html=True)
