# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import shap

st.header("Web-based artificial intelligence application for predicting sepsis among patients with trauma: an internationally validated cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Abdomineinjury = st.sidebar.selectbox("Abdominal trauma", ("No", "Yes"))
Openinjury = st.sidebar.selectbox("Open trauma", ("No", "Yes"))
Smoking = st.sidebar.selectbox("Smoking", ("No", "Yes"))
ISS = st.sidebar.slider("ISS", 15, 50)
SOFA = st.sidebar.slider("SOFA", 0, 18)
GCS = st.sidebar.slider("GCS", 3, 15)
Redbloodcelcount = st.sidebar.slider("Red blood cell count(×10^9/L)", 1.00, 6.00)
Heartrate = st.sidebar.slider("Heart rate (BMP)", 30, 160)
Respiratoryrate = st.sidebar.slider("Respiratory rate (BMP)", 10, 40)
Hct = st.sidebar.slider("Hematocrit", 0.1000, 0.6000)
Totalprotein = st.sidebar.slider("Total protein (g/L)", 20.0, 80.0)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_roundnew.pkl")
    x = pd.DataFrame([[Gender, Abdomineinjury, Openinjury, Smoking, ISS, SOFA, GCS, Redbloodcelcount, Heartrate, Respiratoryrate, Hct, Totalprotein]],
                     columns=["Gender", "Abdomineinjury", "Openinjury", "Smoking", "ISS", "SOFA", "GCS", "Redbloodcelcount", "Heartrate", "Respiratoryrate", "Hct", "Totalprotein"])
    x = x.replace(["Male", "Female"], [1, 0])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of sepsis: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.254:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")
    if prediction < 0.254:
        st.success(f"For low-risk sepsis patients, the focus is on early recognition and prompt treatment of any signs of infection. This may include regular assessments of temperature, heart rate, and respiratory rate, as well as close monitoring of any surgical wounds or invasive lines. In addition, appropriate antibiotic therapy and fluid management are important to prevent the development of sepsis in these patients.")
    else:
        st.error(f"For high-risk patients, early identification and management of sepsis are necessary, including monitoring vital signs, microbiological examinations, and imaging studies, stabilizing respiration and continuously monitoring oxygenation through pulse oximetry, and performing endotracheal intubation and mechanical ventilation when necessary to meet the increased respiratory effort commonly associated with sepsis. It is important to establish venous access as soon as possible for high-risk sepsis patients, although peripheral venous access may meet the needs of some patients during initial resuscitation, central venous access is required for most patients during treatment for the administration of intravenous fluids, medications, blood products, and frequent laboratory testing. Specific recommendations for identifying the focus of sepsis and controlling the source of infection, as well as early administration of antibiotics and fluid resuscitation, are provided. In addition, healthcare institutions should strictly adhere to trauma patient management protocols and infection control protocols to prevent early exacerbation of immune suppression and increased susceptibility to infection in the initial stages after trauma, and to minimize the risk of healthcare-associated infections that may lead to sepsis.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_train0 = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["Gender", "Abdomineinjury", "Openinjury", "Smoking", "ISS", "SOFA", "GCS", "Redbloodcelcount", "Heartrate", "Respiratoryrate", "Hct", "Totalprotein"]]
    y_train = y_train0.Group
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    # st.text(shap_value)

    shap.initjs()
    # image = shap.plots.force(shap_value)
    # image = shap.plots.bar(shap_value)
    st.pyplot(shap.plots.waterfall(shap_value[0]))
    st.pyplot(shap.plots.force(shap_value[0], matplotlib=True))
    #st.pyplot(shap.plots.bar(shap_value[0]))
    st.text(f"Note: Blue items indicate protective factors, while red items indicate risk factors.")
    st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Model introduction')
st.markdown('This freely accessible AI application, exclusively intended for research purposes, has been specifically designed to comprehensively assess the risk of sepsis in critically injured patients within intensive care units (ICUs). The AI application was developed using the eXGBM model, with a prediction performance of 0.912 (AUC). This tool may empower healthcare professionals to make more accurate and timely decisions, ultimately reducing the sepsis rates and improving patient outcomes.')
