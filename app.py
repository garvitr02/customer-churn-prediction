
import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

import os

model_path = os.path.join("C:\\Users\\garvi\\1_MachineLearning\\CustomerChurn", "model.pkl")

pickle_in = open(model_path, "rb")

classifier = pickle.load(pickle_in)
def predict_Customer_Churn(Age,Gender,Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB):

    prediction = classifier.predict([[Age,Gender,Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB]])
    print(prediction)
    return prediction


def main():
    st.title("Customer Churn Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Churn Predictor App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Age = st.text_input("Age", "Type Here")
    Gender = st.text_input("Gender", "Type Here")
    Location = st.text_input("Location", "Type Here")
    Subscription_Length_Months = st.text_input("Subscription_Length_Months", "Type Here")
    Monthly_Bill = st.text_input("Monthly_Bill", "Type Here")
    Total_Usage_GB = st.text_input("Total_Usage_GB", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_Customer_Churn(Age,Gender,Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB)
    st.success('The output is {}'.format(result))
if __name__ == '__main__':
    main()


