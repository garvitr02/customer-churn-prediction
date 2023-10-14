import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Churn Prediction Model",
    layout="centered"
)
model_path = "C:\\Users\\garvi\\1_MachineLearning\\CustomerChurn\\model.pkl"

with open(model_path, "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

st.markdown("# Customer Churn Prediction Model")


age = st.text_input(label="Age")
gender = st.selectbox(label="Gender", options=['Male', 'Female'])
location = st.selectbox(
                label="Location",
                options=['Houston', 'Los Angeles', 'Miami', 'Chicago', 'New York']
            )
subscription_length_months = st.text_input(label="Number of months of subscription")
monthly_bill = st.text_input(label="monthly bill")
total_used_gb = st.text_input(label="How many GB of content is consumed")


features = [{
    'Age': age,
    'Gender': gender,
    'Location': location,
    'Subscription_Length_Months': subscription_length_months,
    'Monthly_Bill': monthly_bill,
    'Total_Usage_GB': total_used_gb
}]


features = pd.DataFrame(features)


if total_used_gb:
    output = classifier.predict(features)
    if output == 1:
        st.markdown("### :red[Churn Warning]")
        st.write("With a '1' value, the customer might be leaving.")
    if output == 0:
        st.markdown("### :green[Churn Resistant]")
        st.write("With a '0' value, the cutomer is likely to stay.")
