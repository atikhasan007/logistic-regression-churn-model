import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model
model = pickle.load(open('7_logistic_model.pkl','rb'))

# Create web app
st.title("Logistic Regression for Churn Prediction")

gender = st.selectbox("Select Gender", options=['Female', 'Male'])
SeniorCitizen = st.selectbox('Are you a senior citizen?', options=['Yes', 'No'])
Partner = st.selectbox("Do you have a partner?", options=['Yes', 'No'])
Dependents = st.selectbox('Are you dependent on others?', options=['Yes', 'No'])
tenure = st.text_input('Enter your tenure:')
PhoneService = st.selectbox('Do you have phone service?', options=['Yes', 'No'])
MultipleLines = st.selectbox("Do you have multiple line services?", options=['Yes', 'No', 'No phone service'])
Contract = st.selectbox('Your Contract Type?', options=['One year', 'Two year', 'Month-to_month'])
TotalCharges = st.text_input("Enter your Total Charges:")

# Prediction function
def predictive(gender, SeniorCitizen, Partner,tenure, Dependents, PhoneService, MultipleLines, Contract, TotalCharges):
    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'tenure': tenure,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'Contract': Contract,
        'TotalCharges': TotalCharges
    }

    df = pd.DataFrame([data])

    # Encode all categorical variables
    for column in df.columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    df = StandardScaler().fit_transform(df)
    prediction = model.predict(df)

    return prediction[0]


#==============================
# Tips for Churn Prevention
churn_tips_data = {
    "Tips for Churn Prevention": [
        "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
        "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
        "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
        "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
        "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
        "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
        "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
        "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
        "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
        "Competitive Analysis: Stay aware of what competitors are offering. Ensure your products, services, and workplace environment remain competitive in the market."
    ]
}

# Tips for Customer Retention (Not Churning)
retention_tips_data = {
    "Tips for Customer Retention": [
        "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
        "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
        "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
        "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
        "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
        "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
        "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
        "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
        "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
        "Show Appreciation: Express gratitude to loyal customers and acknowledge their continued support."
    ]
}

#create DataFrames
churn_tips_df = pd.DataFrame(churn_tips_data)
retention_tips_df = pd.DataFrame(retention_tips_data)


# Button for prediction
if st.button('Predict churn or not'):
    result = predictive(
        gender,
        SeniorCitizen,
        Partner,
        tenure,
        Dependents,
        PhoneService,
        MultipleLines,
        Contract,
        TotalCharges
    )

    if result == 1:
        st.success("Customer is **likely to churn**.")
        st.write('here are 10 tips for churn prevention:')
        st.dataframe(churn_tips_df, height=400,width=600)

    else:
        st.title("Customer is **not likely to churn**.")
        st.write('here are 10 tips for customer Retention (not Churning):')
        st.dataframe(retention_tips_df, height=400, width=600)
