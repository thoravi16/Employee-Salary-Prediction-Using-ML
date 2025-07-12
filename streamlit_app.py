import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and dataset
model = joblib.load("salary_model.pkl")
df = pd.read_csv('adult 3.csv', on_bad_lines='skip')
df = df.replace(" ?", pd.NA).dropna()
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction Using ML")
st.markdown("🔍 Enter employee details below to predict their income category.")

# Sidebar Insights
st.sidebar.header("📊 Data Insights")

if st.sidebar.checkbox("Income Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='income', ax=ax, palette='coolwarm')
    ax.set_xticklabels(['<=50K', '>50K'])
    ax.set_title("Income Distribution")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Education vs Income"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="education", y="income", ax=ax)
    ax.set_title("Education Level vs Income")
    plt.xticks(rotation=45)
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Work Hours vs Income"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="hours-per-week", y="income", hue="education", ax=ax)
    ax.set_title("Work Hours vs Income")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Avg Income by Occupation"):
    income_by_job = df.groupby("occupation")["income"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    income_by_job.plot(kind='barh', ax=ax, color='teal')
    ax.set_title("Avg Income by Occupation")
    st.sidebar.pyplot(fig)

# Main Form Input
st.subheader("🧾 Enter Employee Details")

age = st.slider("Age", 18, 65, 30)
workclass = st.selectbox("Workclass", df["workclass"].unique())
fnlwgt = st.number_input("fnlwgt", min_value=1, value=100000)
education = st.selectbox("Education", df["education"].unique())
educational_num = st.slider("Education Num", 1, 16, 10)
marital_status = st.selectbox("Marital Status", df["marital-status"].unique())
occupation = st.selectbox("Occupation", df["occupation"].unique())
relationship = st.selectbox("Relationship", df["relationship"].unique())
race = st.selectbox("Race", df["race"].unique())
gender = st.selectbox("Gender", df["gender"].unique())
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", df["native-country"].unique())

# Form to dataframe
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Prediction
if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    result = ">50K" if prediction > 0.5 else "<=50K"
    st.success(f"✅ Predicted Income: **{result}**")
