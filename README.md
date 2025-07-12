# 💼 Employee Salary Prediction Using ML

This project is a **Streamlit-based web application** that predicts whether an employee's income is `>50K` or `<=50K` using a machine learning model trained on the **UCI Adult dataset**.

---

## 📂 Project Structure

```
employee-salary-prediction/
├── salary_model.pkl           # Trained VotingRegressor model (Random Forest + Gradient Boosting)
├── adult 3.csv                # Cleaned dataset used for model training and visualizations
├── streamlit_app.py           # Streamlit web application
├── requirements.txt           # Dependencies list (for deployment)
└── README.md                  # Project documentation
```

---

## 🚀 Features

* Predict income category (`>50K` or `<=50K`) based on personal and professional details
* Input fields include:

  * Age
  * Gender
  * Education
  * Workclass
  * Marital Status
  * Occupation
  * Relationship
  * Race
  * Capital Gain / Loss
  * Hours per Week
  * Native Country
  * fnlwgt
  * Educational Number
* Visualizations in Sidebar:

  * Income distribution
  * Education vs Income
  * Work hours vs Income
  * Average Income by Occupation

---

## 🛠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Seaborn & Matplotlib

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Info

* **Model:** Voting Regressor (RandomForestRegressor + GradientBoostingRegressor)
* **Target:** `income` (converted to binary 0/1)
* **R² Score:** \~0.49
* **Evaluation:** MSE, R²

---

## 📁 Dataset Info

* Source: UCI Adult Income Dataset
* File used: `adult 3.csv`
* Cleaned: Replaced `?` with `NaN` and dropped rows with missing data

---

## 🧑‍💻 Author

**Abhishek Saurabh**
B.Tech CSE | Full Stack & AI Enthusiast
[LinkedIn](https://www.linkedin.com/in/abhishek-saurabh/) | [GitHub](https://github.com/abhisheksaurabh)

---

## 📜 License

This project is for educational and internship project purposes only.
