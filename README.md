# 🚀 Numerical Optimization Project  
## Dynamic Soft-Thresholding for Feature Selection in Regression  

### 👨‍💻 Author  
Sri Karthik Manne  
B.E CSE (AI & ML)  

---

# 📌 Project Overview  
This project focuses on applying **Numerical Optimization techniques** to improve regression model performance for **housing price prediction**.  

We implemented and compared:  
- Linear Regression  
- Ridge Regression (L2 Regularization)  
- LASSO Regression (L1 Regularization)  
- ✅ Dynamic LASSO Optimization (Proposed Method)  

The goal is to **reduce prediction error and perform automatic feature selection** using optimization techniques.

---

# 🎯 Objectives  
- Apply numerical optimization in machine learning  
- Compare regression models  
- Perform feature selection using LASSO  
- Optimize regularization parameter (α) dynamically  
- Evaluate performance using Mean Squared Error (MSE)  

---

# 🧠 Key Concept  
Dynamic Soft-Thresholding is used to automatically select the best regularization parameter (α) by testing multiple values and choosing the one with the **lowest error**.

---

# 📊 Dataset  
- Housing Price Dataset  
- Total Samples: 545  
- Features: 13  
- Target Variable: Price  

### Important Features:
- Area  
- Bedrooms  
- Bathrooms  
- Parking  
- Furnishing Status  

---

# ⚙️ Technologies Used  
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

# 📁 Project Structure  
```
Numerical_Optimization_Project/
│
├── main.py
├── model.py
├── optimizer.py
├── requirements.txt
├── README.md
└── train.csv
```

---

# 🚀 Installation & Setup  

### 1. Clone Repository  
```
git clone https://github.com/YOUR-USERNAME/Numerical-Optimization-Project.git
cd Numerical-Optimization-Project
```

### 2. Install Dependencies  
```
pip install -r requirements.txt
```

### 3. Add Dataset  
Download dataset and place as:  
```
train.csv
```

---

# ▶️ How to Run  
```
python main.py
```

---

# 📈 Results  

| Model              | MSE  |
|-------------------|------|
| Linear Regression | 0.17 |
| Ridge Regression  | 0.18 |
| LASSO Regression  | 0.51 |
| Dynamic LASSO     | Best Performance |

---

# 📊 Output  
- Optimal alpha value  
- Mean Squared Error (MSE)  
- Loss convergence graph  

---

# ✅ Advantages  
- Improved prediction accuracy  
- Automatic feature selection  
- Reduced overfitting  
- Efficient optimization  

---

# ⚠️ Limitations  
- Depends on dataset quality  
- Assumes linear relationships  
- Requires parameter tuning  

---

# 🚀 Future Work  
- Apply deep learning models  
- Use advanced optimization algorithms  
- Work with larger datasets  
- Deploy as real-time system  

---

# 📚 References  
- Scikit-learn Documentation  
- Machine Learning Books  
- Research Papers on LASSO & Ridge Regression  

---

# ⭐ Acknowledgement  
This project is submitted as part of **Numerical Optimization coursework**.
