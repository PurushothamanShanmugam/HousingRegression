# 🏠 Housing Price Prediction – ML Ops Project

This project implements a complete ML workflow for predicting house prices using the Boston Housing dataset. It covers data loading, regression model training, hyperparameter tuning, and CI automation using GitHub Actions.

---

## 📁 Project Structure

```
HousingRegression/
├── .github/workflows/ci.yml      # GitHub Actions CI workflow
├── utils.py                      # Data loading and preprocessing
├── regression.py                 # Model training and evaluation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/PurushothamanShanmugam/HousingPricingRegression.git
cd HousingPricingRegression
```

### 2. Create and Activate Virtual Environment (Conda Recommended)

```bash
conda create -n housing-ml python=3.10 -y
conda activate housing-ml
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 👉 For `reg` Branch (Basic Models)

```bash
git checkout reg
python regression.py
```

You’ll see output with MSE and R² scores for:
- Linear Regression
- Decision Tree
- Random Forest

---

### 👉 For `hyper` Branch (With Hyperparameter Tuning)

```bash
git checkout hyper
python regression.py
```

You’ll see best parameters for each model and improved metrics using GridSearchCV.

---

## 🤖 CI Automation with GitHub Actions

This project includes a GitHub Actions workflow that:
- Runs on every push
- Installs dependencies
- Executes `regression.py` to ensure code is working

You can find it in `.github/workflows/ci.yml`.

---

## 📌 Notes

- The Boston Housing dataset is loaded manually from [StatLib](http://lib.stat.cmu.edu/datasets/boston).
- Keep `main`, `reg`, and `hyper` branches intact. Do not delete any.
- Use CLI for Git operations (not web UI) to avoid penalties.

---

## 👤 Author

- **Name**: Purushothaman S  
- **Roll Number**: g24ai1042@iitj.ac.in
