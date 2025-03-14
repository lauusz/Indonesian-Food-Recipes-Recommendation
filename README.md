# 🍽️ Food Recommendation ML System 🚀

Welcome to the **Food Recommendation ML System**! 🎉 This project is all about using **Machine Learning & Text Processing** to suggest delicious **Indonesian food recipes** based on ingredients. Whether you're a food lover, chef, or just looking for cooking inspiration, this project has got you covered! 😋

## 📌 About The Project
This system helps users **find the best Indonesian food recipes** using **TF-IDF and an autoencoder-based recommendation model**. 

You can use this project in two ways:
- **🌍 Web App (Streamlit UI):** A user-friendly interactive web interface.
- **🔗 API (Flask):** A backend service for seamless integration into other applications.

---

## 📊 Dataset
We are using an awesome dataset from Kaggle:  
[📂 Indonesian Food Recipes Dataset](https://www.kaggle.com/datasets/canggih/indonesian-food-recipes)  
This dataset includes a variety of Indonesian food recipes along with their ingredients and other useful details to power our recommendation system. 🚀

---
## 🚀 Getting Started
Follow these simple steps to set up and run the project on your local machine! 🏃💨

### 🛠 1. Clone This Repository
First, clone this repository and move into the project directory:
```bash
# Clone the project from GitHub
git clone https://github.com/lauusz/Indonesian-Food-Recipes-Recommendation.git
cd food-recommendation-ml
```

### 🏗 2. Create a Virtual Environment
Before installing the dependencies, create a virtual environment:
```bash
python -m venv venv
```
Activate it:
- **Windows:**  
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**  
  ```bash
  source venv/bin/activate
  ```

### 📦 3. Install Dependencies
Run this command to install all required libraries:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
You can run the project in **two different modes**:

### 🎨 1. Run Streamlit UI (Interactive Web App)
Want a cool and interactive UI to explore food recommendations? Just run:
```bash
streamlit run main.py
```
Then open the **localhost link** provided in your terminal and start exploring delicious Indonesian recipes! 🤩

### 🔌 2. Run as API (Flask Backend)
Want to integrate this recommendation system with other applications? Use the API mode:
```bash
python app.py
```
This will launch a Flask-based API, so you can send **HTTP requests** and get **recommendations as responses!**

---

## 🎯 Features
✅ **Text processing with TF-IDF** to analyze ingredients & find similar recipes.  
✅ **Autoencoder-based recommendation system** for better suggestions.  
✅ **Two usage modes**: **Streamlit UI** for easy interaction & **Flask API** for integrations.  
✅ **Simple and lightweight setup**, just install dependencies and run! 🎯  

---

## 🤝 Contributing
Want to improve this project? Feel free to contribute! Fork the repo, make changes, and submit a pull request. 🚀

---

Enjoy coding & happy cooking! 🍳🔥

