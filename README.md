# Crop Yield Prediction Using Machine Learning

## Project Overview
This project predicts crop yield using machine learning based on rainfall, temperature, soil fertility, and fertilizer usage. It uses Python, Pandas, Scikit-learn, and Streamlit to provide an interactive dashboard.

## Objectives
- Analyze agricultural data.
- Visualize crop yield trends.
- Predict crop yield using a machine learning model.
- Provide a simple, interactive dashboard.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Features
- View the crop yield dataset.
- Explore summary statistics.
- Visualize rainfall and crop yield relationships.
- Predict crop yield using a Random Forest Regressor.
- View prediction history during the application session.

## Dataset
The sample dataset contains:
- Rainfall (mm)
- Temperature (°C)
- Soil Fertility
- Fertilizer (kg)
- Crop Yield (tons)

**Note:** The included dataset is for demonstration purposes and may contain synthetic data. Predictions should not be used for real agricultural decisions without validation on reliable agricultural data.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Files
- `app.py` — Streamlit application and machine learning model.
- `crop_yield.csv` — Sample crop yield dataset.
- `requirements.txt` — Required Python libraries.
- `README.md` — Project documentation.

## Expected Outcome
An interactive dashboard that analyzes the sample agricultural data and predicts crop yield from user-provided inputs.

## Future Enhancements
- Use verified agricultural datasets.
- Add crop and soil-type selection.
- Compare multiple machine learning algorithms.
- Deploy the application online.

## Author
Krishna Priya S
