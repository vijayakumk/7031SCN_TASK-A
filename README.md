# 🌫️ Air Quality Classification Using Artificial Neural Networks

## Overview
This project applies Artificial Neural Networks (ANN) to classify air quality levels based on PM2.5 concentrations using the **Beijing Multi-Site Air Quality Dataset**. It is completed as part of the MSc AI & Human Factors coursework at Coventry University.

## Dataset
- **Source:** [Beijing Multi-Site Air Quality Data – UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
- **Size:** ~420,000 records across 12 monitoring stations (2013–2017)
- **Features:** PM2.5, PM10, SO₂, NO₂, CO, O₃, temperature, pressure, humidity, wind speed/direction
- **Target Variable:** AQI Category (Good / Moderate / Unhealthy / Very Unhealthy / Hazardous)

## Objectives
- Preprocess and clean multi-site time-series air quality data
- Engineer AQI-based classification labels from PM2.5 readings
- Build and compare two ANN architectures:
  - **MLP (Multi-Layer Perceptron)** – shallow feedforward network
  - **Deep Feedforward ANN** – multiple hidden layers with dropout regularisation

## Project Structure
## Models & Architecture
| Model | Hidden Layers | Activation | Regularisation |
|-------|--------------|------------|----------------|
| MLP   | 1–2          | ReLU       | None           |
| Deep ANN | 4–6       | ReLU       | Dropout, BatchNorm |

## Results
> *(To be updated after model training)*

| Model    | Accuracy | F1-Score |
|----------|----------|----------|
| MLP      | —        | —        |
| Deep ANN | —        | —        |

## Technologies Used
- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (preprocessing, evaluation)
- TensorFlow / Keras (model building)
- Google Colab / Jupyter Notebook

## How to Run
1. Clone the repository:
```bash
   git clone https://github.com/Kishore-Vijayakumar/<repo-name>.git
   cd <repo-name>
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Open notebooks in Jupyter or Google Colab and run sequentially.

## Author
**Kishore Vijayakumar**  
MSc AI & Human Factors – Coventry University  
[LinkedIn](https://linkedin.com/in/kishorev22) | [GitHub](https://github.com/Kishore-Vijayakumar)
