# AI-IoT-Based Early Dengue Prediction and Prevention System

An intelligent healthcare system that combines **Artificial Intelligence (AI)** and the **Internet of Things (IoT)** for the early prediction and prevention of dengue fever. The project integrates Machine Learning, Deep Learning, Explainable AI (XAI), and simulated IoT sensor data to provide real-time dengue risk assessment.

---

## Research Information

**Research Title**

AI-IoT-Based Early Dengue Prediction and Prevention System

**Degree**

Master of Science (M.Sc.) in Computer Science & Engineering  
Major: Cybersecurity

**University**

United International University (UIU)  
Dhaka, Bangladesh

**Research Supervisor**

Prof. Khondaker A. Mamun, PhD  
Professor, Department of Computer Science & Engineering  
Director, MSCSE Program & Director, IRIIC  
United International University (UIU)

**Author**

Nazmul Hassan

---

## Project Objectives

- Develop an AI-based dengue prediction model.
- Integrate IoT sensor data for real-time patient monitoring.
- Compare Machine Learning and Deep Learning approaches.
- Build a Hybrid XGBoost-LSTM model.
- Improve model transparency using Explainable AI (SHAP).
- Generate automated dengue risk alerts.

---

## Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Matplotlib
- Joblib
- Git
- GitHub

---

## Project Structure

```text
ai-iot-dengue-prediction-system
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│
├── src/
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── hybrid_xgboost_lstm.py
│   ├── statistical_validation.py
│   ├── shap_analysis.py
│   ├── iot_patient_simulator.py
│   ├── iot_risk_prediction.py
│   └── model_comparison.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Implemented Modules

- Data Loading
- Data Cleaning
- Feature Encoding
- Exploratory Data Analysis (EDA)
- Data Visualization
- XGBoost Classifier
- Leakage-Safe XGBoost
- LSTM Deep Learning Model
- Hybrid XGBoost-LSTM Model
- Model Comparison
- ROC Curve Analysis
- Precision-Recall Curve
- 5-Fold Cross Validation
- 10-Fold Cross Validation
- Explainable AI (SHAP)
- IoT Patient Sensor Simulator
- IoT Risk Prediction and Alert System

---

## Experimental Results

| Model | Accuracy |
|--------|----------|
| XGBoost | 100% |
| Leakage-Safe XGBoost | 100% |
| LSTM | 100% |
| Hybrid XGBoost-LSTM | 100% |

> **Note:** These results were obtained using the current experimental dataset. The perfect accuracy reflects the characteristics of this dataset and should not be interpreted as guaranteed performance on real-world clinical data.

---

## Explainable AI

The project includes SHAP (SHapley Additive exPlanations) for model interpretability.

Generated outputs include:

- SHAP Summary Plot
- SHAP Feature Importance
- SHAP Bar Plot

---

## IoT Workflow

```text
IoT Sensors
      │
      ▼
Patient Data Collection
      │
      ▼
Data Preprocessing
      │
      ▼
Hybrid AI Model
      │
      ▼
Risk Prediction
      │
      ▼
Alert Generation
```

---

## Dataset

**Source**

Dengue Dataset Bangladesh (Kaggle)

---

## Future Work

- Real-time ESP32 integration
- Raspberry Pi edge deployment
- MQTT communication
- Cloud dashboard
- Mobile application
- Hospital integration
- Clinical validation using real patient data

---

## License

This project is released for academic and research purposes.

---

## Author

**Nazmul Hassan**

Senior Engineer (IT)

M.Sc. in CSE (Cybersecurity)

United International University (UIU)

Bangladesh