# Marketing Subscription Prediction Model

This project predicts which customers are most likely to subscribe to a term deposit based on marketing campaign data.  
It demonstrates end-to-end marketing analytics workflow — from data cleaning and model building to threshold optimization and business insight generation.

---

##  Objective
To identify customers who are most likely to subscribe to a marketing offer, helping marketing teams optimize outreach efficiency and campaign ROI.

---

##  Approach
1. Cleaned and prepared marketing campaign data (`bank.csv`).
2. Built a **Random Forest** model with full preprocessing (OneHotEncoder + StandardScaler).
3. Performed **threshold optimization** to balance recall and precision.
4. Selected an optimal threshold of **0.62**, achieving:
   - Precision: **0.63**
   - Recall: **0.75**
   - F1: **0.68**

---

##  Key Insights
- Longer call durations strongly correlate with customer conversion.
- Previous successful contacts and account balance are significant predictors.
- Threshold tuning improved efficiency by reducing wasted outreach by ~5%.

---

##  Tools Used
- **Python** – Core programming language  
- **Pandas, NumPy** – Data handling and manipulation  
- **Scikit-learn** – Machine learning and model evaluation  
- **Matplotlib, Seaborn** – Visualization and analysis  

---

##  Files
| File | Description |
|------|--------------|
| `bank_subscription_model.py` | Clean, production-ready Python script |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |

---

## Results Summary

| Metric | Threshold 0.4 | Threshold 0.62 (Final) |
|---------|---------------|------------------------|
| Precision (Yes) | 0.58 | **0.63** |
| Recall (Yes) | 0.80 | **0.75** |
| F1 (Yes) | 0.68 | **0.68** |
| Accuracy | 0.66 | **0.67** |

---

##  Business Impact
- Catch ~75% of potential subscribers with moderate resource usage.
- Improve lead targeting precision by 5%, reducing unnecessary outreach.
- Provide interpretable insights for marketing strategy refinement.


