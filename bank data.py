#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# ======================
# 1. Import Libraries
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[3]:


# ======================
# 2. Load Data
# ======================
df = pd.read_csv("C:/Users/kprab/Downloads/bank.csv")

# Drop leakage columns
df = df.drop(['duration', 'contact'], axis=1)


# In[4]:


# ======================
# 3. Basic EDA
# ======================
print(df.info())
print(df['deposit'].value_counts(normalize=True))

# Plot class distribution
sns.countplot(data=df, x='deposit')
plt.title("Target Distribution (deposit)")
plt.show()

# Example numerical feature histograms
df[['age', 'balance', 'campaign']].hist(bins=30, figsize=(10,5))
plt.tight_layout()
plt.show()


# In[5]:


# Target
y = df['deposit'].map({'yes': 1, 'no': 0})

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df['deposit'].value_counts(normalize=True))


# In[6]:


# ======================
# 3. Define Feature Groups
# ======================
feature_groups = {
    "Demographics": ['age', 'job', 'marital', 'education'],
    "Financials": ['balance', 'housing', 'loan', 'default'],
    "Campaign History": ['campaign', 'pdays', 'previous', 'poutcome'],
    "Temporal": ['month', 'day']
}


# In[7]:


# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}


# In[8]:


# ======================
# 4. Incremental Training Function
# ======================
def incremental_comparison(df, y, feature_groups, models):
    results = []
    used_features = []

    for group_name, features in feature_groups.items():
        # Add features incrementally
        used_features.extend(features)
        X = df[used_features]

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )


# In[9]:


def incremental_comparison(df, y, feature_groups, models):
   results = []
   used_features = []

   # Loop through feature groups
   for group_name, features in feature_groups.items():
       used_features.extend(features)
       X = df[used_features]

       # Split into train/val
       X_train, X_val, y_train, y_val = train_test_split(
           X, y, test_size=0.2, stratify=y, random_state=42
       )

       # Preprocessing
       cat_features = X.select_dtypes(include=['object']).columns.tolist()
       num_features = X.select_dtypes(exclude=['object']).columns.tolist()

       preprocessor = ColumnTransformer(
           transformers=[
               ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
               ('num', StandardScaler(), num_features)
           ]
       )

       # Loop through models
       for name, model in models.items():
           pipeline = Pipeline(steps=[
               ('preprocessor', preprocessor),
               ('model', model)
           ])

           pipeline.fit(X_train, y_train)
           y_proba = pipeline.predict_proba(X_val)[:, 1]
           roc = roc_auc_score(y_val, y_proba)

           results.append({
               "Feature Set": group_name,
               "Model": name,
               "ROC-AUC": roc
           })

   return pd.DataFrame(results)


# In[10]:


results_df = incremental_comparison(df, y, feature_groups, models)

# Pivot for easier comparison
results_pivot = results_df.pivot(index="Feature Set", columns="Model", values="ROC-AUC")
print(results_pivot.round(3))


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# === Use all features together (final stage) ===
X = df.drop(columns=['deposit'])
y = df['deposit'].map({'yes': 1, 'no': 0})

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ]
)

# === Train final model ===
model = RandomForestClassifier(random_state=42, class_weight="balanced")
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
pipeline.fit(X_train, y_train)

# === Predict and evaluate ===
y_proba = pipeline.predict_proba(X_val)[:, 1]
threshold = 0.4
y_pred = (y_proba >= threshold).astype(int)

print(classification_report(y_val, y_pred))

# === Plot confusion matrix ===
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred No", "Pred Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.title(f"Confusion Matrix (Random Forest, threshold={threshold})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()



# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
get_ipython().run_line_magic('matplotlib', 'inline')

thresholds = np.arange(0.3, 0.7, 0.02)
precisions, recalls = [], []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_val, y_pred_t))
    recalls.append(recall_score(y_val, y_pred_t))

plt.figure(figsize=(8,5))
plt.plot(thresholds, precisions, marker='o', label='Precision')
plt.plot(thresholds, recalls, marker='o', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision–Recall vs Threshold (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()


# In[22]:


from sklearn.metrics import f1_score

thresholds = np.arange(0.3, 0.8, 0.02)
f1s, precs, recs = [], [], []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    precs.append(precision_score(y_val, y_pred_t))
    recs.append(recall_score(y_val, y_pred_t))
    f1s.append(f1_score(y_val, y_pred_t))

best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

print(f"Best threshold by F1: {best_threshold:.2f}, "
      f"Precision={precs[best_idx]:.2f}, Recall={recs[best_idx]:.2f}, F1={f1s[best_idx]:.2f}")

plt.figure(figsize=(8,5))
plt.plot(thresholds, precs, marker='o', label='Precision', color='orange')
plt.plot(thresholds, recs, marker='o', label='Recall', color='blue')
plt.plot(thresholds, f1s, marker='o', label='F1-score', color='green')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold={best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall & F1 vs Threshold')
plt.legend()
plt.grid(True)
plt.show()


# In[24]:


best_threshold = 0.62
y_pred_opt = (y_proba >= best_threshold).astype(int)

print(f"=== Confusion Matrix at Threshold {best_threshold} ===")
print(classification_report(y_val, y_pred_opt))

sns.heatmap(confusion_matrix(y_val, y_pred_opt),
            annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred No", "Pred Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.title(f"Random Forest Confusion Matrix (Threshold={best_threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Extract trained model and preprocessor
rf_model = rf_tuned.named_steps['model']
preproc = rf_tuned.named_steps['preprocessor']

# Get feature names
ohe = preproc.named_transformers_['cat']
encoded_features = list(ohe.get_feature_names_out(cat_features))
all_features = encoded_features + num_features

# Get importances
importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top 15
plt.figure(figsize=(8,6))
sns.barplot(y='Feature', x='Importance', data=importances.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('')
plt.tight_layout()
plt.show()


# In[30]:


jupyter nbconvert --to script bank_data.ipynb


# In[ ]:




