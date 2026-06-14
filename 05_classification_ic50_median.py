import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv('cleaned_molecular_data.csv')

target_col = 'IC50, mM'
y = (df[target_col] > df[target_col].median()).astype(int)

# удаление все целевые переменные из обучающей выборки для предотвращения утечки данных
targets_all = ['IC50, mM', 'CC50, mM', 'SI', 'IC50, mM_log', 'CC50, mM_log', 'SI_log']
X = df.drop(columns=targets_all)

# разделение и масштабирование
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== ЗАДАЧА: Классификация IC50 > Медианы ===")
print(f"Баланс классов в таргете:\n{y.value_counts()}\n")

# обучение моделей и оценка
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    "XGBoost Classifier": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"[{name}]")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("-" * 30)