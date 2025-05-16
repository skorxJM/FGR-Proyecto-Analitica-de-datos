import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_excel("data/FGR_dataset.xlsx")
print("Columnas:", df.columns)

X = df.drop("C31", axis=1)
y = df["C31"]


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Distribución de clases luego de SMOTE:")
print(y_resampled.value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=5)

grid.fit(X_train, y_train)

print("\nMejores parámetros encontrados por GridSearchCV:")
print(grid.best_params_)

# Predecir y evaluar
y_pred = grid.predict(X_test)

unique, counts = np.unique(y_pred, return_counts=True)
print("Distribución de predicciones en test:", dict(zip(unique, counts)))


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\nExactitud: {accuracy * 100:.2f} %")
print("\nMatriz de confusión:")
print(cm)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# --- Visualizar matriz de confusión ---
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta real')
plt.title('Matriz de Confusión - SVM')
plt.tight_layout()
plt.show()


joblib.dump(scaler, 'models/svm_scaler.pkl')

joblib.dump(grid.best_estimator_, 'models/svm_model.pkl')
