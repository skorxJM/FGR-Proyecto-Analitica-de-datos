import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_excel("data/FGR_dataset.xlsx")  
print("Columnas:", df.columns)

X = df.drop(columns=['C31'])  
y = df['C31']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

print("Distribución de clases luego de SMOTE:")
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Exactitud: {accuracy * 100:.2f} %")
print("Matriz de confusión:")
print(conf_matrix)

# Visualización gráfica
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Restricción"], yticklabels=["Normal", "Restricción"])
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión - Regresión Logística")
plt.tight_layout()
plt.show()

joblib.dump(scaler, 'models/logistic_scaler.pkl')
joblib.dump(model, 'models/logistic_model.pkl')
