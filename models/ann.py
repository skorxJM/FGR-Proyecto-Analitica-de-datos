import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

# Cargar datos
df = pd.read_excel("data/FGR_dataset.xlsx")
print("Columnas:", df.columns)

# Variables
X = df.drop("C31", axis=1)
y = df["C31"]

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Distribución de clases luego de SMOTE:")
print(y_resampled.value_counts())

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42)

# Modelo ANN
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Salida binaria

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# Evaluación
y_pred = (model.predict(X_test) > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"Exactitud: {acc * 100:.2f} %")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

joblib.dump(scaler, 'models/ann_scaler.pkl')
model.save('models/ann_model.h5')
