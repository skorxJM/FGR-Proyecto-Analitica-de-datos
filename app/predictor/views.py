from django.shortcuts import render
from .forms import IndividualPredictionForm
from .forms import BatchPredictionForm
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os
import skfuzzy as fuzz
import pandas as pd
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATHS = {
    "logistic": os.path.join(BASE_DIR, "models", "logistic_model.pkl"),
    "logistic_scaler": os.path.join(BASE_DIR, "models", "logistic_scaler.pkl"),
    "svm": os.path.join(BASE_DIR, "models", "svm_model.pkl"),
    "svm_scaler": os.path.join(BASE_DIR, "models", "svm_scaler.pkl"),
    "ann": os.path.join(BASE_DIR, "models", "ann_model.h5"),
    "ann_scaler": os.path.join(BASE_DIR, "models", "ann_scaler.pkl"),
    "fcm": os.path.join(
        BASE_DIR, "models", "fcm_model.pkl"
    ),  # Contiene scaler internamente
}


def home(request):
    return render(request, "predictor/home.html")


# ... (todo lo anterior se mantiene igual) ...


def preprocess_fcm_input(X, fcm_model):
    """Preprocesamiento completo igual que en entrenamiento"""
    try:
        # 1. Selección de características
        if 'selector' not in fcm_model:
            raise ValueError("El modelo FCM no contiene el selector de características")
        X_selected = fcm_model['selector'].transform(X)
        
        # 2. Escalado
        if 'scaler' not in fcm_model:
            raise ValueError("El modelo FCM no contiene el scaler")
        X_scaled = fcm_model['scaler'].transform(X_selected)
        
        # 3. PCA si existe
        if 'pca' in fcm_model and fcm_model['pca'] is not None:
            X_processed = fcm_model['pca'].transform(X_scaled)
        else:
            X_processed = X_scaled
        
        return X_processed  # No transponer todavía
    
    except Exception as e:
        raise ValueError(f"Error en preprocesamiento FCM: {str(e)}")



def predict_individual(request):
    prediction = None
    selected_model = None

    if request.method == "POST":
        form = IndividualPredictionForm(request.POST)
        if form.is_valid():
            selected_model = form.cleaned_data["model"]
            input_data = [form.cleaned_data[f"C{i}"] for i in range(1, 31)]
            X = np.array(input_data).reshape(1, -1)

            if selected_model == "svm":
                model = joblib.load(MODEL_PATHS["svm"])
                scaler = joblib.load(MODEL_PATHS["svm_scaler"])
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]

            elif selected_model == "logistic":
                model = joblib.load(MODEL_PATHS["logistic"])
                scaler = joblib.load(MODEL_PATHS["logistic_scaler"])
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]

            elif selected_model == "ann":
                model = load_model(MODEL_PATHS["ann"])
                scaler = joblib.load(MODEL_PATHS["ann_scaler"])
                X_scaled = scaler.transform(X)
                prediction = (model.predict(X_scaled)[0][0] > 0.5).astype(int)

            elif selected_model == "fcm":
                fcm_model = joblib.load(MODEL_PATHS["fcm"])
                centroids = fcm_model["centroids"]
                m = fcm_model["m"]
                error_fcm = 0.001
                maxiter = 1000

                X_df = pd.DataFrame(X, columns=[f"C{i}" for i in range(1, 31)])
                X_processed = preprocess_fcm_input(X_df, fcm_model).T

                u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                    X_processed, centroids, m, error=error_fcm, maxiter=maxiter
                )
                cluster = np.argmax(u, axis=0)[0]
                prediction = cluster

    else:
        form = IndividualPredictionForm()

    return render(
        request,
        "predictor/predict_individual.html",
        {
            "form": form,
            "prediction": prediction,
            "selected_model": selected_model,
        },
    )


def predict_batch(request):
    accuracy = None
    plot_url = None
    error = None
    predictions_table = None

    if request.method == "POST":
        form = BatchPredictionForm(request.POST, request.FILES)
        if form.is_valid():
            selected_model = form.cleaned_data["model"]
            file = request.FILES["file"]

            if not file.name.endswith(".xlsx"):
                error = "El archivo debe tener extensión .xlsx"
            else:
                try:
                    df = pd.read_excel(file)
                    expected_columns = [f"C{i}" for i in range(1, 31)]

                    if not all(col in df.columns for col in expected_columns):
                        error = "El archivo debe contener al menos las columnas C1 a C30"
                    else:
                        X = df[expected_columns].to_numpy()
                        y_pred_clusters = None  # Inicializamos la variable

                        if selected_model == "svm":
                            model = joblib.load(MODEL_PATHS["svm"])
                            scaler = joblib.load(MODEL_PATHS["svm_scaler"])
                            X_scaled = scaler.transform(X)
                            y_pred = model.predict(X_scaled)

                        elif selected_model == "logistic":
                            model = joblib.load(MODEL_PATHS["logistic"])
                            scaler = joblib.load(MODEL_PATHS["logistic_scaler"])
                            X_scaled = scaler.transform(X)
                            y_pred = model.predict(X_scaled)

                        elif selected_model == "ann":
                            model = load_model(MODEL_PATHS["ann"])
                            scaler = joblib.load(MODEL_PATHS["ann_scaler"])
                            X_scaled = scaler.transform(X)
                            y_pred = (model.predict(X_scaled) > 0.5).astype(int).flatten()

                        elif selected_model == "fcm":
                            fcm_model = joblib.load(MODEL_PATHS["fcm"])
                            X_processed = preprocess_fcm_input(pd.DataFrame(X, columns=expected_columns), fcm_model)
                            
                            u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                                X_processed.T,
                                fcm_model["centroids"],
                                fcm_model["m"],
                                error=0.001,
                                maxiter=1000
                            )
                            y_pred_clusters = np.argmax(u, axis=0)
                            y_pred = np.array([fcm_model['cluster_mapping'].get(c, c) for c in y_pred_clusters])

                        # Preparar tabla de resultados
                        predictions_table = []
                        for i in range(len(y_pred)):
                            pred_entry = {
                                "index": i + 1,
                                "prediction": "Positivo (FGR)" if y_pred[i] == 1 else "Negativo (Normal)",
                            }
                            
                            # Solo agregar info del cluster si es FCM y la variable existe
                            if selected_model == "fcm" and y_pred_clusters is not None:
                                pred_entry["cluster"] = int(y_pred_clusters[i])
                            
                            predictions_table.append(pred_entry)

                        if "C31" in df.columns:
                            y_true = df["C31"]
                            cm = confusion_matrix(y_true, y_pred)
                            accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)

                            plt.figure(figsize=(5, 4))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                            plt.xlabel("Predicción")
                            plt.ylabel("Valor Real")
                            plt.title("Matriz de Confusión")
                            buf = BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            image_png = buf.getvalue()
                            buf.close()
                            plot_url = base64.b64encode(image_png).decode("utf-8")
                            plt.close()

                except Exception as e:
                    error = f"Error al procesar el archivo: {str(e)}"
                    print(f"[ERROR] {error}")

    else:
        form = BatchPredictionForm()

    return render(
        request,
        "predictor/predict_batch.html",
        {
            "form": form,
            "accuracy": accuracy,
            "plot_url": plot_url,
            "predictions_table": predictions_table,
            "error": error,
        },
    )
