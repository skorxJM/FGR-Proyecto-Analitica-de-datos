import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import skfuzzy as fuzz
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
import warnings


def load_data():
    """Carga y prepara el dataset"""
    df = pd.read_excel("data/FGR_dataset.xlsx")
    print("\n[INFO] Columnas originales:", df.columns.tolist())
    print("[INFO] Distribución de clases original:\n", df['C31'].value_counts())
    return df


def preprocess_data(df):
    """Realiza el preprocesamiento completo de los datos"""
    X = df.drop(columns='C31')
    y = df['C31']
    
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Balanceo con SMOTE (solo en entrenamiento)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("\n[INFO] Distribución después de SMOTE:\n", y_resampled.value_counts())
    
    return X_resampled, y_resampled, X_test, y_test

#Selección y transformación de características
def feature_engineering(X_train, y_train, X_test):
    """Selección de características y reducción de dimensionalidad"""
    # Selección de características
    selector = SelectKBest(f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # PCA (Opcional - descomentar si se necesita)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"[INFO] Número de componentes PCA: {pca.n_components_}")
    return X_train_pca.T, X_test_pca.T, scaler, selector, pca

    return X_train_scaled.T, X_test_scaled.T, scaler, selector, None

#Modelo FCM
def train_fcm(X_train_T, n_clusters=2, m=2.0):
    """Entrena el modelo FCM con los parámetros dados"""
    try:
        print("\n[INFO] Entrenando modelo FCM...")
        print(f"[INFO] Dimensiones de entrada: {X_train_T.shape} (deben ser [características, muestras])")
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_train_T,
            c=n_clusters,
            m=m,
            error=0.001,
            maxiter=1000,
            init=None
        )
        
        print(f"[SUCCESS] Modelo entrenado con FPC: {fpc:.4f}")
        return cntr, u, fpc
    except Exception as e:
        print(f"[ERROR] Fallo en entrenamiento FCM: {str(e)}")
        return None, None, None


#Evaluación
def evaluate_model(cntr, u, X_T, y_true, dataset_name="entrenamiento"):
    """Evalúa el modelo FCM"""
    if cntr is None:
        return
    
    cluster_labels = np.argmax(u, axis=0)

    # Mapeo de clusters a etiquetas reales
    def map_clusters_to_labels(clusters, true_labels):
        label_mapping = {}
        labels = np.zeros_like(clusters)
        for i in range(np.max(clusters)+1):
            mask = (clusters == i)
            if np.any(mask):
                
                label_mapping[i] = mode(true_labels[mask], keepdims=True)[0][0]
                labels[mask] = label_mapping[i]
        return labels, label_mapping

    mapped_labels, cluster_mapping = map_clusters_to_labels(cluster_labels, y_true.to_numpy())
    accuracy = accuracy_score(y_true, mapped_labels)

    print(f"\n[RESULTADOS] Evaluación en {dataset_name}:")
    print(f"Precisión: {accuracy*100:.2f}%")
    print(f"Mapeo de clusters: {cluster_mapping}")

    # Suprimir warnings de métricas indefinidas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        print("Reporte de clasificación:")
        print(classification_report(y_true, mapped_labels))

    # Verifica si hay predicciones en ambas clases
    unique_preds = np.unique(mapped_labels)
    print(f"[INFO] Clases predichas: {unique_preds}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, mapped_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión ({dataset_name})')
    plt.tight_layout()
    plt.show()
    plt.close()  

    return accuracy, cluster_mapping  

# 6. Pipeline completo
def main():
    df = load_data()
    
    X_resampled, y_resampled, X_test, y_test = preprocess_data(df)
    
    X_train_T, X_test_T, scaler, selector, pca = feature_engineering(X_resampled, y_resampled, X_test)
    
    # Entrenamiento FCM
    cntr, u, fpc = train_fcm(X_train_T, n_clusters=2, m=2.0)
    
    if cntr is not None:
        # Evaluación en entrenamiento
        train_acc, cluster_mapping = evaluate_model(cntr, u, X_train_T, y_resampled, "entrenamiento")
        
        # Evaluación en prueba
        u_test = fuzz.cluster.cmeans_predict(
            X_test_T,
            cntr,
            m=2.0,
            error=0.001,
            maxiter=1000
        )[1]
        test_acc = evaluate_model(cntr, u_test, X_test_T, y_test, "prueba")
        
        # Guardar modelo
        model_data = {
            'centroids': cntr,
            'm': 2.0,
            'scaler': scaler,
            'selector': selector,
            'pca': pca,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'fpc': fpc,
            'cluster_mapping': cluster_mapping
        }
        joblib.dump(model_data, 'models/fcm_model.pkl')
        print("\n[SUCCESS] Modelo guardado exitosamente en 'models/fcm_model.pkl'")

if __name__ == "__main__":
    main()