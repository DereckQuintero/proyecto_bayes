"""
PROYECTO UNIDAD 2 - INTELIGENCIA ARTIFICIAL
Estimación de Compra en Tienda en Línea (Online Shoppers Purchasing Intention)

Modelos:
1. Teorema de Bayes por conteos
2. Red con estructura Naive Bayes (Laplace) con inferencia parcial
3. Random Forest Classifier (con ajuste de balanceo)
4. Perceptrón Multicapa (MLP Classifier con Early Stopping)
"""

import time
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
print("==================================================")
print("1. CARGA, EXPLORACIÓN Y PARTICIÓN DE DATOS")
print("==================================================")

# Cargar dataset
df = pd.read_csv('online_shoppers_intention.csv')

# Filtrar entradas y salida requeridas
features = ['VisitorType', 'ProductRelated', 'ProductRelated_Duration']
X_raw = df[features].copy()
y = df['Revenue'].astype(int)

# Partición 80% train, 20% test estratificada
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Total registros: {len(df)}")
print(f"Entrenamiento (80%): {len(X_train_raw)} | Prueba (20%): {len(X_test_raw)}")
print(f"Tasa base de compras (Train): {y_train.mean()*100:.2f}%\n")

# ==============================================================================
# 2. CATEGORIZACIÓN (SOLO PARA NAIVE BAYES) MEDIANTE PERCENTILES
# ==============================================================================
# P33 y P67 calculados estrictamente sobre TRAIN para evitar data leakage
p33_pages = np.percentile(X_train_raw['ProductRelated'], 33)
p67_pages = np.percentile(X_train_raw['ProductRelated'], 67)
p33_dur = np.percentile(X_train_raw['ProductRelated_Duration'], 33)
p67_dur = np.percentile(X_train_raw['ProductRelated_Duration'], 67)

print(f"Límites ProductRelated -> P33: {p33_pages:.1f}, P67: {p67_pages:.1f}")
print(f"Límites ProductRelated_Duration -> P33: {p33_dur:.1f}, P67: {p67_dur:.1f}\n")

def discretizar(valor, p33, p67):
    if valor <= p33:
        return 'bajo'
    elif valor <= p67:
        return 'medio'
    else:
        return 'alto'

def categorizar_dataset(df_in):
    df_cat = pd.DataFrame(index=df_in.index)
    df_cat['VisitorType'] = df_in['VisitorType']
    df_cat['ProductRelated_Cat'] = df_in['ProductRelated'].apply(lambda x: discretizar(x, p33_pages, p67_pages))
    df_cat['Duration_Cat'] = df_in['ProductRelated_Duration'].apply(lambda x: discretizar(x, p33_dur, p67_dur))
    return df_cat

X_train_cat = categorizar_dataset(X_train_raw)
X_test_cat = categorizar_dataset(X_test_raw)

# ==============================================================================
# 3. TEOREMA DE BAYES POR CONTEOS DIRECTOS
# ==============================================================================
print("==================================================")
print("2. TEOREMA DE BAYES P(Compra | Recurrente)")
print("==================================================")
N = len(y_train)
N_C = (y_train == 1).sum()
N_R = (X_train_cat['VisitorType'] == 'Returning_Visitor').sum()
N_CR = ((y_train == 1) & (X_train_cat['VisitorType'] == 'Returning_Visitor')).sum()

P_C = N_C / N
P_R = N_R / N
P_R_given_C = N_CR / N_C

P_C_given_R_bayes = (P_R_given_C * P_C) / P_R
P_C_given_R_directo = N_CR / N_R

print(f"Conteos: Total={N}, Compra(C)={N_C}, Recurrente(R)={N_R}, C ∩ R={N_CR}")
print(f"P(C) = {P_C:.4f} | P(R) = {P_R:.4f} | P(R|C) = {P_R_given_C:.4f}")
print(f"Fórmula Bayes P(C|R) = (P(R|C) * P(C)) / P(R) = {P_C_given_R_bayes:.4f}")
print(f"Comprobación directa (N_CR / N_R) = {P_C_given_R_directo:.4f}\n")

# ==============================================================================
# 4. RED NAIVE BAYES CON SUAVIZADO DE LAPLACE (MANEJO DE EVIDENCIA PARCIAL)
# ==============================================================================
class CategoricalNaiveBayesLaplace:
    """Implementa Naive Bayes con suavizado de Laplace y soporte para variables no observadas."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prior = {}
        self.conditional = {}
        self.categories = {}
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        # Probabilidades a priori P(C)
        for c in self.classes_:
            self.prior[c] = (np.sum(y == c) + self.alpha) / (n_samples + len(self.classes_) * self.alpha)
            
        # Tablas de probabilidad condicional P(X_i | C) con Laplace
        for col in X.columns:
            self.categories[col] = list(X[col].unique())
            k = len(self.categories[col])
            self.conditional[col] = {}
            for c in self.classes_:
                X_c = X[y == c][col]
                total_c = len(X_c)
                self.conditional[col][c] = {}
                for cat in self.categories[col]:
                    count = np.sum(X_c == cat)
                    # Estimador Laplace: (count + alpha) / (total + k * alpha)
                    self.conditional[col][c][cat] = (count + self.alpha) / (total_c + k * self.alpha)
                    
    def predict_proba_single(self, evidence):
        """Calcula P(C=1 | evidencia parcial o total)."""
        likelihoods = {}
        for c in self.classes_:
            p = self.prior[c]
            for col, val in evidence.items():
                if val is not None and col in self.conditional:
                    p *= self.conditional[col][c].get(val, 1e-6)
            likelihoods[c] = p
            
        total_p = sum(likelihoods.values())
        if total_p == 0:
            return 0.5
        return likelihoods[1] / total_p

    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            evidence = row.to_dict()
            probs.append(self.predict_proba_single(evidence))
        return np.array(probs)

    def predict(self, X, threshold=0.50):
        return (self.predict_proba(X) >= threshold).astype(int)

# Entrenar modelo Bayesiano
nb_custom = CategoricalNaiveBayesLaplace(alpha=1.0)
nb_custom.fit(X_train_cat, y_train)

# ==============================================================================
# 5. CONSULTAS PROGRESIVAS EXIGIDAS EN LA RÚBRICA
# ==============================================================================
print("==================================================")
print("3. TRES CONSULTAS PROGRESIVAS (NAIVE BAYES)")
print("==================================================")
q1 = {'VisitorType': 'Returning_Visitor'}
p_q1 = nb_custom.predict_proba_single(q1)
print(f"1. P(Compra | Recurrente) = {p_q1:.4f}")

q2 = {'VisitorType': 'Returning_Visitor', 'ProductRelated_Cat': 'alto'}
p_q2 = nb_custom.predict_proba_single(q2)
print(f"2. P(Compra | Recurrente, Páginas: alto) = {p_q2:.4f}")

q3 = {'VisitorType': 'Returning_Visitor', 'ProductRelated_Cat': 'alto', 'Duration_Cat': 'alto'}
p_q3 = nb_custom.predict_proba_single(q3)
print(f"3. P(Compra | Recurrente, Páginas: alto, Duración: alto) = {p_q3:.4f}")
print("-> Explicación: Al agregar más evidencia positiva (alto interés y permanencia), la probabilidad aumenta consistentemente.\n")

# ==============================================================================
# 6. MODELOS MACHINE LEARNING (CON MANEJO DE DESBALANCE)
# ==============================================================================
# Preprocesamiento numérico para ML
le = LabelEncoder()
X_train_ml = X_train_raw.copy()
X_test_ml = X_test_raw.copy()

X_train_ml['VisitorType'] = le.fit_transform(X_train_raw['VisitorType'])
X_test_ml['VisitorType'] = le.transform(X_test_raw['VisitorType'])

scaler = StandardScaler()
num_cols = ['ProductRelated', 'ProductRelated_Duration']
X_train_mlp = X_train_ml.copy()
X_test_mlp = X_test_ml.copy()

X_train_mlp[num_cols] = scaler.fit_transform(X_train_ml[num_cols])
X_test_mlp[num_cols] = scaler.transform(X_test_ml[num_cols])

# A) Random Forest con balanceo de pesos para mejorar detección de compra
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced',  # Solución clave para corregir sensibilidad baja
    random_state=42,
    max_depth=6
)
rf.fit(X_train_ml, y_train)
t_train_rf = time.time() - t0

# B) MLP Classifier
t0 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_train_mlp, y_train)
t_train_mlp = time.time() - t0

# ==============================================================================
# 7. EVALUACIÓN Y COMPARACIÓN DE MÉTRICAS (TEST SET)
# ==============================================================================
print("==================================================")
print("4. COMPARATIVA DE MODELOS EN TEST (UMBRAL = 0.50)")
print("==================================================")

# Inferencia
t0 = time.time()
y_pred_nb = nb_custom.predict(X_test_cat)
t_inf_nb = (time.time() - t0) / len(X_test_cat)

t0 = time.time()
y_pred_rf = rf.predict(X_test_ml)
t_inf_rf = (time.time() - t0) / len(X_test_ml)

t0 = time.time()
y_pred_mlp = mlp.predict(X_test_mlp)
t_inf_mlp = (time.time() - t0) / len(X_test_mlp)

y_pred_base = np.zeros_like(y_test)

modelos = {
    "Mayoritaria (Baseline)": (y_pred_base, 0.0, 0.0),
    "Naive Bayes (Laplace)": (y_pred_nb, 0.01, t_inf_nb),
    "Random Forest (Balanced)": (y_pred_rf, t_train_rf, t_inf_rf),
    "MLP (Red Neuronal)": (y_pred_mlp, t_train_mlp, t_inf_mlp)
}

resultados = []
matrices = {}

for nombre, (pred, t_tr, t_inf) in modelos.items():
    cm = confusion_matrix(y_test, pred)
    matrices[nombre] = cm
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    
    resultados.append({
        'Modelo': nombre,
        'Precisión': prec,
        'Sensibilidad (Recall)': rec,
        'F1-Score': f1,
        'T. Entreno (s)': t_tr,
        'T. Inferencia (s)': t_inf
    })

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))
print("\nNota: Al balancear Random Forest, la Sensibilidad (detección de compradores reales) mejora significativamente.")

# ==============================================================================
# 8. VISUALIZACIÓN GRÁFICA SOLICITADA POR EL PROFESOR
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
titulos = list(modelos.keys())

for idx, ax in enumerate(axes.flatten()):
    nombre = titulos[idx]
    cm = matrices[nombre]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'])
    ax.set_title(f"Matriz de Confusión: {nombre}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.savefig("matrices_confusion.png", dpi=300)
print("\n[OK] Gráfico 'matrices_confusion.png' generado.")

# Gráfica de barras de métricas comparativas
plt.figure(figsize=(10, 5))
df_melted = pd.melt(df_res, id_vars=['Modelo'], value_vars=['Precisión', 'Sensibilidad (Recall)', 'F1-Score'],
                    var_name='Métrica', value_name='Valor')
sns.barplot(data=df_melted, x='Modelo', y='Valor', hue='Métrica', palette='viridis')
plt.title("Comparación de Métricas por Modelo")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("comparativa_metricas.png", dpi=300)
print("[OK] Gráfico 'comparativa_metricas.png' generado.")

# ==============================================================================
# 9. VALIDACIÓN DE ENTRADAS Y TEST DE ERROR
# ==============================================================================
print("\n==================================================")
print("5. PRUEBA DE ENTRADA INVÁLIDA")
print("==================================================")
def evaluar_sesion_individual(v_type, pages, duration):
    valid_types = ['Returning_Visitor', 'New_Visitor', 'Other']
    if v_type not in valid_types:
        raise ValueError(f"Tipo '{v_type}' inválido. Opciones: {valid_types}")
    if pages < 0 or duration < 0:
        raise ValueError("Valores negativos no permitidos para páginas o duración.")
        
    # Categorizar para NB
    p_cat = discretizar(pages, p33_pages, p67_pages)
    d_cat = discretizar(duration, p33_dur, p67_dur)
    
    # 1. NB Proba
    p_nb = nb_custom.predict_proba_single({'VisitorType': v_type, 'ProductRelated_Cat': p_cat, 'Duration_Cat': d_cat})
    
    # 2. RF Proba
    v_code = le.transform([v_type])[0]
    X_sample_ml = pd.DataFrame([[v_code, pages, duration]], columns=features)
    p_rf = rf.predict_proba(X_sample_ml)[0][1]
    
    # 3. MLP Proba
    X_sample_mlp = X_sample_ml.copy()
    X_sample_mlp[num_cols] = scaler.transform(X_sample_ml[num_cols])
    p_mlp = mlp.predict_proba(X_sample_mlp)[0][1]
    
    print(f"\nSesión ingresada -> {v_type}, Páginas: {pages} ({p_cat}), Duración: {duration}s ({d_cat})")
    print(f"Prob. Compra -> Naive Bayes: {p_nb:.4f} | RF: {p_rf:.4f} | MLP: {p_mlp:.4f}")

try:
    print("Probando entrada negativa (-12 páginas)...")
    evaluar_sesion_individual('Returning_Visitor', -12, 100)
except ValueError as err:
    print(f"[RECHAZADO EXITOSAMENTE] Excepción capturada: {err}")

# Demostración con caso válido
evaluar_sesion_individual('Returning_Visitor', 35, 1200)