import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ==========================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ==========================================
print("==================================================")
print("1. EXPLORACIÓN Y PARTICIÓN DE DATOS")
print("==================================================")
df = pd.read_csv('online_shoppers_intention.csv')

X_raw = df[['VisitorType', 'ProductRelated', 'ProductRelated_Duration']].copy()
y = df['Revenue'].astype(int)

# División 80% entrenamiento, 20% prueba
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Total registros: {len(df)}")
print(f"Sesiones Entrenamiento (80%): {len(X_train_raw)}")
print(f"Sesiones Prueba (20%): {len(X_test_raw)}")
print(f"Proporción de compras en Train: {y_train.mean():.4f}\n")

# ==========================================
# 2. CATEGORIZACIÓN PARA NAIVE BAYES (PERCENTILES)
# ==========================================
p33_pages = np.percentile(X_train_raw['ProductRelated'], 33)
p67_pages = np.percentile(X_train_raw['ProductRelated'], 67)
p33_dur = np.percentile(X_train_raw['ProductRelated_Duration'], 33)
p67_dur = np.percentile(X_train_raw['ProductRelated_Duration'], 67)

print(f"Cortes ProductRelated (P33, P67): {p33_pages:.2f}, {p67_pages:.2f}")
print(f"Cortes ProductRelated_Duration (P33, P67): {p33_dur:.2f}, {p67_dur:.2f}\n")

def discretize(val, p33, p67):
    if val <= p33:
        return 'bajo'
    elif val <= p67:
        return 'medio'
    else:
        return 'alto'

def prepare_cat_data(df_raw):
    df_cat = pd.DataFrame()
    df_cat['VisitorType'] = df_raw['VisitorType']
    df_cat['Pages_Cat'] = df_raw['ProductRelated'].apply(lambda x: discretize(x, p33_pages, p67_pages))
    df_cat['Duration_Cat'] = df_raw['ProductRelated_Duration'].apply(lambda x: discretize(x, p33_dur, p67_dur))
    return df_cat

X_train_cat = prepare_cat_data(X_train_raw)
X_test_cat = prepare_cat_data(X_test_raw)

# ==========================================
# 3. TEOREMA DE BAYES POR CONTEOS
# ==========================================
print("==================================================")
print("2. CÁLCULO DE BAYES P(Compra | Recurrente)")
print("==================================================")
N_total = len(y_train)
N_C = (y_train == 1).sum()
N_R = (X_train_cat['VisitorType'] == 'Returning_Visitor').sum()
N_C_and_R = ((y_train == 1) & (X_train_cat['VisitorType'] == 'Returning_Visitor')).sum()

P_C = N_C / N_total
P_R = N_R / N_total
P_R_given_C = N_C_and_R / N_C

P_C_given_R_bayes = (P_R_given_C * P_C) / P_R
P_C_given_R_direct = N_C_and_R / N_R

print(f"Conteos -> Total: {N_total}, Compras (C): {N_C}, Recurrentes (R): {N_R}, Ambos (C ∩ R): {N_C_and_R}")
print(f"P(C) = {P_C:.4f} | P(R) = {P_R:.4f} | P(R|C) = {P_R_given_C:.4f}")
print(f"Resultado Teorema de Bayes P(C|R): {P_C_given_R_bayes:.4f}")
print(f"Comprobación Conteo Directo (C ∩ R)/R: {P_C_given_R_direct:.4f}")
print("Interpretación: Aproximadamente el 14.8% de las visitas recurrentes terminan en compra.\n")

# ==========================================
# 4. MODELOS DE MACHINE LEARNING (RF Y MLP)
# ==========================================
le = LabelEncoder()
X_train_num = X_train_raw.copy()
X_test_num = X_test_raw.copy()

X_train_num['VisitorType'] = le.fit_transform(X_train_raw['VisitorType'])
X_test_num['VisitorType'] = le.transform(X_test_raw['VisitorType'])

scaler = StandardScaler()
X_train_scaled = X_train_num.copy()
X_test_scaled = X_test_num.copy()
num_cols = ['ProductRelated', 'ProductRelated_Duration']
X_train_scaled[num_cols] = scaler.fit_transform(X_train_num[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test_num[num_cols])

# Entrenar Random Forest
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_num, y_train)
t_train_rf = time.time() - t0

# Entrenar MLP
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
mlp.fit(X_train_scaled, y_train)
t_train_mlp = time.time() - t0

# ==========================================
# 5. EVALUACIÓN Y COMPARACIÓN DE METRICAS
# ==========================================
print("==================================================")
print("3. EVALUACIÓN DE MODELOS EN TEST (UMBRAL = 0.50)")
print("==================================================")

y_pred_base = np.zeros_like(y_test)
y_pred_rf = rf.predict(X_test_num)
y_pred_mlp = mlp.predict(X_test_scaled)

def print_metrics(name, y_true, y_pred, t_train):
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"--- Modelo: {name} ---")
    print(f"Matriz de Confusión:\n{cm}")
    print(f"Precisión: {prec:.4f} | Sensibilidad: {rec:.4f} | F1-Score: {f1:.4f} | T. Entreno: {t_train:.4f}s\n")

print_metrics("Referencia (Clase Mayoritaria)", y_test, y_pred_base, 0.0)
print_metrics("Random Forest", y_test, y_pred_rf, t_train_rf)
print_metrics("Perceptrón Multicapa (MLP)", y_test, y_pred_mlp, t_train_mlp)

# ==========================================
# 6. VALIDACIÓN DE ENTRADAS (PRUEBA DE ERROR)
# ==========================================
print("==================================================")
print("4. PRUEBA DE ENTRADA INVÁLIDA")
print("==================================================")
def validar_entrada(visitor_type, pages, duration):
    valid_types = ['Returning_Visitor', 'New_Visitor', 'Other']
    if visitor_type not in valid_types:
        raise ValueError(f"Tipo de visitante '{visitor_type}' no válido. Opciones: {valid_types}")
    if pages < 0 or duration < 0:
        raise ValueError("Las páginas y la duración no pueden ser valores negativos.")
    return True

try:
    print("Probando entrada con páginas negativas (-5)...")
    validar_entrada('Returning_Visitor', -5, 100)
except ValueError as e:
    print(f"[RECHAZADO CORRECTAMENTE] Error capturado: {e}")