import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

st.set_page_config(page_title="Red Neuronal MLP", layout="wide")

# ---------- CARGA DE DATOS ----------
def load_data(problem_type):
    if problem_type == "Regresión":
        df = pd.read_csv("diamantes.csv").dropna()
        X = df[["carat", "depth", "table"]]
        y = df["price"]
    else:
        df = pd.read_csv("atletas.csv").dropna()
        df['Atleta'] = df['Atleta'].map({'Fondista': 1, 'Velocista': 0})
        X = df[["Edad", "Peso", "Volumen_O2_max"]]
        y = df["Atleta"]
    return X, y

# ---------- SIDEBAR DE CONFIGURACIÓN ----------
st.sidebar.title("Configuración del Modelo")
problem_type = st.sidebar.selectbox("Tipo de problema", ["Regresión", "Clasificación"])
optimizer_option = st.sidebar.selectbox("Optimizador", ["adam", "sgd"])
metric_option = st.sidebar.selectbox("Métrica", ["mae", "mse", "accuracy"])
epochs = st.sidebar.slider("Épocas", 10, 500, 100, step=10)
st.sidebar.markdown("Valores para predicción")

# ---------- INPUTS PARA PREDICCIÓN ----------
predict_inputs = {}
for col in ["Edad", "Peso", "Volumen_O2_max"]:
    predict_inputs[col] = st.sidebar.slider(col, 0.0, 100.0, 50.0)

# ---------- CONFIGURACIÓN DEL MODELO ----------
activation_output = "sigmoid" if problem_type == "Clasificación" else "linear"
loss_function = "binary_crossentropy" if problem_type == "Clasificación" else "mse"
activation_hidden = "tanh" if problem_type == "Clasificación" else "relu"

# ---------- CARGA Y PREPARACIÓN DE DATOS ----------
X, y = load_data(problem_type)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------- CREACIÓN DEL MODELO ----------
model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation=activation_hidden))
model.add(Dense(8, activation=activation_hidden))
model.add(Dense(1, activation=activation_output))
optimizer = Adam() if optimizer_option == "adam" else SGD()
model.compile(optimizer=optimizer, loss=loss_function, metrics=[metric_option])

# ---------- ENTRENAMIENTO ----------
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)

# ---------- INTERFAZ PRINCIPAL ----------
st.title("Red Neuronal Multicapa")
st.write(f"## Problema seleccionado: {problem_type}")

# ---------- REPRESENTACIÓN DE LA ARQUITECTURA ----------
st.subheader("Arquitectura de la red (Horizontal)")
dot = graphviz.Digraph(format='png')
dot.attr(rankdir='LR', splines='line')

with dot.subgraph() as s:
    s.attr(rank='same')
    for i, feature in enumerate(["Edad", "Peso", "VO2_max"]):
        s.node(f'I{i}', label=feature, shape='ellipse', style='filled', color='green')

with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(16):
        s.node(f'H1_{i}', label=f"H1-{i+1}\n{activation_hidden}", shape='ellipse', style='filled', color='green')

with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(8):
        s.node(f'H2_{i}', label=f"H2-{i+1}\n{activation_hidden}", shape='ellipse', style='filled', color='green')

dot.node("O", f"Salida\n{activation_output}", shape="ellipse", style="filled", color="orangered")

for i in range(3):
    for j in range(16):
        dot.edge(f'I{i}', f'H1_{j}')
for i in range(16):
    for j in range(8):
        dot.edge(f'H1_{i}', f'H2_{j}')
for i in range(8):
    dot.edge(f'H2_{i}', 'O')

st.graphviz_chart(dot)

# ---------- GRÁFICO DE PÉRDIDA ----------
st.subheader("Evolución de la pérdida durante el entrenamiento")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.legend()
st.pyplot(fig)

# ---------- EVALUACIÓN DEL MODELO ----------
st.subheader("Evaluación del Modelo")
y_pred = model.predict(X_test).flatten()
if problem_type == "Clasificación":
    y_pred_labels = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_labels)
    report = classification_report(y_test, y_pred_labels, output_dict=False)
    st.text(report)
else:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

# ---------- PREDICCIÓN PERSONALIZADA ----------
st.subheader("Haz tu propia predicción")
input_array = np.array(list(predict_inputs.values())).reshape(1, -1)
input_scaled = scaler.transform(input_array)
pred = model.predict(input_scaled)[0][0]

if problem_type == "Clasificación":
    resultado = "Fondista" if pred >= 0.5 else "Velocista"
    st.write(f"**Resultado:** {resultado} ({pred:.2f})")
else:
    st.write(f"**Precio estimado del diamante:** {round(pred, 2)} USD")
