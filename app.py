"""Perceptron didáctico para aprobar una experiencia curricular.

Implementa manualmente la regla de aprendizaje explicada en clase.
"""

from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


LEARNING_RATE = 0.5
INITIAL_WEIGHTS = (0.5, 0.4, 0.5)
INITIAL_BIAS = 1.5
PASSING_GRADE = 14.0
MAX_EPOCHS = 1000


def desired_output(x1: int, x2: int, x3: int) -> int:
    """Etiqueta supervisada: aprueba con al menos dos unidades aprobadas."""
    return int(x1 + x2 + x3 >= 2)


def make_patterns() -> list[dict]:
    """Construye las ocho combinaciones binarias en el orden solicitado."""
    return [
        {"Patrón": i, "x1": x1, "x2": x2, "x3": x3, "yd": desired_output(x1, x2, x3)}
        for i, (x1, x2, x3) in enumerate(product((0, 1), repeat=3), start=1)
    ]


def activation(net: float) -> int:
    """Función escalón usada en las diapositivas del profesor."""
    return int(net >= 0)


def train_perceptron() -> tuple[list[dict], list[dict], tuple[float, ...], float, int, bool]:
    """Entrena secuencialmente; aplica las actualizaciones tras cada patrón."""
    weights = list(INITIAL_WEIGHTS)
    bias = INITIAL_BIAS
    patterns = make_patterns()
    history: list[dict] = []
    epoch_summaries: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        errors = 0
        for pattern in patterns:
            x = (pattern["x1"], pattern["x2"], pattern["x3"])
            weights_before = tuple(weights)
            bias_before = bias
            products = tuple(x[i] * weights[i] for i in range(3))
            net = sum(products) + bias
            y = activation(net)
            error = pattern["yd"] - y

            weights_after = tuple(
                weights[i] + LEARNING_RATE * error * x[i] for i in range(3)
            )
            bias_after = bias + LEARNING_RATE * error * 1
            updated = error != 0

            if updated:
                errors += 1
            weights = list(weights_after)
            bias = bias_after

            history.append(
                {
                    "Época": epoch,
                    "Patrón": pattern["Patrón"],
                    "x1": x[0],
                    "x2": x[1],
                    "x3": x[2],
                    "yd": pattern["yd"],
                    "w1 antes": weights_before[0],
                    "w2 antes": weights_before[1],
                    "w3 antes": weights_before[2],
                    "b antes": bias_before,
                    "x1·w1": products[0],
                    "x2·w2": products[1],
                    "x3·w3": products[2],
                    "NET": net,
                    "y": y,
                    "error": error,
                    "w1 después": weights_after[0],
                    "w2 después": weights_after[1],
                    "w3 después": weights_after[2],
                    "b después": bias_after,
                    "Actualización": "Sí" if updated else "No (error = 0)",
                    "Operaciones de actualización": (
                        f"w1={weights_before[0]:g}+0.5×{error}×{x[0]}={weights_after[0]:g}; "
                        f"w2={weights_before[1]:g}+0.5×{error}×{x[1]}={weights_after[1]:g}; "
                        f"w3={weights_before[2]:g}+0.5×{error}×{x[2]}={weights_after[2]:g}; "
                        f"b={bias_before:g}+0.5×{error}×1={bias_after:g}"
                    ),
                }
            )

        epoch_summaries.append(
            {
                "Época": epoch,
                "Errores de clasificación": errors,
                "w1 final": weights[0],
                "w2 final": weights[1],
                "w3 final": weights[2],
                "b final": bias,
                "Convergió": errors == 0,
            }
        )
        if errors == 0:
            return history, epoch_summaries, tuple(weights), bias, epoch, True

    return history, epoch_summaries, tuple(weights), bias, MAX_EPOCHS, False


def format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def decision_figure(patterns: list[dict], weights: tuple[float, ...], bias: float):
    """Muestra los ocho casos y, si cruza el cubo, el plano NET=0."""
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = {0: "#d95f59", 1: "#168a72"}
    for label in (0, 1):
        rows = [p for p in patterns if p["yd"] == label]
        ax.scatter(
            [p["x1"] for p in rows],
            [p["x2"] for p in rows],
            [p["x3"] for p in rows],
            color=colors[label],
            s=90,
            label=f"Clase {label}: {'aprueba' if label else 'desaprueba'}",
            edgecolors="white",
            linewidths=0.8,
        )
    if abs(weights[2]) > 1e-12:
        import numpy as np

        gx, gy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        gz = -(weights[0] * gx + weights[1] * gy + bias) / weights[2]
        gz = np.ma.masked_where((gz < 0) | (gz > 1), gz)
        ax.plot_surface(gx, gy, gz, color="#e2b84b", alpha=0.26, linewidth=0)
    ax.set(xlabel="Unidad 1 (x1)", ylabel="Unidad 2 (x2)", zlabel="Unidad 3 (x3)")
    ax.set_xticks((0, 1))
    ax.set_yticks((0, 1))
    ax.set_zticks((0, 1))
    ax.set_title("Patrones binarios y frontera de decisión NET = 0")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Perceptrón académico", page_icon="🧠", layout="wide")

st.title("Perceptrón para aprobar una experiencia curricular")
st.caption("Entrenamiento manual con la regla delta explicada en clase")

history, epoch_summaries, final_weights, final_bias, epochs, converged = train_perceptron()
patterns = make_patterns()

with st.sidebar:
    st.header("Configuración del ejercicio")
    st.metric("Entradas", "3")
    st.metric("Factor de aprendizaje (fa)", format_number(LEARNING_RATE))
    st.write("**Pesos iniciales**")
    st.code(f"w1 = {INITIAL_WEIGHTS[0]}\nw2 = {INITIAL_WEIGHTS[1]}\nw3 = {INITIAL_WEIGHTS[2]}")
    st.metric("Bias inicial", format_number(INITIAL_BIAS))
    st.metric("Nota mínima por unidad", format_number(PASSING_GRADE))
    st.write("Promedios permitidos: **0 a 20**")
    st.divider()
    st.caption("Las notas se convierten a 1 si son ≥ 14 y a 0 si son < 14.")

tab_train, tab_evaluate, tab_method = st.tabs(
    ["Entrenamiento", "Evaluar estudiante", "Método y resultados"]
)

with tab_train:
    st.subheader("Conjunto de entrenamiento")
    st.write("La salida deseada se define para entrenar el perceptrón: 1 si hay al menos dos unidades aprobadas; en caso contrario, 0.")
    st.dataframe(pd.DataFrame(patterns), hide_index=True, use_container_width=True)

    if converged:
        st.success(f"Convergencia alcanzada en {epochs} épocas: la última época clasificó correctamente los ocho patrones.")
    else:
        st.error(f"No hubo convergencia en {MAX_EPOCHS} épocas.")

    st.subheader("Historial completo, patrón por patrón")
    st.caption("‘Antes’ son los parámetros usados para calcular NET. ‘Después’ incluye la actualización por error; cuando error=0 se aplican las mismas fórmulas y los valores quedan iguales.")
    st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True, height=440)

    st.subheader("Operaciones de actualización")
    selected_epoch = st.selectbox("Consultar operaciones de una época", range(1, epochs + 1), index=epochs - 1)
    selected_rows = [row for row in history if row["Época"] == selected_epoch]
    for row in selected_rows:
        with st.expander(f"Época {row['Época']} · Patrón {row['Patrón']} · error {row['error']}"):
            st.write(
                f"NET = ({row['x1']}×{format_number(row['w1 antes'])}) + "
                f"({row['x2']}×{format_number(row['w2 antes'])}) + "
                f"({row['x3']}×{format_number(row['w3 antes'])}) + "
                f"({format_number(row['b antes'])}) = **{format_number(row['NET'])}**"
            )
            st.write(f"Activación: NET {'≥' if row['NET'] >= 0 else '<'} 0, por tanto y = **{row['y']}**. Error = yd − y = {row['yd']} − {row['y']} = **{row['error']}**.")
            st.code(row["Operaciones de actualización"])

    st.subheader("Resumen por época")
    st.dataframe(pd.DataFrame(epoch_summaries), hide_index=True, use_container_width=True)

with tab_evaluate:
    st.subheader("Ingresar promedios de las tres unidades")
    with st.form("student_scores"):
        col1, col2, col3 = st.columns(3)
        with col1:
            grade1 = st.number_input("Promedio unidad 1", min_value=0.0, max_value=20.0, value=14.0, step=0.1)
        with col2:
            grade2 = st.number_input("Promedio unidad 2", min_value=0.0, max_value=20.0, value=13.0, step=0.1)
        with col3:
            grade3 = st.number_input("Promedio unidad 3", min_value=0.0, max_value=20.0, value=14.0, step=0.1)
        submitted = st.form_submit_button("Evaluar con el perceptrón", type="primary")

    if submitted:
        binary = tuple(int(grade >= PASSING_GRADE) for grade in (grade1, grade2, grade3))
        net_terms = tuple(binary[i] * final_weights[i] for i in range(3))
        net = sum(net_terms) + final_bias
        prediction = activation(net)
        a, b_col = st.columns(2)
        with a:
            st.write("**Conversión de promedio a entrada**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Unidad": [1, 2, 3],
                        "Promedio": [grade1, grade2, grade3],
                        "Entrada binaria": binary,
                        "Interpretación": ["Aprobada" if bit else "Desaprobada" for bit in binary],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        with b_col:
            st.write("**Cálculo con los parámetros aprendidos**")
            st.code(
                f"NET = ({binary[0]}×{format_number(final_weights[0])}) + "
                f"({binary[1]}×{format_number(final_weights[1])}) + "
                f"({binary[2]}×{format_number(final_weights[2])}) + "
                f"({format_number(final_bias)}) = {format_number(net)}"
            )
            st.write(f"y = 1 si NET ≥ 0; en caso contrario, y = 0. Resultado: **y = {prediction}**.")
        if prediction:
            st.success("El perceptrón clasifica al estudiante como APROBADO.")
        else:
            st.error("El perceptrón clasifica al estudiante como DESAPROBADO.")

with tab_method:
    st.subheader("Método aplicado")
    st.latex(r"NET=(x_1w_1)+(x_2w_2)+(x_3w_3)+b")
    st.latex(r"y=\begin{cases}1,& NET\geq 0\\0,& NET<0\end{cases}\qquad error=yd-y")
    st.latex(r"w_i(t+1)=w_i(t)+fa\cdot error\cdot x_i\qquad b(t+1)=b(t)+fa\cdot error\cdot 1")
    st.info("La regla de aprobación se usa para crear yd en los patrones de entrenamiento. La evaluación usa solamente NET, la activación y los parámetros aprendidos.")

    st.subheader("Resultados del aprendizaje")
    c1, c2, c3 = st.columns(3)
    c1.metric("Épocas", epochs)
    c2.metric("Estado", "Convergió" if converged else "Sin convergencia")
    c3.metric("Factor de aprendizaje", format_number(LEARNING_RATE))
    st.write(f"**Pesos finales:** w1 = {format_number(final_weights[0])}, w2 = {format_number(final_weights[1])}, w3 = {format_number(final_weights[2])}.  **Bias final:** b = {format_number(final_bias)}.")

    comparison = []
    for pattern in patterns:
        net = sum(pattern[f"x{i}"] * final_weights[i - 1] for i in range(1, 4)) + final_bias
        output = activation(net)
        comparison.append({**pattern, "NET final": net, "Salida obtenida": output, "Correcto": output == pattern["yd"]})
    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    st.pyplot(decision_figure(patterns, final_weights, final_bias), use_container_width=True)

    with st.expander("Aclaración sobre las diapositivas"):
        st.write("En las operaciones de NET visibles en las diapositivas 10 y 11, los resultados 1.5 y 1.4 son correctos. En la actualización final del ejemplo de la diapositiva 10, la etiqueta W2(t+1)=1 debe corregirse a b(t+1)=1, porque se está actualizando el bias.")

