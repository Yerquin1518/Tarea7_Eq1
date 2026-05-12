import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador Gamma-Pydra",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #00D4FF; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        color: #aaa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4FF22;
        color: #00D4FF;
        border-bottom: 2px solid #00D4FF;
    }
    .card {
        background-color: #1e1e2e;
        border-left: 4px solid #00D4FF;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .card-green { border-left-color: #4ECDC4; }
    .card-orange { border-left-color: #FFB347; }
    .card-red { border-left-color: #FF6B6B; }
    code {
        background-color: #1a1a2e;
        color: #00D4FF;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    pre {
        background-color: #12121f !important;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# CLASIFICADOR GAMMA-PYDRA
# ─────────────────────────────────────────────
class GammaPydra:
    def __init__(self, rho_mode="max", weights=None, missing_val=0, both_missing_val=1):
        self.rho_mode = rho_mode
        self.weights = weights
        self.missing_val = missing_val
        self.both_missing_val = both_missing_val
        self.classes_ = self.X_train_ = self.y_train_ = self.rho_ = None
        self.n_features_ = self.weights_ = None
 
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.weights_ = np.array(self.weights, dtype=float) if self.weights is not None else np.ones(self.n_features_, dtype=float)
        with np.errstate(all="ignore"):
            all_diffs = np.abs(X[:, None, :] - X[None, :, :])
            if self.rho_mode == "max":
                self.rho_ = float(np.nanmax(all_diffs))
            else:
                col_max = np.nanmax(all_diffs, axis=(0, 1))
                self.rho_ = float(np.nanmin(col_max))
        return self
 
    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._classify_one(x) for x in X])
 
    def _classify_one(self, sample):
        theta = 0.0
        while True:
            scores = {}
            for cls in self.classes_:
                mask = self.y_train_ == cls
                X_cls = self.X_train_[mask]
                scores[cls] = max(self._weighted_sum(sample, X_cls[k], theta) for k in range(len(X_cls)))
            max_score = max(scores.values())
            winners = [c for c, s in scores.items() if s == max_score]
            if len(winners) == 1:
                return winners[0]
            if theta < self.rho_:
                theta += 1.0
            else:
                return winners[0]
 
    def _weighted_sum(self, sample, pattern, theta):
        return sum(self.weights_[j] * self._gamma_pydra(sample[j], pattern[j], theta) for j in range(self.n_features_))
 
    def _gamma_pydra(self, xa, xb, theta):
        xa_nan, xb_nan = np.isnan(xa), np.isnan(xb)
        if xa_nan and xb_nan:
            return self.both_missing_val
        if xa_nan or xb_nan:
            return self.missing_val
        return 1 if abs(xa - xb) <= theta else 0
 
# ─────────────────────────────────────────────
# CARGA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_and_train():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df_raw = pd.read_csv(url)
 
    df = df_raw.copy()
    df = df[df["sex"].notna()]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
 
    df["male"] = (df["sex"] == "Male").astype(int)
    for island in df["island"].unique():
        df["Island: " + island] = (df["island"] == island).astype(int)
    df.drop(["sex", "island"], axis=1, inplace=True)
 
    SEED = 0
    np.random.seed(SEED)
 
    X = df.drop("species", axis=1).values
    y = df["species"]
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )
 
    results = {}
 
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results["KNN (k=5)"] = {
        "acc": accuracy_score(y_test, y_pred_knn),
        "cm": confusion_matrix(y_test, y_pred_knn),
        "report": classification_report(y_test, y_pred_knn, output_dict=True),
        "y_pred": y_pred_knn,
        "y_test": y_test
    }
 
    # Naïve Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results["Naïve Bayes"] = {
        "acc": accuracy_score(y_test, y_pred_nb),
        "cm": confusion_matrix(y_test, y_pred_nb),
        "report": classification_report(y_test, y_pred_nb, output_dict=True),
        "y_pred": y_pred_nb,
        "y_test": y_test
    }
 
    # Árbol de Decisión
    dt = DecisionTreeClassifier(random_state=SEED)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results["Árbol de Decisión"] = {
        "acc": accuracy_score(y_test, y_pred_dt),
        "cm": confusion_matrix(y_test, y_pred_dt),
        "report": classification_report(y_test, y_pred_dt, output_dict=True),
        "y_pred": y_pred_dt,
        "y_test": y_test
    }
 
    # Random Forest
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "acc": accuracy_score(y_test, y_pred_rf),
        "cm": confusion_matrix(y_test, y_pred_rf),
        "report": classification_report(y_test, y_pred_rf, output_dict=True),
        "y_pred": y_pred_rf,
        "y_test": y_test
    }
 
    # Gamma-Pydra
    gamma = GammaPydra()
    gamma.fit(X_train, y_train.values)
    y_pred_gamma = gamma.predict(X_test)
    results["Gamma-Pydra"] = {
        "acc": accuracy_score(y_test, y_pred_gamma),
        "cm": confusion_matrix(y_test, y_pred_gamma),
        "report": classification_report(y_test, y_pred_gamma, output_dict=True),
        "y_pred": y_pred_gamma,
        "y_test": y_test
    }
 
    return df_raw, df, results
 
with st.spinner("Entrenando modelos..."):
    df_raw, df_clean, results = load_and_train()
 
# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🐧 Clasificador Gamma-Pydra")
st.markdown("**Tarea 7 · Equipo 1 · Presentación de Resultados**")
st.markdown("---")
 
tab1, tab2, tab3 = st.tabs(["🧠 ¿Qué es Gamma?", "📊 Dataset", "📈 Comparativa de Modelos"])
 
# ═══════════════════════════════════════════════════════
# TAB 1 — EXPLICACIÓN DEL CLASIFICADOR GAMMA
# ═══════════════════════════════════════════════════════
with tab1:
    st.header("El Clasificador Gamma")
    st.markdown("""
    El **Clasificador Gamma** es un método de clasificación basado en similitud por umbrales,
    desarrollado como alternativa a modelos como KNN o SVM. En lugar de medir distancias métricas,
    define si dos valores son "similares" mediante el operador **γ** y un parámetro de tolerancia **θ**.
    """)
 
    st.markdown("---")
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("⚙️ Operador γ (Gamma)")
        st.markdown("""
        <div class="card">
        El operador γ compara dos valores escalares <b>xₐ</b> y <b>x_b</b> con un umbral θ:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        \gamma(x_a,\, x_b,\, \theta) =
        \begin{cases}
        1 & \text{si } |x_a - x_b| \leq \theta \\
        0 & \text{si } |x_a - x_b| > \theta
        \end{cases}
        """)
        st.markdown("""
        - Retorna **1** cuando los valores son suficientemente similares.
        - Retorna **0** cuando difieren más que el umbral **θ**.
        - θ comienza en **0** y se incrementa de a 1 hasta resolver empates.
        """)
 
    with col2:
        st.subheader("📐 Score por clase")
        st.markdown("""
        <div class="card card-green">
        El score de un patrón de prueba <b>x</b> frente a una clase <b>C</b> es el máximo de las sumas ponderadas contra todos los patrones de entrenamiento de esa clase:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        \text{score}(x, C) = \max_{k \in C}\ \sum_{j=1}^{d} w_j \cdot \gamma(x_j,\, x_j^{(k)},\, \theta)
        """)
        st.markdown("""
        - **w_j**: peso de la dimensión *j* (por defecto todos = 1).
        - El patrón se asigna a la clase con mayor score.
        - Si hay empate, θ sube y se reevalúa hasta **ρ** (umbral de paro).
        """)
 
    st.markdown("---")
 
    st.subheader("🔄 Parámetro de paro ρ (rho)")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="card">
        <b>ρ (modo max)</b><br>
        Máximo de todas las diferencias absolutas entre pares de patrones de entrenamiento. Es el umbral más conservador.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card card-green">
        <b>ρ (modo min)</b><br>
        Mínimo de los máximos por dimensión. Más restrictivo; detiene la búsqueda antes.
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="card card-orange">
        <b>Rol de ρ</b><br>
        Si θ supera ρ sin resolver el empate, se elige al primer ganador por orden. Garantiza que el algoritmo siempre termina.
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("---")
 
    st.subheader("🧬 Extensión PYDRA — Valores Perdidos")
    st.markdown("""
    La extensión **PYDRA** adapta el operador γ para manejar **NaN** sin eliminar instancias ni imputar valores artificialmente:
    """)
 
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.latex(r"""
        \gamma_{\text{PYDRA}}(x_a, x_b, \theta) =
        \begin{cases}
        1 & \text{si } x_a = \text{NaN} \land x_b = \text{NaN} \\
        0 & \text{si } x_a = \text{NaN} \oplus x_b = \text{NaN} \\
        \gamma(x_a, x_b, \theta) & \text{en otro caso}
        \end{cases}
        """)
    with col_right:
        st.markdown("""
        | Situación | Resultado |
        |---|---|
        | Ambos son NaN | **1** → se asumen similares |
        | Solo uno es NaN | **0** → no contribuye al score |
        | Ninguno es NaN | Se aplica γ normal con θ |
        """)
 
    st.markdown("---")
    st.subheader("🔁 Algoritmo de clasificación — Paso a paso")
    st.markdown("""
    <div class="card">
    <ol>
        <li>Inicializar <b>θ = 0</b>.</li>
        <li>Para cada clase, calcular su score máximo contra el patrón a clasificar.</li>
        <li>Seleccionar la clase con el score más alto.</li>
        <li>Si hay <b>un único ganador</b> → asignar esa clase. ✅</li>
        <li>Si hay <b>empate</b> y θ < ρ → incrementar θ en 1 y repetir desde el paso 2.</li>
        <li>Si θ ≥ ρ → elegir al primer ganador en lista. 🏁</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.subheader("💻 Implementación clave: `_classify_one`")
    st.code("""
def _classify_one(self, sample):
    theta = 0.0
 
    while True:
        scores = {}
        for cls in self.classes_:
            mask = self.y_train_ == cls
            X_cls = self.X_train_[mask]
            scores[cls] = max(
                self._weighted_sum(sample, X_cls[k], theta)
                for k in range(len(X_cls))
            )
 
        max_score = max(scores.values())
        winners = [c for c, s in scores.items() if s == max_score]
 
        if len(winners) == 1:
            return winners[0]       # Ganador único → clasificar
 
        if theta < self.rho_:
            theta += 1.0            # Empate → ampliar umbral
        else:
            return winners[0]       # Llegamos a rho → desempate por orden
    """, language="python")
 
# ═══════════════════════════════════════════════════════
# TAB 2 — DATASET
# ═══════════════════════════════════════════════════════
with tab2:
    st.header("Dataset: Palmer Penguins 🐧")
    st.markdown("""
    El dataset **Palmer Penguins** contiene mediciones morfológicas de **344 pingüinos**
    de tres especies distintas, recolectadas en las islas del archipiélago Palmer (Antártida).
    Es ampliamente usado como alternativa al dataset Iris para tareas de clasificación multiclase.
    """)
 
    st.markdown("---")
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de registros", f"{len(df_raw)}")
    with col2:
        st.metric("Tras limpieza", f"{len(df_clean) if 'species' in df_raw.columns else '~333'}")
    with col3:
        st.metric("Variables originales", "8")
    with col4:
        st.metric("Clases", "3 especies")
 
    st.markdown("---")
 
    # Variables y clases
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Variables del dataset")
        st.markdown("""
        | Variable | Tipo | Descripción |
        |---|---|---|
        | `species` | Categórica (target) | Adelie / Chinstrap / Gentoo |
        | `island` | Categórica | Biscoe / Dream / Torgersen |
        | `bill_length_mm` | Numérica | Longitud del pico (mm) |
        | `bill_depth_mm` | Numérica | Profundidad del pico (mm) |
        | `flipper_length_mm` | Numérica | Longitud de la aleta (mm) |
        | `body_mass_g` | Numérica | Masa corporal (g) |
        | `sex` | Categórica | MALE / FEMALE |
        """)
 
    with col_b:
        st.subheader("Preprocesamiento aplicado")
        st.markdown("""
        <div class="card card-green">
        <ol>
            <li>Eliminación de filas con <code>sex == "."</code> y NaN.</li>
            <li>Codificación de <b>sex</b> → columna binaria <code>male</code>.</li>
            <li>One-hot encoding de <b>island</b> → 3 columnas binarias.</li>
            <li>Normalización con <b>StandardScaler</b>.</li>
            <li>Split 80/20 estratificado por especie.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.subheader("Distribución de especies")
 
    species_counts = df_raw["species"].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=species_counts.index,
        values=species_counts.values,
        hole=0.45,
        marker_colors=["#00D4FF", "#4ECDC4", "#FF6B6B"]
    )])
    fig_pie.update_layout(template="plotly_dark", height=320, showlegend=True)
 
    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_bar:
        fig_bar = px.bar(
            x=species_counts.index, y=species_counts.values,
            color=species_counts.index,
            color_discrete_sequence=["#00D4FF", "#4ECDC4", "#FF6B6B"],
            labels={"x": "Especie", "y": "Cantidad"},
            template="plotly_dark"
        )
        fig_bar.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
 
    st.markdown("---")
    st.subheader("EDA — Distribuciones de variables numéricas por especie")
    st.markdown("Histogramas superpuestos de las 4 variables continuas, separados por especie (Adelie=azul, Chinstrap=verde, Gentoo=rojo).")
 
    df_eda = df_raw.dropna(subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "species"])
    color_map = {"Adelie": "#00D4FF", "Chinstrap": "#4ECDC4", "Gentoo": "#FF6B6B"}
    numeric_vars = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    labels_map = {
        "bill_length_mm": "Longitud del pico (mm)",
        "bill_depth_mm": "Profundidad del pico (mm)",
        "flipper_length_mm": "Longitud de aleta (mm)",
        "body_mass_g": "Masa corporal (g)"
    }
 
    col_hist1, col_hist2 = st.columns(2)
    for i, var in enumerate(numeric_vars):
        fig_h = px.histogram(
            df_eda, x=var, color="species",
            barmode="overlay",
            color_discrete_map=color_map,
            opacity=0.7,
            labels={"species": "Especie", var: labels_map[var]},
            template="plotly_dark"
        )
        fig_h.update_layout(height=300, showlegend=(i == 0),
                            legend=dict(orientation="h", y=-0.3))
        if i % 2 == 0:
            with col_hist1:
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            with col_hist2:
                st.plotly_chart(fig_h, use_container_width=True)
 
    st.markdown("---")
    st.subheader("EDA — Distribuciones por sexo")
 
    df_sex = df_raw.dropna(subset=["sex"] + numeric_vars)
    df_sex = df_sex[df_sex["sex"].isin(["Male", "Female"])]
    sex_color_map = {"Male": "#00D4FF", "Female": "#FF6B6B"}
 
    col_s1, col_s2 = st.columns(2)
    for i, var in enumerate(numeric_vars):
        fig_s = px.histogram(
            df_sex, x=var, color="sex",
            barmode="overlay",
            color_discrete_map=sex_color_map,
            opacity=0.7,
            labels={"sex": "Sexo", var: labels_map[var]},
            template="plotly_dark"
        )
        fig_s.update_layout(height=300, showlegend=(i == 0),
                            legend=dict(orientation="h", y=-0.3))
        if i % 2 == 0:
            with col_s1:
                st.plotly_chart(fig_s, use_container_width=True)
        else:
            with col_s2:
                st.plotly_chart(fig_s, use_container_width=True)
 
# ═══════════════════════════════════════════════════════
# TAB 3 — COMPARATIVA DE MODELOS
# ═══════════════════════════════════════════════════════
with tab3:
    st.header("Comparativa de Clasificadores")
    st.markdown("Evaluación sobre el **conjunto de prueba (20%)** con split estratificado por especie.")
 
    modelos = list(results.keys())
    accs = [results[m]["acc"] for m in modelos]
 
    # ── Tabla de accuracy ──
    st.subheader("Accuracy general")
    df_acc = pd.DataFrame({
        "Modelo": modelos,
        "Accuracy (%)": [round(a * 100, 2) for a in accs]
    }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    df_acc.index += 1
 
    col_t, col_b_acc = st.columns([1, 2])
    with col_t:
        st.dataframe(df_acc, use_container_width=True)
    with col_b_acc:
        fig_acc = px.bar(
            df_acc, x="Modelo", y="Accuracy (%)",
            color="Accuracy (%)",
            color_continuous_scale=["#FF6B6B", "#FFB347", "#4ECDC4", "#00D4FF"],
            range_color=[85, 100],
            text="Accuracy (%)",
            template="plotly_dark"
        )
        fig_acc.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_acc.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_acc, use_container_width=True)
 
    st.markdown("---")
 
    # ── Matrices de confusión ──
    st.subheader("Matrices de confusión")
    classes = ["Adelie", "Chinstrap", "Gentoo"]
 
    cols = st.columns(5)
    for i, (modelo, col) in enumerate(zip(modelos, cols)):
        cm = results[modelo]["cm"]
        acc = results[modelo]["acc"]
        with col:
            fig_cm = px.imshow(
                cm,
                x=classes, y=classes,
                color_continuous_scale="Blues",
                text_auto=True,
                template="plotly_dark",
                labels=dict(x="Predicho", y="Real", color="N")
            )
            fig_cm.update_layout(
                height=260,
                title=dict(text=f"{modelo}<br><sup>Acc: {acc:.2%}</sup>", font=dict(size=12, color="#00D4FF")),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=60, b=10),
                xaxis=dict(tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig_cm, use_container_width=True)
 
    st.markdown("---")
 
    # ── Métricas por clase ──
    st.subheader("Métricas por clase (Precision / Recall / F1)")
    selected_model = st.selectbox("Selecciona un modelo para ver su reporte:", modelos, index=modelos.index("Gamma-Pydra"))
 
    report = results[selected_model]["report"]
    rows = []
    for cls in classes:
        if cls in report:
            rows.append({
                "Clase": cls,
                "Precision": round(report[cls]["precision"], 3),
                "Recall": round(report[cls]["recall"], 3),
                "F1-Score": round(report[cls]["f1-score"], 3),
                "Support": int(report[cls]["support"])
            })
    df_report = pd.DataFrame(rows)
 
    col_rep, col_radar = st.columns([1, 1])
    with col_rep:
        st.dataframe(df_report, use_container_width=True, hide_index=True)
        macro = report.get("macro avg", {})
        st.markdown(f"""
        <div class="card card-green" style="margin-top:12px">
        <b>Macro Avg</b> → Precision: <b>{macro.get('precision', 0):.3f}</b> |
        Recall: <b>{macro.get('recall', 0):.3f}</b> |
        F1: <b>{macro.get('f1-score', 0):.3f}</b>
        </div>
        """, unsafe_allow_html=True)
 
    with col_radar:
        fig_radar = go.Figure()
        categories = ["Precision", "Recall", "F1-Score"]
        color_list = ["#00D4FF", "#4ECDC4", "#FF6B6B"]
        for j, row in df_report.iterrows():
            vals = [row["Precision"], row["Recall"], row["F1-Score"]]
            vals_closed = vals + [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Clase"],
                line_color=color_list[j],
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.7, 1.0])),
            template="plotly_dark",
            height=330,
            showlegend=True,
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
 
    st.markdown("---")
 
    # ── Comparativa F1 por clase ──
    st.subheader("F1-Score por clase — todos los modelos")
    rows_all = []
    for mod in modelos:
        rep = results[mod]["report"]
        for cls in classes:
            if cls in rep:
                rows_all.append({"Modelo": mod, "Clase": cls, "F1-Score": round(rep[cls]["f1-score"], 3)})
    df_f1 = pd.DataFrame(rows_all)
 
    fig_f1 = px.bar(
        df_f1, x="Clase", y="F1-Score", color="Modelo",
        barmode="group",
        color_discrete_sequence=["#00D4FF", "#4ECDC4", "#FFB347", "#FF6B6B", "#9B59B6"],
        template="plotly_dark",
        range_y=[0.7, 1.05],
        text="F1-Score"
    )
    fig_f1.update_traces(texttemplate="%{text:.3f}", textposition="outside", textfont_size=9)
    fig_f1.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_f1, use_container_width=True)
 
    st.markdown("---")
 
    # ── Conclusión ──
    best_model = df_acc.iloc[0]["Modelo"]
    best_acc = df_acc.iloc[0]["Accuracy (%)"]
    gamma_acc = results["Gamma-Pydra"]["acc"] * 100
 
    st.subheader("📝 Conclusiones")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"""
        <div class="card card-green">
        <b>Mejor modelo general:</b> {best_model} con <b>{best_acc:.2f}%</b> de accuracy.
        </div>
        <div class="card" style="margin-top:12px">
        <b>Gamma-Pydra</b> obtuvo <b>{gamma_acc:.2f}%</b> de accuracy, siendo un clasificador
        completamente implementado desde cero sin librerías de ML externas.
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown("""
        <div class="card card-orange">
        <b>Ventajas de Gamma-Pydra:</b>
        <ul>
            <li>No requiere supuestos de distribución.</li>
            <li>Maneja valores perdidos nativamente (PYDRA).</li>
            <li>Interpretable: decisiones basadas en similitud directa.</li>
            <li>Parámetros ajustables (ρ, pesos por dimensión).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
