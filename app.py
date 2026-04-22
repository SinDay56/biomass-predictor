import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="Biomass Elemental Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif !important; }
.stApp { background:#f0f4fa; }
.block-container { padding:0 !important; max-width:100% !important; }

/* ── HEADER ── */
.app-header {
    background: linear-gradient(120deg, #0a2463 0%, #1248a8 55%, #1e6abf 100%);
    padding: 1.2rem 2rem;
    display: flex; align-items:center; justify-content:center; gap:16px;
    box-shadow: 0 4px 20px rgba(10,36,99,0.35);
    border-bottom: 3px solid #3a86ff;
}
.header-title {
    font-size:1.6rem !important; font-weight:800 !important;
    color:#ffffff !important; margin:0 !important;
    letter-spacing:-0.02em;
}
.header-sub { font-size:0.78rem; color:rgba(200,220,255,0.85); margin:0 !important;
              letter-spacing:0.08em; text-transform:uppercase; font-weight:500; }

/* ── BODY ── */
.body-wrap { padding:1.6rem 2rem 2.5rem; }

/* ── CARDS ── */
.card {
    background:#ffffff;
    border-radius:16px;
    padding:1.4rem 1.6rem;
    box-shadow:0 2px 16px rgba(10,36,99,0.08), 0 1px 4px rgba(10,36,99,0.04);
    margin-bottom:1.2rem;
    border-top: 3px solid #3a86ff;
}
.card-title {
    font-size:1rem !important; font-weight:700 !important;
    color:#0a2463 !important; margin:0 0 1.1rem 0 !important;
    padding-bottom:0.7rem; border-bottom:2px solid #e8eef8;
    letter-spacing:-0.01em;
}

/* ── Tabla de ingreso de datos ── */
.col-header {
    font-size:0.72rem; font-weight:700; color:#1248a8;
    text-transform:uppercase; letter-spacing:0.07em;
    text-align:center; padding-bottom:4px;
}
.col-range {
    font-size:0.6rem; color:#94a3b8;
    font-family:'JetBrains Mono',monospace; text-align:center; margin-bottom:6px;
}
.sample-label {
    font-size:0.78rem; font-weight:700; color:#1248a8;
    font-family:'JetBrains Mono',monospace; padding:6px 0 2px;
}
.oor-hint {
    font-size:0.6rem; color:#dc2626; text-align:center;
    font-family:'JetBrains Mono',monospace; margin-top:-4px;
}

/* ── Cluster badge ── */
.cluster-pill {
    display:inline-block;
    background:linear-gradient(135deg,#0a2463,#1e6abf);
    color:#fff; border-radius:8px; padding:5px 14px;
    font-size:0.8rem; font-weight:700; letter-spacing:0.03em;
    box-shadow:0 3px 12px rgba(10,36,99,0.3);
}

/* ── Badges elementales ── */
.elem-row-wrap { display:flex; gap:7px; margin-top:0.5rem; }
.ebadge {
    flex:1; border:1.5px solid; border-radius:10px;
    padding:8px 4px; text-align:center;
}
.eb-sym { font-size:0.9rem; font-weight:800; line-height:1; }
.eb-pct { font-size:0.78rem; font-weight:600; margin-top:2px;
          font-family:'JetBrains Mono',monospace; }
.e-C { border-color:#1e6abf; color:#0a2463; background:#eff6ff; }
.e-H { border-color:#0891b2; color:#0e4f6b; background:#ecfeff; }
.e-O { border-color:#3a86ff; color:#1248a8; background:#f0f6ff; }
.e-N { border-color:#6366f1; color:#3730a3; background:#f5f3ff; }
.e-S { border-color:#0ea5e9; color:#0369a1; background:#f0f9ff; }

/* ── Resultado lista lateral ── */
.res-row { display:flex; align-items:center; gap:10px;
           padding:5px 0; border-bottom:1px solid #f1f5f9; }
.res-row:last-child { border-bottom:none; }
.res-sym { font-size:0.9rem; font-weight:800; min-width:18px; }
.res-val { color:#0f172a; font-family:'JetBrains Mono',monospace;
           font-size:0.88rem; font-weight:600; }
.rc{color:#0a2463;} .rh{color:#0e4f6b;} .ro{color:#1248a8;}
.rn{color:#3730a3;} .rs{color:#0369a1;}

/* ── Separador entre muestras ── */
.sample-divider {
    height:2px;
    background:linear-gradient(90deg,transparent,#dde8f8 20%,#dde8f8 80%,transparent);
    margin:1.5rem 0;
}

/* ── BOTONES ── */
.stButton>button {
    background:linear-gradient(135deg,#0a2463,#1e6abf) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.9rem !important;
    font-weight:700 !important; padding:0.65rem 1rem !important;
    width:100% !important; letter-spacing:0.02em !important;
    box-shadow:0 4px 14px rgba(10,36,99,0.3) !important;
    transition:all 0.2s !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#1248a8,#3a86ff) !important;
    transform:translateY(-1px) !important;
    box-shadow:0 6px 18px rgba(10,36,99,0.4) !important;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input {
    background:#f8faff !important; border:1.5px solid #dde8f8 !important;
    border-radius:9px !important; color:#0f172a !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.88rem !important;
    font-weight:600 !important; text-align:center !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:#3a86ff !important;
    box-shadow:0 0 0 3px rgba(58,134,255,0.15) !important;
    background:#fff !important;
}
[data-testid="stNumberInput"] button {
    background:#eef4ff !important; border:1px solid #dde8f8 !important; color:#1248a8 !important;
}
.stNumberInput label { color:#334155 !important; font-size:0.78rem !important; font-weight:600 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#ffffff !important; border-radius:12px !important;
    padding:4px !important; gap:3px !important;
    box-shadow:0 1px 6px rgba(10,36,99,0.08) !important; margin-bottom:1rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.82rem !important;
    font-weight:600 !important; color:#64748b !important;
    border-radius:9px !important; padding:6px 14px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,#0a2463,#1e6abf) !important; color:#fff !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#071840 !important; }
[data-testid="stSidebar"] * { color:#7aafdf !important; }
[data-testid="stTextInput"] input {
    background:#0a2463 !important; border:1px solid rgba(58,134,255,0.2) !important;
    border-radius:7px !important; color:#c0d8f8 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.74rem !important;
}
.stTextInput label { color:#3a86ff !important; font-size:0.74rem !important; }
[data-testid="stFileUploader"] {
    background:rgba(58,134,255,0.05) !important;
    border:1.5px dashed rgba(58,134,255,0.35) !important; border-radius:12px !important;
}
[data-testid="stDownloadButton"]>button {
    background:#fff !important; border:1.5px solid #1248a8 !important;
    color:#1248a8 !important; font-weight:700 !important;
    border-radius:10px !important; width:100% !important;
}
[data-testid="stDataFrame"] * { color:#0f172a !important; }
[data-testid="stDataFrame"] { border-radius:12px !important; overflow:hidden; }

/* alertas */
.al { border-radius:10px; padding:0.65rem 1rem; margin:0.5rem 0;
      font-size:0.81rem; line-height:1.6; border-left:3px solid; }
.a-ok   { background:#f0fdf4; border-color:#22c55e; color:#166534; }
.a-warn { background:#fffbeb; border-color:#f59e0b; color:#92400e; }
.a-err  { background:#fef2f2; border-color:#f87171; color:#991b1b; }
.a-info { background:#eff6ff; border-color:#3a86ff; color:#0a2463; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
RUTA = os.path.dirname(os.path.abspath(__file__))
COLS = ['AR_Moisturecontent','AR_Ashcontent','AR_Volatilematter',
        'AR_Fixedcarbon','AR_Netcalorificvalue(LHV)']
META = {
    'AR_Moisturecontent':        ('Humedad',     '%',     0.10, 83.70, 12.5),
    'AR_Ashcontent':             ('Cenizas',      '%',     0.06, 64.11,  0.5),
    'AR_Volatilematter':         ('Mat. Volátil', '%',     4.82, 85.78, 60.2),
    'AR_Fixedcarbon':            ('C. Fijo',      '%',     0.60, 87.70,  6.8),
    'AR_Netcalorificvalue(LHV)': ('LHV',          'MJ/kg', 0.19, 34.34, 18.6),
}
OUT_KEYS  = ['AR_Carbon','AR_Hydrogen','AR_Nitrogen','AR_Sulphur','AR_Oxygen']
OUT_DISP  = {'AR_Carbon':'C (%)','AR_Hydrogen':'H (%)','AR_Nitrogen':'N (%)',
             'AR_Sulphur':'S (%)','AR_Oxygen':'O (%)'}
OUT_NAMES = ['C','H','N','S','O']
ELEM_ORDER = [
    ('AR_Carbon',   'C', 'e-C', 'rc', '#22c55e', 'Carbono (C)'),
    ('AR_Hydrogen', 'H', 'e-H', 'rh', '#3b82f6', 'Hidrógeno (H)'),
    ('AR_Oxygen',   'O', 'e-O', 'ro', '#0ea5e9', 'Oxígeno (O)'),
    ('AR_Nitrogen', 'N', 'e-N', 'rn', '#f97316', 'Nitrógeno (N)'),
    ('AR_Sulphur',  'S', 'e-S', 'rs', '#84cc16', 'Azufre (S)'),
]

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
clf_path = os.path.join(RUTA, "modelo_final.pkl")
nn_path  = RUTA

with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:0.8rem 0 1rem;">
        <div style="font-size:1.5rem;">🌿</div>
        <div style="font-family:'Fira Code',monospace;font-size:0.82rem;
             color:#60c090;font-weight:500;margin-top:4px;">BiomassIQ</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.6rem;color:#304878;text-transform:uppercase;"
                "letter-spacing:0.12em;'>Rangos válidos (AR)</p>", unsafe_allow_html=True)
    for _,(nm,un,vmin,vmax,_d) in META.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.68rem;">'
            f'<span style="color:#5080b0;">{nm}</span>'
            f'<span style="color:#304878;font-family:Fira Code,monospace;">'
            f'{vmin}–{vmax} {un}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_clf(p): return joblib.load(p)

@st.cache_resource(show_spinner=False)
def load_nn(ruta):
    import tensorflow as tf
    @tf.keras.utils.register_keras_serializable(package='Custom')
    class GlobalR2Metric(tf.keras.metrics.Metric):
        def __init__(self,n_outputs=5,name='global_r2',**kwargs):
            super().__init__(name=name,**kwargs)
            self.n_outputs=n_outputs
            self.sum_sq_res=self.add_weight('ssr',shape=(n_outputs,),initializer='zeros')
            self.sum_y     =self.add_weight('sy', shape=(n_outputs,),initializer='zeros')
            self.sum_y_sq  =self.add_weight('sy2',shape=(n_outputs,),initializer='zeros')
            self.count     =self.add_weight('n',  initializer='zeros')
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_true=tf.cast(y_true,tf.float32);y_pred=tf.cast(y_pred,tf.float32)
            self.sum_sq_res.assign_add(tf.reduce_sum(tf.square(y_true-y_pred),axis=0))
            self.sum_y.assign_add(tf.reduce_sum(y_true,axis=0))
            self.sum_y_sq.assign_add(tf.reduce_sum(tf.square(y_true),axis=0))
            self.count.assign_add(tf.cast(tf.shape(y_true)[0],tf.float32))
        def result(self):
            mu=self.sum_y/(self.count+1e-8)
            ss=self.sum_y_sq-self.count*tf.square(mu)
            return tf.reduce_mean(1.0-self.sum_sq_res/(ss+1e-8))
        def reset_state(self):
            for w in [self.sum_sq_res,self.sum_y,self.sum_y_sq]:w.assign(tf.zeros_like(w))
            self.count.assign(0.0)
        def get_config(self):
            c=super().get_config();c.update({'n_outputs':self.n_outputs});return c
        @classmethod
        def from_config(cls,cfg):return cls(**cfg)
    from tensorflow.keras.models import load_model
    m=load_model(os.path.join(ruta,'model.keras'),compile=False,
                 custom_objects={'GlobalR2Metric':GlobalR2Metric})
    m.compile(optimizer='adam',loss='mse')
    return (m, joblib.load(os.path.join(ruta,'scaler_X.pkl')),
            joblib.load(os.path.join(ruta,'scalers_y.pkl')),
            joblib.load(os.path.join(ruta,'encoder_cluster.pkl')))

def predecir(df_in):
    clf=load_clf(clf_path); df=df_in.copy()
    df['cluster']=clf.predict(df[COLS].values)
    modelo,sx,sy,enc=load_nn(nn_path)
    num_cols=list(sx.feature_names_in_)
    Xn=sx.transform(df[num_cols].astype(float))
    Xc=enc.transform(df[['cluster']])
    Xf=np.hstack([Xn,Xc]).astype(np.float32)
    yp=modelo.predict(Xf,verbose=0)
    for i,col in enumerate(OUT_NAMES):
        df[OUT_DISP[OUT_KEYS[i]]]=sy[col].inverse_transform(yp[:,[i]]).ravel()
    df['Cluster']=df['cluster']
    return df

def chart_muestra(row, label):
    """Gráfica de barras para una muestra."""
    labels = [t[5] for t in ELEM_ORDER]
    vals   = [row[OUT_DISP[t[0]]] for t in ELEM_ORDER]
    colors = [t[4] for t in ELEM_ORDER]
    fig = go.Figure()
    for lbl,val,clr in zip(labels,vals,colors):
        fig.add_trace(go.Bar(
            name=lbl, x=[lbl], y=[val],
            marker_color=clr,
            text=[f'{val:.1f}%'], textposition='outside',
            textfont=dict(size=11, color='#111827', family='Fira Code'),
            width=0.55,
        ))
    fig.update_layout(
        barmode='group', showlegend=False, height=280,
        margin=dict(l=20,r=10,t=20,b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#374151', size=11),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=10, color='#374151')),
        yaxis=dict(gridcolor='#e5e7eb', zeroline=False,
                   ticksuffix='%', tickfont=dict(size=9, color='#9ca3af')),
    )
    return fig

def render_resultado_muestra(row, idx):
    """Renderiza resultado completo de UNA muestra."""
    cluster_val = row.get('Cluster', row.get('cluster', '—'))

    # ── Fila superior: Clasificación | Badges ──
    cc, cb = st.columns([1, 2], gap="medium")

    with cc:
        st.markdown('<div class="card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Clasificación de Biomasa</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;align-items:center;'
            f'justify-content:center;padding:1rem 0;">'
            f'<div class="cluster-pill">Cluster: {cluster_val}</div></div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Predicción Elemental</p>',
                    unsafe_allow_html=True)
        # Fila 1: C H O
        r1 = '<div class="elem-row-wrap">'
        for key,sym,css,_,_,_ in ELEM_ORDER[:3]:
            val = row[OUT_DISP[key]]
            r1 += (f'<div class="ebadge {css}">'
                   f'<div class="eb-sym">{sym}</div>'
                   f'<div class="eb-pct">{val:.1f}%</div></div>')
        r1 += '</div>'
        # Fila 2: N S
        r2 = '<div class="elem-row-wrap" style="margin-top:6px;">'
        for key,sym,css,_,_,_ in ELEM_ORDER[3:]:
            val = row[OUT_DISP[key]]
            r2 += (f'<div class="ebadge {css}">'
                   f'<div class="eb-sym">{sym}</div>'
                   f'<div class="eb-pct">{val:.1f}%</div></div>')
        r2 += '</div>'
        st.markdown(r1 + r2, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Gráfica + lista ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Resultados Predicción</p>',
                unsafe_allow_html=True)
    gc, gl = st.columns([2.4, 1], gap="large")
    with gc:
        st.plotly_chart(chart_muestra(row, f"M{idx+1}"), width='stretch')
    with gl:
        st.markdown('<div style="padding-top:1.5rem;">', unsafe_allow_html=True)
        for key,sym,_,rcls,_,_ in ELEM_ORDER:
            val = row[OUT_DISP[key]]
            st.markdown(
                f'<div class="res-row">'
                f'<span class="res-sym {rcls}">{sym}:</span>'
                f'<span class="res-val">{val:.1f}%</span>'
                f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div style="font-size:2.2rem;line-height:1;">🔬</div>
    <div>
        <p class="header-title">Biomass Elemental Predictor</p>
        <p class="header-sub">Machine Learning · Análisis Elemental</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="body-wrap">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SELECTOR N
# ══════════════════════════════════════════════════════════════════
cn, ch = st.columns([1, 4])
with cn:
    n = st.number_input("Número de muestras", min_value=1, max_value=10000,
                        value=1, step=1)
with ch:
    if n <= 5:
        st.markdown(
            f'<div class="al a-ok" style="margin-top:1.8rem;">'
            f'✏️ &nbsp;Ingresa los datos de las <b>{n}</b> '
            f'muestra{"s" if n>1 else ""} y presiona <b>Clasificar Biomasa</b>.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="al a-info" style="margin-top:1.8rem;">'
            '📂 &nbsp;Para más de 5 muestras carga un archivo Excel.</div>',
            unsafe_allow_html=True)

st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FLUJO MANUAL ≤ 5 muestras
# ══════════════════════════════════════════════════════════════════
if n <= 5:
    # ── Card de ingreso de datos ──────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Datos de Entrada</p>', unsafe_allow_html=True)

    # Cabeceras de columnas (variable + rango)
    col_headers = st.columns([0.7] + [1]*5)
    col_headers[0].markdown("", unsafe_allow_html=True)
    meta_items = list(META.items())
    for hcol, (col_key,(nm,un,vmin,vmax,_)) in zip(col_headers[1:], meta_items):
        hcol.markdown(
            f'<div class="col-header">{nm}</div>'
            f'<div class="col-range">{un} [{vmin}–{vmax}]</div>',
            unsafe_allow_html=True)

    # Filas de muestras
    all_vals = []
    fuera_rango = []
    for i in range(n):
        row_cols = st.columns([0.7] + [1]*5)
        row_cols[0].markdown(
            f'<div class="sample-label">M{i+1}</div>',
            unsafe_allow_html=True)
        vals = []
        for col_ui, (col_key,(nm,un,vmin,vmax,defval)) in \
                zip(row_cols[1:], meta_items):
            v = col_ui.number_input(
                f"{nm} M{i+1}", key=f"{col_key}_{i}",
                min_value=0.0, max_value=999.0,
                value=float(defval), step=0.01,
                label_visibility="collapsed")
            if v < vmin or v > vmax:
                col_ui.markdown(
                    f'<div class="oor-hint">⚠ [{vmin}–{vmax}]</div>',
                    unsafe_allow_html=True)
                fuera_rango.append(f"M{i+1}·{nm}: {v:.2f}")
            vals.append(v)
        all_vals.append(vals)

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

    if fuera_rango:
        st.markdown(
            f'<div class="al a-warn">⚠ <b>Valores fuera del rango de entrenamiento:</b> '
            f'{" &nbsp;·&nbsp; ".join(fuera_rango)}</div>',
            unsafe_allow_html=True)

    # Botón único al final del card
    cb1, cb2 = st.columns([1, 3])
    ejecutar = cb1.button("🔬  Clasificar Biomasa")
    st.markdown('</div>', unsafe_allow_html=True)   # cierra .card

    # ── Predicción + resultados ───────────────────────────────────
    if ejecutar:
        err = []
        if not os.path.exists(clf_path): err.append(f"Clasificador no encontrado: {clf_path}")
        if not os.path.isdir(nn_path):   err.append(f"Carpeta no encontrada: {nn_path}")
        for e in err:
            st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
        if err:
            st.stop()

        df_input = pd.DataFrame(all_vals, columns=COLS)
        with st.spinner("Analizando todas las muestras..."):
            try:
                df_res = predecir(df_input)
            except Exception as e:
                st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
                st.stop()

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        # ── Todas las muestras juntas en columnas ──
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Resultados Predicción</p>',
                    unsafe_allow_html=True)
        cols_ch = st.columns(len(df_res))
        for i, col in enumerate(cols_ch):
            row = df_res.iloc[i]
            cluster_val = row.get('Cluster', row.get('cluster','—'))
            with col:
                # Título muestra + cluster
                st.markdown(
                    f'<div style="text-align:center;margin-bottom:6px;">'
                    f'<span style="font-family:Fira Code,monospace;font-size:0.72rem;'
                    f'font-weight:700;color:#1a7a3c;">MUESTRA {i+1}</span>'
                    f'&nbsp;&nbsp;'
                    f'<span style="background:linear-gradient(135deg,#1a7a3c,#2ab54a);'
                    f'color:#fff;border-radius:6px;padding:2px 10px;'
                    f'font-size:0.68rem;font-weight:700;'
                    f'font-family:Fira Code,monospace;">Cluster: {cluster_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True)
                st.plotly_chart(chart_muestra(row, f"M{i+1}"), width='stretch')
                # Lista de valores
                for key,sym,_,rcls,_,_ in ELEM_ORDER:
                    val = row[OUT_DISP[key]]
                    st.markdown(
                        f'<div class="res-row">'
                        f'<span class="res-sym {rcls}">{sym}:</span>'
                        f'<span class="res-val">{val:.1f}%</span>'
                        f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Descarga ──
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
        cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
        df_dl     = df_res.rename(columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}})[cols_show].copy()
        df_dl.index = [f"M{i+1}" for i in range(len(df_dl))]
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df_dl.to_excel(w, sheet_name='Resultados', index=True)
        cd, _ = st.columns([1, 4])
        cd.download_button(
            "⬇  Descargar Excel",
            data=buf.getvalue(),
            file_name="predicciones_biomasa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════
# FLUJO EXCEL > 5 muestras
# ══════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Cargar archivo Excel</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="al a-info" style="margin-bottom:0.8rem;">Columnas requeridas: '
        '<code style="font-family:Fira Code,monospace;font-size:0.76rem;'
        'background:#dbeafe;padding:1px 5px;border-radius:4px;color:#1e40af;">'
        'AR_Moisturecontent · AR_Ashcontent · AR_Volatilematter · '
        'AR_Fixedcarbon · AR_Netcalorificvalue(LHV)</code></div>',
        unsafe_allow_html=True)

    archivo = st.file_uploader("Seleccionar archivo Excel (.xlsx)",
                               type=["xlsx"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo:
        try:
            df_raw = pd.read_excel(archivo, engine='openpyxl')
            miss   = [c for c in COLS if c not in df_raw.columns]
            if miss:
                st.markdown(f'<div class="al a-err">❌ Columnas faltantes: {", ".join(miss)}</div>',
                            unsafe_allow_html=True)
                st.stop()

            # Validar rangos
            probs = []
            for ck,(nm,un,vmin,vmax,_) in META.items():
                bad = df_raw[(df_raw[ck]<vmin)|(df_raw[ck]>vmax)]
                if not bad.empty: probs.append(f"{nm}: {len(bad)} muestra(s)")
            if probs:
                st.markdown(
                    f'<div class="al a-warn">⚠ <b>{len(df_raw)} muestras cargadas.</b> '
                    f'Fuera de rango → {" · ".join(probs)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="al a-ok">✅ <b>{len(df_raw)} muestras cargadas</b> '
                    f'— todos los valores dentro del rango.</div>', unsafe_allow_html=True)

            cb1, _ = st.columns([1, 3])
            if cb1.button("🔬  Predecir todas las muestras"):
                err = []
                if not os.path.exists(clf_path): err.append("Clasificador no encontrado.")
                if not os.path.isdir(nn_path):   err.append("Carpeta no encontrada.")
                for e in err:
                    st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
                if err: st.stop()

                with st.spinner("Analizando todas las muestras..."):
                    try:
                        df_res = predecir(df_raw)
                    except Exception as e:
                        st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
                        st.stop()

                st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="card-title">Resultados — Todas las muestras</p>',
                            unsafe_allow_html=True)
                lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
                cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
                df_tabla  = df_res.rename(
                    columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}}
                )[cols_show].copy()
                df_tabla.index = [f"M{i+1}" for i in range(len(df_tabla))]
                st.dataframe(
                    df_tabla.style
                        .format(precision=3)
                        .background_gradient(subset=list(OUT_DISP.values()), cmap='Greens')
                        .set_properties(**{'color':'#111827','font-weight':'500'}),
                    width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)

                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    df_tabla.to_excel(w, sheet_name='Resultados', index=True)
                cd, _ = st.columns([1, 3])
                cd.download_button(
                    "⬇  Descargar Excel completo", data=buf.getvalue(),
                    file_name="predicciones_biomasa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.markdown(f'<div class="al a-err">❌ Error: {e}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
