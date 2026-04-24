import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="BiomassIQ — Elemental Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# CSS — Diseño verde llamativo con fondo estructurado
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family:'Outfit',sans-serif !important; }

/* ── FONDO CON ESTRUCTURA MOLECULAR ── */
.stApp {
    background-color: #f0faf3;
    background-image:
        radial-gradient(circle at 1px 1px, rgba(34,197,94,0.12) 1px, transparent 0),
        radial-gradient(ellipse at 80% 20%, rgba(16,185,129,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 10% 80%, rgba(5,150,105,0.07) 0%, transparent 50%);
    background-size: 32px 32px, 100% 100%, 100% 100%;
}

/* Líneas moleculares decorativas */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(60deg, rgba(34,197,94,0.04) 1px, transparent 1px),
        linear-gradient(120deg, rgba(34,197,94,0.04) 1px, transparent 1px);
    background-size: 80px 80px;
    pointer-events: none;
    z-index: 0;
}

.block-container { padding:0 !important; max-width:100% !important; position:relative; z-index:1; }

/* ── Ocultar barra superior de Streamlit ── */
[data-testid="stToolbar"] { display:none !important; }
[data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { display:none !important; }

/* ── HEADER ── */
.app-header {
    background: linear-gradient(120deg, #064e3b 0%, #065f46 40%, #047857 75%, #059669 100%);
    padding: 1.4rem 2rem;
    display: flex; align-items:center; justify-content:center; gap:18px;
    box-shadow: 0 6px 30px rgba(6,78,59,0.4);
    border-bottom: 4px solid #34d399;
    position: relative; overflow: hidden;
}
.app-header::after {
    content: '⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡';
    position: absolute; right: -10px; top: 50%; transform: translateY(-50%);
    font-size: 1.4rem; color: rgba(255,255,255,0.06);
    letter-spacing: 8px; white-space: nowrap;
}
.header-icon { font-size:2.8rem; line-height:1; filter:drop-shadow(0 2px 8px rgba(0,0,0,0.3)); }
.header-title {
    font-size:1.8rem !important; font-weight:900 !important;
    color:#ffffff !important; margin:0 !important;
    letter-spacing:-0.03em; text-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.header-sub {
    font-size:0.75rem; color:rgba(167,243,208,0.9); margin:0 !important;
    letter-spacing:0.15em; text-transform:uppercase; font-weight:600;
}

/* ── BODY ── */
.body-wrap { padding:1.8rem 2rem 3rem; }

/* ── CARDS ── */
.card {
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(10px);
    border-radius:20px;
    padding:1.5rem 1.8rem;
    box-shadow: 0 4px 24px rgba(6,78,59,0.1), 0 1px 6px rgba(6,78,59,0.06);
    margin-bottom:1.4rem;
    border: 1px solid rgba(52,211,153,0.2);
    border-top: 4px solid #10b981;
    position: relative;
}
.card::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%);
    border-radius: 0 20px 0 0;
    pointer-events: none;
}
.card-title {
    font-size:1.05rem !important; font-weight:800 !important;
    color:#064e3b !important; margin:0 0 1.1rem 0 !important;
    padding-bottom:0.8rem; border-bottom:2px solid #d1fae5;
    letter-spacing:-0.01em; display:flex; align-items:center; gap:8px;
}

/* ── Tabla de ingreso de datos ── */
.col-header {
    font-size:0.72rem; font-weight:800; color:#065f46;
    text-transform:uppercase; letter-spacing:0.08em;
    text-align:center; padding-bottom:4px;
}
.col-range {
    font-size:0.6rem; color:#6ee7b7;
    font-family:'JetBrains Mono',monospace; text-align:center; margin-bottom:6px;
    background: #064e3b; padding: 2px 5px; border-radius: 4px;
}
.sample-label {
    font-size:0.82rem; font-weight:800; color:#059669;
    font-family:'JetBrains Mono',monospace; padding:6px 0 2px;
}
.oor-hint {
    font-size:0.6rem; color:#dc2626; text-align:center;
    font-family:'JetBrains Mono',monospace; margin-top:-4px;
}

/* ── Cluster badge ── */
.cluster-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:linear-gradient(135deg,#064e3b,#059669);
    color:#fff; border-radius:10px; padding:6px 16px;
    font-size:0.85rem; font-weight:800; letter-spacing:0.02em;
    box-shadow:0 4px 16px rgba(6,78,59,0.4);
}

/* ── Descripción de cluster ── */
.cluster-desc {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #6ee7b7;
    border-left: 4px solid #10b981;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #064e3b;
    font-weight: 500;
    margin-top: 10px;
    font-style: italic;
    line-height: 1.5;
}

/* ── Badges elementales ── */
.elem-row-wrap { display:flex; gap:8px; margin-top:0.6rem; }
.ebadge {
    flex:1; border:2px solid; border-radius:12px;
    padding:10px 4px; text-align:center;
    transition: transform 0.15s;
}
.ebadge:hover { transform: translateY(-2px); }
.eb-sym { font-size:1rem; font-weight:900; line-height:1; }
.eb-pct { font-size:0.8rem; font-weight:700; margin-top:3px;
          font-family:'JetBrains Mono',monospace; }
.e-C { border-color:#059669; color:#064e3b; background:linear-gradient(135deg,#ecfdf5,#d1fae5); }
.e-H { border-color:#0891b2; color:#0e4f6b; background:linear-gradient(135deg,#ecfeff,#cffafe); }
.e-O { border-color:#7c3aed; color:#4c1d95; background:linear-gradient(135deg,#f5f3ff,#ede9fe); }
.e-N { border-color:#d97706; color:#92400e; background:linear-gradient(135deg,#fffbeb,#fef3c7); }
.e-S { border-color:#dc2626; color:#991b1b; background:linear-gradient(135deg,#fef2f2,#fee2e2); }

/* ── Resultado lista lateral ── */
.res-row { display:flex; align-items:center; gap:10px;
           padding:6px 0; border-bottom:1px solid #ecfdf5; }
.res-row:last-child { border-bottom:none; }
.res-sym { font-size:0.95rem; font-weight:900; min-width:20px; }
.res-val { color:#111827; font-family:'JetBrains Mono',monospace;
           font-size:0.9rem; font-weight:700; }
.rc{color:#059669;} .rh{color:#0891b2;} .ro{color:#7c3aed;}
.rn{color:#d97706;} .rs{color:#dc2626;}

/* ── Separador entre muestras ── */
.sample-divider {
    height:2px;
    background:linear-gradient(90deg,transparent,#6ee7b7 20%,#6ee7b7 80%,transparent);
    margin:1.5rem 0;
}

/* ── BOTONES ── */
.stButton>button {
    background:linear-gradient(135deg,#064e3b,#059669) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-family:'Outfit',sans-serif !important; font-size:0.92rem !important;
    font-weight:800 !important; padding:0.7rem 1.2rem !important;
    width:100% !important; letter-spacing:0.02em !important;
    box-shadow:0 4px 16px rgba(6,78,59,0.35) !important;
    transition:all 0.2s !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#065f46,#10b981) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(6,78,59,0.45) !important;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input {
    background:#f0fdf4 !important; border:2px solid #a7f3d0 !important;
    border-radius:10px !important; color:#064e3b !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.9rem !important;
    font-weight:700 !important; text-align:center !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:#10b981 !important;
    box-shadow:0 0 0 3px rgba(16,185,129,0.2) !important;
    background:#fff !important;
}
[data-testid="stNumberInput"] button {
    background:#d1fae5 !important; border:1px solid #a7f3d0 !important; color:#065f46 !important;
}
.stNumberInput label { color:#374151 !important; font-size:0.78rem !important; font-weight:700 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.9) !important; border-radius:14px !important;
    padding:5px !important; gap:4px !important;
    box-shadow:0 2px 12px rgba(6,78,59,0.1) !important; margin-bottom:1rem !important;
    border: 1px solid rgba(52,211,153,0.2) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'Outfit',sans-serif !important; font-size:0.84rem !important;
    font-weight:700 !important; color:#6b7280 !important;
    border-radius:10px !important; padding:7px 16px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,#064e3b,#059669) !important; color:#fff !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#022c22 !important; }
[data-testid="stSidebar"] * { color:#6ee7b7 !important; }
[data-testid="stFileUploader"] {
    background:rgba(16,185,129,0.05) !important;
    border:2px dashed rgba(52,211,153,0.4) !important; border-radius:14px !important;
}
[data-testid="stDownloadButton"]>button {
    background:linear-gradient(135deg,#ecfdf5,#d1fae5) !important;
    border:2px solid #059669 !important;
    color:#064e3b !important; font-weight:800 !important;
    border-radius:12px !important; width:100% !important;
}
[data-testid="stDataFrame"] * { color:#111827 !important; }
[data-testid="stDataFrame"] { border-radius:14px !important; overflow:hidden; }

/* ── Alertas ── */
.al { border-radius:12px; padding:0.7rem 1.1rem; margin:0.5rem 0;
      font-size:0.82rem; line-height:1.6; border-left:4px solid;
      font-weight:500; }
.a-ok   { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border-color:#22c55e; color:#166534; }
.a-warn { background:linear-gradient(135deg,#fffbeb,#fef3c7); border-color:#f59e0b; color:#92400e; }
.a-err  { background:linear-gradient(135deg,#fef2f2,#fee2e2); border-color:#f87171; color:#991b1b; }
.a-info { background:linear-gradient(135deg,#ecfdf5,#d1fae5); border-color:#34d399; color:#064e3b; }

/* ── Suma proximal ── */
.suma-ok  { background:linear-gradient(135deg,#064e3b,#059669); color:#fff;
            border-radius:10px; padding:6px 14px; font-size:0.8rem;
            font-weight:800; font-family:'JetBrains Mono',monospace;
            display:inline-block; box-shadow:0 3px 10px rgba(6,78,59,0.3); }
.suma-err { background:linear-gradient(135deg,#991b1b,#dc2626); color:#fff;
            border-radius:10px; padding:6px 14px; font-size:0.8rem;
            font-weight:800; font-family:'JetBrains Mono',monospace;
            display:inline-block; box-shadow:0 3px 10px rgba(220,38,38,0.3); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
RUTA = os.path.dirname(os.path.abspath(__file__))
COLS = ['AR_Moisturecontent','AR_Ashcontent','AR_Volatilematter',
        'AR_Fixedcarbon','AR_Netcalorificvalue(LHV)']
COLS_SUMA = ['AR_Moisturecontent','AR_Ashcontent','AR_Volatilematter','AR_Fixedcarbon']

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
    ('AR_Carbon',   'C', 'e-C', 'rc', '#10b981', 'Carbono (C)'),
    ('AR_Hydrogen', 'H', 'e-H', 'rh', '#0891b2', 'Hidrógeno (H)'),
    ('AR_Oxygen',   'O', 'e-O', 'ro', '#7c3aed', 'Oxígeno (O)'),
    ('AR_Nitrogen', 'N', 'e-N', 'rn', '#d97706', 'Nitrógeno (N)'),
    ('AR_Sulphur',  'S', 'e-S', 'rs', '#dc2626', 'Azufre (S)'),
]

# ── Descripciones de cluster ──
CLUSTER_DESC = {
    0: "🌱 Biomasa con características favorables para la producción de hidrógeno.",
    1: "⚡ Biomasa con alto potencial para la generación de energía termoquímica.",
    2: "♻️ Biomasa con propiedades adecuadas para distintas rutas de valorización.",
    3: "🍂 Biomasa con menor aptitud para la conversión termoquímica directa.",
}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
clf_path = os.path.join(RUTA, "modelo_final.pkl")
nn_path  = RUTA

with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:1rem 0 1.2rem;">
        <div style="font-size:2rem;">🌿</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1rem;
             color:#34d399;font-weight:800;margin-top:6px;">BiomassIQ</div>
        <div style="font-size:0.65rem;color:#065f46;margin-top:2px;
             letter-spacing:0.1em;text-transform:uppercase;">Elemental Predictor</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(52,211,153,0.1);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.6rem;color:#065f46;text-transform:uppercase;"
                "letter-spacing:0.12em;'>Rangos válidos (AR)</p>", unsafe_allow_html=True)
    for _,(nm,un,vmin,vmax,_d) in META.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
            f'border-bottom:1px solid rgba(52,211,153,0.08);font-size:0.7rem;">'
            f'<span style="color:#34d399;">{nm}</span>'
            f'<span style="color:#065f46;font-family:JetBrains Mono,monospace;">'
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

def validar_suma(vals_dict, idx):
    """Valida que Humedad+Cenizas+Mat.Volátil+C.Fijo ≈ 100%"""
    suma = sum(vals_dict[c] for c in COLS_SUMA)
    return suma, abs(suma - 100) <= 1.0

def chart_muestra(row, label):
    labels = [t[5] for t in ELEM_ORDER]
    vals   = [row[OUT_DISP[t[0]]] for t in ELEM_ORDER]
    colors = [t[4] for t in ELEM_ORDER]
    fig = go.Figure()
    for lbl,val,clr in zip(labels,vals,colors):
        fig.add_trace(go.Bar(
            name=lbl, x=[lbl], y=[val],
            marker_color=clr,
            marker_line=dict(color='rgba(255,255,255,0.3)', width=1),
            text=[f'{val:.1f}%'], textposition='outside',
            textfont=dict(size=11, color='#064e3b', family='JetBrains Mono'),
            width=0.6,
        ))
    fig.update_layout(
        barmode='group', showlegend=False, height=260,
        margin=dict(l=20,r=10,t=20,b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(240,253,244,0.5)',
        font=dict(family='Outfit', color='#374151', size=11),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=10, color='#374151')),
        yaxis=dict(gridcolor='#d1fae5', zeroline=False,
                   ticksuffix='%', tickfont=dict(size=9, color='#9ca3af')),
    )
    return fig

def render_cluster_info(cluster_val):
    """Renderiza badge + descripción de cluster"""
    desc = CLUSTER_DESC.get(int(cluster_val), f"Cluster {cluster_val}")
    st.markdown(
        f'<div style="text-align:center;padding:0.5rem 0;">'
        f'<div class="cluster-pill">Cluster {cluster_val}</div>'
        f'</div>'
        f'<div class="cluster-desc">{desc}</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-icon">🌿</div>
    <div>
        <p class="header-title">BiomassIQ Elemental Predictor</p>
        <p class="header-sub">Machine Learning · Análisis Elemental de Biomasa</p>
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
            f'muestra{"s" if n>1 else ""} y presiona <b>Clasificar Biomasa</b>.'
            f'<br><span style="font-size:0.75rem;opacity:0.8;">⚠️ La suma de Humedad + Cenizas + Mat. Volátil + C. Fijo debe ser 100%</span></div>',
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">📊 Datos de Entrada</p>', unsafe_allow_html=True)

    col_headers = st.columns([0.7] + [1]*5)
    col_headers[0].markdown("", unsafe_allow_html=True)
    meta_items = list(META.items())
    for hcol, (col_key,(nm,un,vmin,vmax,_)) in zip(col_headers[1:], meta_items):
        hcol.markdown(
            f'<div class="col-header">{nm}</div>'
            f'<div class="col-range">{un} [{vmin}–{vmax}]</div>',
            unsafe_allow_html=True)

    all_vals = []
    fuera_rango = []
    errores_suma = []

    for i in range(n):
        row_cols = st.columns([0.7] + [1]*5)
        row_cols[0].markdown(
            f'<div class="sample-label">M{i+1}</div>',
            unsafe_allow_html=True)
        vals = []
        vals_dict = {}
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
            vals_dict[col_key] = v
        all_vals.append(vals)

        # Validar suma proximal
        suma = sum(vals_dict[c] for c in COLS_SUMA)
        ok_suma = abs(suma - 100) <= 1.0
        if not ok_suma:
            errores_suma.append((i+1, suma))
        row_cols[0].markdown(
            f'<div style="margin-top:4px;">'
            f'<span class="{"suma-ok" if ok_suma else "suma-err"}">Σ={suma:.1f}%</span>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

    if fuera_rango:
        st.markdown(
            f'<div class="al a-warn">⚠ <b>Valores fuera del rango de entrenamiento:</b> '
            f'{" &nbsp;·&nbsp; ".join(fuera_rango)}</div>',
            unsafe_allow_html=True)

    if errores_suma:
        muestras_err = ", ".join([f"M{idx} (Σ={s:.1f}%)" for idx,s in errores_suma])
        st.markdown(
            f'<div class="al a-err">❌ <b>Error en suma proximal:</b> {muestras_err} '
            f'— La suma de Humedad + Cenizas + Mat. Volátil + C. Fijo debe ser 100% (±1%). '
            f'Verifica tus datos antes de continuar.</div>',
            unsafe_allow_html=True)

    cb1, cb2 = st.columns([1, 3])
    ejecutar = cb1.button("🔬  Clasificar Biomasa")
    st.markdown('</div>', unsafe_allow_html=True)

    if ejecutar:
        if errores_suma:
            st.markdown(
                f'<div class="al a-err">❌ No se puede predecir: corrige la suma proximal primero.</div>',
                unsafe_allow_html=True)
            st.stop()

        err = []
        if not os.path.exists(clf_path): err.append(f"Clasificador no encontrado: {clf_path}")
        if not os.path.isdir(nn_path):   err.append(f"Carpeta no encontrada: {nn_path}")
        for e in err:
            st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
        if err:
            st.stop()

        df_input = pd.DataFrame(all_vals, columns=COLS)
        with st.spinner("🌿 Analizando muestras con IA..."):
            try:
                df_res = predecir(df_input)
            except Exception as e:
                st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
                st.stop()

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">🎯 Resultados Predicción</p>', unsafe_allow_html=True)

        cols_ch = st.columns(len(df_res))
        for i, col in enumerate(cols_ch):
            row = df_res.iloc[i]
            cluster_val = row.get('Cluster', row.get('cluster','—'))
            with col:
                st.markdown(
                    f'<div style="text-align:center;margin-bottom:8px;">'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.75rem;'
                    f'font-weight:800;color:#059669;background:#d1fae5;'
                    f'padding:3px 10px;border-radius:6px;">MUESTRA {i+1}</span>'
                    f'</div>',
                    unsafe_allow_html=True)
                render_cluster_info(cluster_val)
                st.plotly_chart(chart_muestra(row, f"M{i+1}"), width='stretch')
                for key,sym,_,rcls,_,_ in ELEM_ORDER:
                    val = row[OUT_DISP[key]]
                    st.markdown(
                        f'<div class="res-row">'
                        f'<span class="res-sym {rcls}">{sym}:</span>'
                        f'<span class="res-val">{val:.1f}%</span>'
                        f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Descarga
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
    st.markdown('<p class="card-title">📂 Cargar archivo Excel</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="al a-info" style="margin-bottom:0.8rem;">Columnas requeridas: '
        '<code style="font-family:JetBrains Mono,monospace;font-size:0.76rem;'
        'background:#d1fae5;padding:2px 6px;border-radius:5px;color:#064e3b;">'
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

            # Validar suma proximal en Excel
            df_raw['_suma'] = df_raw[COLS_SUMA].sum(axis=1)
            bad_suma = df_raw[abs(df_raw['_suma'] - 100) > 1.0]
            df_raw = df_raw.drop(columns=['_suma'])

            # Validar rangos
            probs = []
            for ck,(nm,un,vmin,vmax,_) in META.items():
                bad = df_raw[(df_raw[ck]<vmin)|(df_raw[ck]>vmax)]
                if not bad.empty: probs.append(f"{nm}: {len(bad)} muestra(s)")

            if not bad_suma.empty:
                st.markdown(
                    f'<div class="al a-err">❌ <b>{len(bad_suma)} muestras con suma proximal ≠ 100%</b> '
                    f'(filas: {", ".join(str(x+2) for x in bad_suma.index[:10].tolist())}). '
                    f'Verifica que Humedad + Cenizas + Mat. Volátil + C. Fijo = 100.</div>',
                    unsafe_allow_html=True)

            if probs:
                st.markdown(
                    f'<div class="al a-warn">⚠ <b>{len(df_raw)} muestras cargadas.</b> '
                    f'Fuera de rango → {" · ".join(probs)}</div>', unsafe_allow_html=True)
            elif bad_suma.empty:
                st.markdown(
                    f'<div class="al a-ok">✅ <b>{len(df_raw)} muestras cargadas</b> '
                    f'— todos los valores dentro del rango y suma proximal correcta.</div>',
                    unsafe_allow_html=True)

            cb1, _ = st.columns([1, 3])
            if cb1.button("🔬  Predecir todas las muestras"):
                if not bad_suma.empty:
                    st.markdown(
                        '<div class="al a-err">❌ Corrige las muestras con suma proximal incorrecta antes de predecir.</div>',
                        unsafe_allow_html=True)
                    st.stop()

                err = []
                if not os.path.exists(clf_path): err.append("Clasificador no encontrado.")
                if not os.path.isdir(nn_path):   err.append("Carpeta no encontrada.")
                for e in err:
                    st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
                if err: st.stop()

                with st.spinner("🌿 Analizando todas las muestras con IA..."):
                    try:
                        df_res = predecir(df_raw)
                    except Exception as e:
                        st.markdown(f'<div class="al a-err">❌ {e}</div>', unsafe_allow_html=True)
                        st.stop()

                st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="card-title">🎯 Resultados — Todas las muestras</p>',
                            unsafe_allow_html=True)
                lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
                cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
                df_tabla  = df_res.rename(
                    columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}}
                )[cols_show].copy()
                df_tabla.index = [f"M{i+1}" for i in range(len(df_tabla))]

                # Añadir descripción de cluster
                df_tabla['Descripción'] = df_tabla['Cluster'].apply(
                    lambda c: CLUSTER_DESC.get(int(c), f"Cluster {c}"))

                st.dataframe(
                    df_tabla.style
                        .format(precision=3)
                        .set_properties(**{'color':'#064e3b','font-weight':'500'}),
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
