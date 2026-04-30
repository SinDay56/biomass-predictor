import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="BiomassIQ — Predictor Elemental",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# CSS — Estilo BioPredict: cálido, limpio, científico
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }

[data-testid="stToolbar"], [data-testid="stDecoration"],
header[data-testid="stHeader"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"] { display:none !important; }
.block-container { padding:0 !important; max-width:100% !important; }

/* ── FONDO ── */
.stApp { background:#F7F4EE; }

/* ── NAVBAR ── */
.navbar {
    background:#ffffff;
    border-bottom:2px solid #f0ebe0;
    padding:0 2.5rem;
    display:flex; align-items:center; justify-content:space-between;
    height:64px;
    box-shadow:0 2px 12px rgba(139,90,68,0.07);
    position:sticky; top:0; z-index:100;
}
.nav-logo {
    display:flex; align-items:center; gap:10px;
    font-size:1.25rem; font-weight:800; color:#3d2b1f;
    letter-spacing:-0.02em;
}
.nav-logo-icon {
    width:36px; height:36px;
    background:linear-gradient(135deg,#F6BE2C,#e8a820);
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.1rem;
    box-shadow:0 3px 10px rgba(246,190,44,0.3);
}
.nav-badge {
    background:linear-gradient(135deg,#F6BE2C,#e8a820);
    color:#3d2b1f; font-size:0.7rem; font-weight:700;
    padding:5px 14px; border-radius:20px;
    letter-spacing:0.03em; box-shadow:0 2px 8px rgba(246,190,44,0.3);
}

/* ── HERO ── */
.hero {
    background:linear-gradient(135deg,#3d2b1f 0%,#5c3d2e 50%,#8C5A44 100%);
    padding:3rem 2.5rem 2.5rem;
    display:grid; grid-template-columns:1fr auto;
    align-items:center; gap:2rem;
    position:relative; overflow:hidden;
}
.hero::before {
    content:'';
    position:absolute; top:-40px; right:-40px;
    width:320px; height:320px;
    border-radius:50%;
    background:radial-gradient(circle,rgba(246,190,44,0.15) 0%,transparent 70%);
}
.hero-title {
    font-size:2.2rem; font-weight:900; color:#fff;
    line-height:1.2; letter-spacing:-0.03em;
    margin:0 0 0.8rem 0;
}
.hero-title span { color:#F6BE2C; }
.hero-sub { font-size:0.95rem; color:rgba(255,255,255,0.75); margin:0; line-height:1.6; }
.hero-chips { display:flex; gap:10px; margin-top:1.2rem; flex-wrap:wrap; }
.hero-chip {
    background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);
    color:rgba(255,255,255,0.9); border-radius:20px;
    padding:5px 14px; font-size:0.75rem; font-weight:600;
    letter-spacing:0.03em;
}
.hero-sun { font-size:5rem; filter:drop-shadow(0 4px 16px rgba(246,190,44,0.5)); }

/* ── BODY ── */
.body-wrap { padding:2rem 2.5rem 3rem; }

/* ── SECTION LABEL ── */
.sec-label {
    display:flex; align-items:center; gap:8px;
    font-size:0.8rem; font-weight:700; color:#8C5A44;
    text-transform:uppercase; letter-spacing:0.1em;
    margin-bottom:0.9rem;
}
.sec-num {
    width:22px; height:22px; background:#F6BE2C; border-radius:6px;
    display:flex; align-items:center; justify-content:center;
    font-size:0.7rem; font-weight:800; color:#3d2b1f;
    flex-shrink:0;
}

/* ── CARDS ── */
.card {
    background:#ffffff; border-radius:18px; padding:1.6rem 1.8rem;
    box-shadow:0 2px 16px rgba(139,90,68,0.07), 0 1px 4px rgba(139,90,68,0.04);
    border:1px solid rgba(240,235,224,0.8);
    margin-bottom:1.2rem;
}
.card-sm { padding:1.2rem 1.4rem; }

/* ── INPUT LABELS ── */
.inp-label {
    font-size:0.8rem; font-weight:600; color:#4A4A4A;
    margin-bottom:4px; display:block;
}
.inp-unit { color:#8C5A44; font-size:0.72rem; font-weight:500; }

/* ── BOTÓN PRINCIPAL ── */
.stButton > button {
    background:linear-gradient(135deg,#F6BE2C,#e8a820) !important;
    color:#3d2b1f !important; border:none !important;
    border-radius:12px !important; font-family:'Inter',sans-serif !important;
    font-size:0.95rem !important; font-weight:800 !important;
    padding:0.85rem 1.5rem !important; width:100% !important;
    box-shadow:0 4px 16px rgba(246,190,44,0.35) !important;
    transition:all 0.25s !important; letter-spacing:0.01em !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#e8a820,#d4960f) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 22px rgba(246,190,44,0.45) !important;
}

/* ── INPUTS ── */
[data-testid="stNumberInput"] input {
    background:#faf8f4 !important; border:1.5px solid #e8e2d9 !important;
    border-radius:10px !important; color:#3d2b1f !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.95rem !important;
    font-weight:600 !important; padding:0.6rem 0.8rem !important;
    transition:all 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:#F6BE2C !important;
    box-shadow:0 0 0 3px rgba(246,190,44,0.2) !important;
    background:#fff !important;
}
[data-testid="stNumberInput"] button {
    background:#f0ebe0 !important; border:1px solid #e8e2d9 !important;
    color:#8C5A44 !important; border-radius:8px !important;
}
.stNumberInput label { color:#4A4A4A !important; font-size:0.82rem !important; font-weight:600 !important; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#fff !important; border-radius:12px !important;
    padding:4px !important; gap:4px !important;
    box-shadow:0 1px 8px rgba(139,90,68,0.08) !important;
    border:1px solid #f0ebe0 !important; margin-bottom:1.2rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'Inter',sans-serif !important; font-size:0.85rem !important;
    font-weight:600 !important; color:#6b6b6b !important;
    border-radius:9px !important; padding:7px 16px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,#F6BE2C,#e8a820) !important;
    color:#3d2b1f !important; font-weight:700 !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background:rgba(246,190,44,0.05) !important;
    border:2px dashed rgba(246,190,44,0.4) !important; border-radius:14px !important;
}
[data-testid="stDownloadButton"] > button {
    background:#fff !important; border:2px solid #F6BE2C !important;
    color:#3d2b1f !important; font-weight:700 !important;
    border-radius:12px !important; width:100% !important;
}
[data-testid="stDataFrame"] { border-radius:14px !important; overflow:hidden; }

/* ── ELEMENTO BADGES ── */
.elem-grid { display:flex; gap:10px; margin:1rem 0; }
.ebadge {
    flex:1; text-align:center; border-radius:14px; padding:16px 8px;
    border:2px solid; transition:transform 0.2s;
}
.ebadge:hover { transform:translateY(-3px); }
.eb-sym { font-size:1.1rem; font-weight:900; }
.eb-val { font-size:1rem; font-weight:800; font-family:'JetBrains Mono',monospace; margin-top:4px; }
.eb-name { font-size:0.62rem; font-weight:600; opacity:0.7; margin-top:2px; text-transform:uppercase; letter-spacing:0.05em; }
.e-C { border-color:#8FAE4A; color:#4a6a1a; background:linear-gradient(135deg,#f4f9ea,#e8f4d0); }
.e-H { border-color:#8C5A44; color:#5c3020; background:linear-gradient(135deg,#fdf4f0,#f9e4d8); }
.e-O { border-color:#8FAE4A; color:#3a5a10; background:linear-gradient(135deg,#f0f8e0,#dff0c0); }
.e-N { border-color:#9ca3af; color:#4b5563; background:linear-gradient(135deg,#f9fafb,#f3f4f6); }
.e-S { border-color:#9ca3af; color:#4b5563; background:linear-gradient(135deg,#f9fafb,#f3f4f6); }

/* ── GRUPO CARD ── */
.grupo-card {
    display:flex; gap:16px; align-items:flex-start;
    background:linear-gradient(135deg,#fffdf5,#fff9e6);
    border:1px solid #fde68a; border-left:5px solid #F6BE2C;
    border-radius:14px; padding:1.2rem 1.4rem;
    margin:1rem 0;
}
.grupo-num {
    width:48px; height:48px; flex-shrink:0;
    background:linear-gradient(135deg,#F6BE2C,#e8a820);
    border-radius:12px; display:flex; align-items:center; justify-content:center;
    font-size:1.4rem; font-weight:900; color:#3d2b1f;
    box-shadow:0 4px 12px rgba(246,190,44,0.35);
}
.grupo-content { flex:1; }
.grupo-title { font-size:0.95rem; font-weight:800; color:#3d2b1f; margin:0 0 4px 0; }
.grupo-desc { font-size:0.83rem; color:#4A4A4A; line-height:1.6; margin:0 0 8px 0; }
.grupo-examples { font-size:0.75rem; color:#8C5A44; font-style:italic; }
.grupo-examples span { font-weight:700; font-style:normal; }

/* ── RESUMEN 4 GRUPOS ── */
.grupos-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:0.5rem; }
.gcard {
    background:#fff; border-radius:14px; padding:1.1rem 1rem;
    border:1px solid #f0ebe0; border-top:4px solid;
    transition:transform 0.2s, box-shadow 0.2s;
}
.gcard:hover { transform:translateY(-3px); box-shadow:0 8px 24px rgba(139,90,68,0.1); }
.gcard-0 { border-top-color:#8FAE4A; }
.gcard-1 { border-top-color:#F6BE2C; }
.gcard-2 { border-top-color:#8C5A44; }
.gcard-3 { border-top-color:#9ca3af; }
.gcard-num {
    width:28px; height:28px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:0.85rem; font-weight:800; color:#fff; margin-bottom:8px;
}
.gn-0{background:#8FAE4A;} .gn-1{background:#F6BE2C;color:#3d2b1f !important;}
.gn-2{background:#8C5A44;} .gn-3{background:#9ca3af;}
.gcard-title { font-size:0.78rem; font-weight:800; color:#3d2b1f; margin:0 0 4px 0; }
.gcard-desc { font-size:0.7rem; color:#6b6b6b; line-height:1.5; margin:0 0 6px 0; }
.gcard-ex { font-size:0.65rem; color:#8C5A44; font-style:italic; }

/* ── SUMA BADGE ── */
.suma-ok { display:inline-flex; align-items:center; gap:5px;
    background:linear-gradient(135deg,#8FAE4A,#6d8f30); color:#fff;
    border-radius:8px; padding:5px 12px; font-size:0.78rem; font-weight:800;
    font-family:'JetBrains Mono',monospace; box-shadow:0 2px 8px rgba(143,174,74,0.3); }
.suma-err { display:inline-flex; align-items:center; gap:5px;
    background:linear-gradient(135deg,#dc2626,#b91c1c); color:#fff;
    border-radius:8px; padding:5px 12px; font-size:0.78rem; font-weight:800;
    font-family:'JetBrains Mono',monospace; box-shadow:0 2px 8px rgba(220,38,38,0.3); }

/* ── ALERTAS ── */
.al { border-radius:12px; padding:0.9rem 1.1rem; margin:0.6rem 0;
    font-size:0.84rem; line-height:1.6; border-left:4px solid; font-weight:500;
    display:flex; align-items:flex-start; gap:10px; }
.al-icon { font-size:1.1rem; line-height:1; flex-shrink:0; }
.a-ok   { background:#f0fdf4; border-color:#22c55e; color:#166534; }
.a-warn { background:#fffbeb; border-color:#f59e0b; color:#92400e; }
.a-err  { background:#fef2f2; border-color:#f87171; color:#991b1b; }
.a-info { background:#fffdf0; border-color:#F6BE2C; color:#3d2b1f; }

/* ── DIVIDER ── */
.divider { height:1px; background:linear-gradient(90deg,transparent,#e8e2d9 20%,#e8e2d9 80%,transparent); margin:1.5rem 0; }

/* ── STAT CHIPS ── */
.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin:0.8rem 0; }
.stat-chip {
    background:#fff; border:1px solid #f0ebe0; border-radius:12px;
    padding:10px 14px; text-align:center; flex:1; min-width:80px;
}
.stat-val { font-size:1rem; font-weight:800; color:#3d2b1f; font-family:'JetBrains Mono',monospace; }
.stat-lbl { font-size:0.65rem; font-weight:600; color:#8C5A44; text-transform:uppercase; letter-spacing:0.06em; }

/* ── RESPONSIVE ── */
@media (max-width:768px) {
    .grupos-grid { grid-template-columns:repeat(2,1fr); }
    .hero-title { font-size:1.5rem; }
    .body-wrap { padding:1rem; }
    .elem-grid { flex-wrap:wrap; }
}
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
    'AR_Moisturecontent':        ('Humedad',      '%',     0.10, 83.70, 8.30),
    'AR_Ashcontent':             ('Cenizas',       '%',     0.06, 64.11, 4.20),
    'AR_Volatilematter':         ('Mat. Volátil',  '%',     4.82, 85.78, 72.10),
    'AR_Fixedcarbon':            ('Carbono Fijo',  '%',     0.60, 87.70, 15.40),
    'AR_Netcalorificvalue(LHV)': ('LHV',           'MJ/kg', 0.19, 34.34, 18.50),
}
OUT_KEYS  = ['AR_Carbon','AR_Hydrogen','AR_Nitrogen','AR_Sulphur','AR_Oxygen']
OUT_DISP  = {'AR_Carbon':'C (%)','AR_Hydrogen':'H (%)','AR_Nitrogen':'N (%)',
             'AR_Sulphur':'S (%)','AR_Oxygen':'O (%)'}
OUT_NAMES = ['C','H','N','S','O']

ELEM_ORDER = [
    ('AR_Carbon',   'C', 'e-C', 'Carbono',    '#8FAE4A'),
    ('AR_Hydrogen', 'H', 'e-H', 'Hidrógeno',  '#8C5A44'),
    ('AR_Oxygen',   'O', 'e-O', 'Oxígeno',    '#6d8f30'),
    ('AR_Nitrogen', 'N', 'e-N', 'Nitrógeno',  '#9ca3af'),
    ('AR_Sulphur',  'S', 'e-S', 'Azufre',     '#d1d5db'),
]

GRUPOS = {
    0: {
        "icon": "🌿",
        "title": "Grupo 0 — Alto volátil y oxígeno",
        "desc": "Biomasa con alto contenido de material volátil y oxígeno. Favorece la producción de hidrógeno en procesos termoquímicos como la gasificación.",
        "examples": "Corteza, madera, bagazo, cáscaras de almendra, semillas de girasol, paja, alfalfa, cacao, cáñamo, sorgo, papel reciclado, pellets de miscanthus.",
        "summary": "Alto material volátil y oxígeno. Alto potencial de producción de hidrógeno.",
        "color": "#8FAE4A",
        "ex_short": "Madera, corteza, bagazo, sorgo, papel reciclado.",
    },
    1: {
        "icon": "⚡",
        "title": "Grupo 1 — Alto carbono fijo, bajo en humedad",
        "desc": "Biomasa con alto carbono fijo y baja humedad. Presenta alto poder calorífico y es óptima para combustión directa y generación de energía.",
        "examples": "Carbón, lodo seco, papel, paja carbonizada, estiércol tratado.",
        "summary": "Alto carbono fijo y bajo humedad. Alto potencial energético.",
        "color": "#F6BE2C",
        "ex_short": "Carbón, lodo seco, papel, paja carbonizada.",
    },
    2: {
        "icon": "♻️",
        "title": "Grupo 2 — Composición intermedia y balanceada",
        "desc": "Biomasa con composición equilibrada entre sus componentes. Útil para múltiples rutas de valorización incluyendo producción de biochar y biocombustibles.",
        "examples": "Madera, pino, semillas de girasol, flor de palma.",
        "summary": "Composición intermedia y balanceada. Útil para distintos procesos de biochar.",
        "color": "#8C5A44",
        "ex_short": "Madera, pino, semillas de girasol, flor de palma.",
    },
    3: {
        "icon": "🍂",
        "title": "Grupo 3 — Alta humedad, bajo carbono",
        "desc": "Biomasa con alta humedad y bajo contenido de carbono. Bajo poder calorífico y difícil ignición directa. Requiere pretratamiento o secado previo.",
        "examples": "Lodo de papel, lignina, cáscara de naranja, estiércol fresco, residuos húmedos.",
        "summary": "Alta humedad y bajo carbono. Bajo poder calorífico y difícil ignición.",
        "color": "#9ca3af",
        "ex_short": "Lodo de papel, lignina, cáscara de naranja, estiércol.",
    },
}

clf_path = os.path.join(RUTA, "modelo_final.pkl")
nn_path  = RUTA

# ══════════════════════════════════════════════════════════════════
# FUNCIONES ML
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
            y_true=tf.cast(y_true,tf.float32); y_pred=tf.cast(y_pred,tf.float32)
            self.sum_sq_res.assign_add(tf.reduce_sum(tf.square(y_true-y_pred),axis=0))
            self.sum_y.assign_add(tf.reduce_sum(y_true,axis=0))
            self.sum_y_sq.assign_add(tf.reduce_sum(tf.square(y_true),axis=0))
            self.count.assign_add(tf.cast(tf.shape(y_true)[0],tf.float32))
        def result(self):
            mu=self.sum_y/(self.count+1e-8)
            ss=self.sum_y_sq-self.count*tf.square(mu)
            return tf.reduce_mean(1.0-self.sum_sq_res/(ss+1e-8))
        def reset_state(self):
            for w in [self.sum_sq_res,self.sum_y,self.sum_y_sq]: w.assign(tf.zeros_like(w))
            self.count.assign(0.0)
        def get_config(self):
            c=super().get_config(); c.update({'n_outputs':self.n_outputs}); return c
        @classmethod
        def from_config(cls,cfg): return cls(**cfg)
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

def chart_barras(row):
    labels = [t[3] for t in ELEM_ORDER]
    vals   = [max(0, row[OUT_DISP[t[0]]]) for t in ELEM_ORDER]
    colors = [t[4] for t in ELEM_ORDER]
    syms   = [t[0].split('_')[1][0] for t in ELEM_ORDER]

    fig = go.Figure()
    for lbl,val,clr,sym in zip(labels,vals,colors,syms):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val],
            marker_color=clr,
            marker_line=dict(color='rgba(255,255,255,0.5)', width=1.5),
            text=[f'<b>{val:.1f}%</b>'],
            textposition='outside',
            textfont=dict(size=12, color='#3d2b1f', family='JetBrains Mono'),
            width=0.55,
            hovertemplate=f'<b>{lbl}</b><br>{val:.2f}%<extra></extra>',
        ))
    fig.update_layout(
        barmode='group', showlegend=False, height=260,
        margin=dict(l=10,r=10,t=30,b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,248,244,0.5)',
        font=dict(family='Inter', color='#4A4A4A', size=11),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color='#4A4A4A', family='Inter')),
        yaxis=dict(gridcolor='#f0ebe0', zeroline=False,
                   ticksuffix='%', tickfont=dict(size=9, color='#9ca3af'),
                   range=[0, max(vals)*1.25 if max(vals)>0 else 10]),
    )
    return fig

# ══════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-icon">🌾</div>
        BiomassIQ
    </div>
    <div class="nav-badge">🔬 AI-Powered</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div>
        <h1 class="hero-title">Predice el análisis<br>elemental de tu <span>biomasa</span></h1>
        <p class="hero-sub">Ingresa el poder calorífico y el análisis proximal para obtener una predicción precisa del análisis elemental mediante inteligencia artificial.</p>
        <div class="hero-chips">
            <span class="hero-chip">🧬 C · H · N · S · O</span>
            <span class="hero-chip">⚡ Resultado instantáneo</span>
            <span class="hero-chip">📊 4 grupos de biomasa</span>
        </div>
    </div>
    <div class="hero-sun">🌻</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="body-wrap">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS: Manual / Excel
# ══════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["✏️  Ingreso manual", "📂  Cargar Excel (>5 muestras)"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — INGRESO MANUAL
# ══════════════════════════════════════════════════════════════════
with tab1:
    n_col, _ = st.columns([1,4])
    with n_col:
        n = st.number_input("Número de muestras (máx. 5)", min_value=1, max_value=5, value=1, step=1)

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

    meta_items = list(META.items())
    all_vals = []
    fuera_rango = []
    errores_suma = []

    for i in range(n):
        if i > 0:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="sec-label"><div class="sec-num">{i+1}</div>MUESTRA {i+1}</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        # LHV separado arriba
        c_lhv, c_rest = st.columns([1, 3])
        with c_lhv:
            col_key, (nm, un, vmin, vmax, defval) = meta_items[4]
            st.markdown(f'<span class="inp-label">Poder calorífico <span class="inp-unit">(MJ/kg)</span></span>', unsafe_allow_html=True)
            v_lhv = st.number_input(f"LHV_M{i}", key=f"{col_key}_{i}",
                min_value=0.0, max_value=999.0, value=float(defval), step=0.01,
                label_visibility="collapsed")
            if v_lhv < vmin or v_lhv > vmax:
                st.markdown(f'<div style="font-size:0.62rem;color:#dc2626;">⚠ Rango: {vmin}–{vmax}</div>', unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.8rem;font-weight:700;color:#4A4A4A;margin:1rem 0 0.5rem;">Análisis proximal (base tal-como-recibido)</p>', unsafe_allow_html=True)

        # 4 columnas para análisis proximal
        proximal_cols = st.columns(4)
        vals_dict = {}
        vals = []

        for col_ui, (col_key,(nm,un,vmin,vmax,defval)) in zip(proximal_cols, meta_items[:4]):
            with col_ui:
                st.markdown(f'<span class="inp-label">{nm} <span class="inp-unit">(%)</span></span>', unsafe_allow_html=True)
                v = col_ui.number_input(f"{nm}_M{i}", key=f"{col_key}_{i}",
                    min_value=0.0, max_value=999.0, value=float(defval), step=0.01,
                    label_visibility="collapsed")
                if v < vmin or v > vmax:
                    st.markdown(f'<div style="font-size:0.62rem;color:#dc2626;">⚠ {vmin}–{vmax}</div>', unsafe_allow_html=True)
                    fuera_rango.append(f"M{i+1}·{nm}: {v:.2f}")
                vals_dict[col_key] = v
                vals.append(v)

        vals.append(v_lhv)
        vals_dict[meta_items[4][0]] = v_lhv

        suma = sum(vals_dict[c] for c in COLS_SUMA)
        ok_suma = abs(suma - 100) <= 1.0
        if not ok_suma:
            errores_suma.append((i+1, suma))

        st.markdown(
            f'<div style="margin-top:0.8rem;display:flex;align-items:center;gap:10px;">'
            f'<span class="{"suma-ok" if ok_suma else "suma-err"}">{"✓" if ok_suma else "✗"} Σ = {suma:.1f}%</span>'
            f'<span style="font-size:0.75rem;color:#8C5A44;">{"Suma proximal correcta ✓" if ok_suma else "La suma debe ser 100% ± 1%"}</span>'
            f'</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        all_vals.append(vals)

    # Alertas
    if fuera_rango:
        st.markdown(
            f'<div class="al a-warn"><span class="al-icon">⚠️</span>'
            f'<div><b>Valores fuera del rango:</b> {" · ".join(fuera_rango)}</div></div>',
            unsafe_allow_html=True)
    if errores_suma:
        muestras_err = ", ".join([f"M{idx} (Σ={s:.1f}%)" for idx,s in errores_suma])
        st.markdown(
            f'<div class="al a-err"><span class="al-icon">❌</span>'
            f'<div><b>Error suma proximal:</b> {muestras_err} — debe ser 100% (±1%)</div></div>',
            unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 3])
    ejecutar = btn_col.button("🔬  Calcular predicción")

    # ── RESULTADOS ──────────────────────────────────────────────
    if ejecutar:
        if errores_suma:
            st.markdown('<div class="al a-err"><span class="al-icon">🚫</span><div>Corrige la suma proximal antes de predecir.</div></div>', unsafe_allow_html=True)
            st.stop()

        df_input = pd.DataFrame(all_vals, columns=COLS)
        with st.spinner("Analizando con IA..."):
            try:
                df_res = predecir(df_input)
            except Exception as e:
                st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>{e}</div></div>', unsafe_allow_html=True)
                st.stop()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        for i in range(len(df_res)):
            row = df_res.iloc[i]
            cluster_val = int(row.get('Cluster', row.get('cluster', 0)))
            grupo = GRUPOS.get(cluster_val, GRUPOS[0])

            st.markdown(
                f'<div class="sec-label"><div class="sec-num">{i+1}</div>RESULTADOS — MUESTRA {i+1}</div>',
                unsafe_allow_html=True)

            left_col, right_col = st.columns([1, 1.4])

            with left_col:
                # Gráfico barras
                st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="sec-label" style="margin-bottom:0.5rem;">'
                    f'<div class="sec-num" style="background:#f0ebe0;color:#8C5A44;">2</div>'
                    f'Análisis elemental (base seca)</div>',
                    unsafe_allow_html=True)
                st.plotly_chart(chart_barras(row), use_container_width=True)

                # Badges
                badges_html = '<div class="elem-grid">'
                for key, sym, cls, name, color in ELEM_ORDER:
                    val = row[OUT_DISP[key]]
                    display = f"<0.1%" if val < 0.1 else f"{val:.1f}%"
                    badges_html += (
                        f'<div class="ebadge {cls}">'
                        f'<div class="eb-sym">{sym}</div>'
                        f'<div class="eb-val">{display}</div>'
                        f'<div class="eb-name">{name}</div>'
                        f'</div>'
                    )
                badges_html += '</div>'
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with right_col:
                # Grupo al que pertenece
                st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="sec-label" style="margin-bottom:0.5rem;">'
                    f'<div class="sec-num" style="background:#f0ebe0;color:#8C5A44;">3</div>'
                    f'Grupo al que pertenece</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="grupo-card">'
                    f'<div class="grupo-num" style="background:linear-gradient(135deg,{grupo["color"]},{grupo["color"]}cc);">{cluster_val}</div>'
                    f'<div class="grupo-content">'
                    f'<p class="grupo-title">{grupo["icon"]} Pertenece al {grupo["title"]}</p>'
                    f'<p class="grupo-desc">{grupo["desc"]}</p>'
                    f'<p class="grupo-examples"><span>Ejemplos representativos:</span> {grupo["examples"]}</p>'
                    f'</div></div>',
                    unsafe_allow_html=True)

                # Resumen de datos ingresados
                st.markdown(
                    f'<div class="sec-label" style="margin:1rem 0 0.5rem;">'
                    f'<div class="sec-num" style="background:#f0ebe0;color:#8C5A44;">4</div>'
                    f'Resumen del resultado</div>',
                    unsafe_allow_html=True)
                stat_html = '<div class="stat-row">'
                stat_items = [
                    (f"{row.get(OUT_DISP['AR_Carbon'],0):.1f}%", "Carbono (C)"),
                    (f"{row['AR_Netcalorificvalue(LHV)']:.2f}", "LHV (MJ/kg)"),
                    (f"{row['AR_Ashcontent']:.2f}%", "Cenizas"),
                    (f"{row['AR_Moisturecontent']:.2f}%", "Humedad"),
                ]
                for val, lbl in stat_items:
                    stat_html += f'<div class="stat-chip"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>'
                stat_html += '</div>'
                st.markdown(stat_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Descarga
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
        cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
        df_dl     = df_res.rename(columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}})[cols_show].copy()
        df_dl.index = [f"M{i+1}" for i in range(len(df_dl))]
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df_dl.to_excel(w, sheet_name='Resultados', index=True)
        dl_col, _ = st.columns([1, 3])
        dl_col.download_button("📥  Descargar resultados (Excel)",
            data=buf.getvalue(), file_name="predicciones_biomasa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── RESUMEN 4 GRUPOS ────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-label"><div class="sec-num">5</div>Resumen del grupo (4 grupos en total)</div>',
            unsafe_allow_html=True)
        grupos_html = '<div class="grupos-grid">'
        for gid, g in GRUPOS.items():
            activo = "box-shadow:0 0 0 3px " + g["color"] + ";" if gid == cluster_val else ""
            grupos_html += (
                f'<div class="gcard gcard-{gid}" style="{activo}">'
                f'<div class="gcard-num gn-{gid}">{g["icon"]}</div>'
                f'<p class="gcard-title">Grupo {gid}</p>'
                f'<p class="gcard-desc">{g["summary"]}</p>'
                f'<p class="gcard-ex">Ej: {g["ex_short"]}</p>'
                f'</div>'
            )
        grupos_html += '</div>'
        st.markdown(grupos_html, unsafe_allow_html=True)

        st.markdown(
            '<div style="margin-top:1.5rem;padding:1rem 1.2rem;background:#fffdf0;'
            'border-radius:12px;border:1px solid #fde68a;font-size:0.78rem;color:#8C5A44;">'
            '🌿 Estos valores son una predicción basada en modelos estadísticos entrenados '
            'con datos experimentales. Úsalos como referencia para la selección de procesos '
            'de conversión de biomasa.</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — EXCEL
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label"><div class="sec-num">1</div>Cargar archivo Excel</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="al a-info" style="margin-bottom:1rem;"><span class="al-icon">📋</span>'
        '<div><b>Columnas requeridas:</b><br>'
        '<code style="font-size:0.75rem;background:#fef3c7;padding:2px 6px;border-radius:5px;color:#78350f;">'
        'AR_Moisturecontent · AR_Ashcontent · AR_Volatilematter · AR_Fixedcarbon · AR_Netcalorificvalue(LHV)'
        '</code></div></div>', unsafe_allow_html=True)

    archivo = st.file_uploader("Seleccionar archivo Excel (.xlsx)",
                               type=["xlsx"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo:
        try:
            df_raw = pd.read_excel(archivo, engine='openpyxl')
            miss = [c for c in COLS if c not in df_raw.columns]
            if miss:
                st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>Columnas faltantes: {", ".join(miss)}</div></div>', unsafe_allow_html=True)
                st.stop()

            df_raw['_suma'] = df_raw[COLS_SUMA].sum(axis=1)
            bad_suma = df_raw[abs(df_raw['_suma'] - 100) > 1.0]
            df_raw = df_raw.drop(columns=['_suma'])

            probs = []
            for ck,(nm,un,vmin,vmax,_) in META.items():
                bad = df_raw[(df_raw[ck]<vmin)|(df_raw[ck]>vmax)]
                if not bad.empty: probs.append(f"{nm}: {len(bad)} fila(s)")

            if not bad_suma.empty:
                st.markdown(
                    f'<div class="al a-err"><span class="al-icon">❌</span>'
                    f'<div><b>{len(bad_suma)} muestras con suma proximal ≠ 100%.</b> '
                    f'Filas: {", ".join(str(x+2) for x in bad_suma.index[:8].tolist())}</div></div>',
                    unsafe_allow_html=True)

            if probs:
                st.markdown(f'<div class="al a-warn"><span class="al-icon">⚠️</span><div><b>{len(df_raw)} muestras cargadas.</b> Fuera de rango → {" · ".join(probs)}</div></div>', unsafe_allow_html=True)
            elif bad_suma.empty:
                st.markdown(f'<div class="al a-ok"><span class="al-icon">✅</span><div><b>{len(df_raw)} muestras cargadas</b> correctamente.</div></div>', unsafe_allow_html=True)

            cb1, _ = st.columns([1, 3])
            if cb1.button("🔬  Predecir todas las muestras"):
                if not bad_suma.empty:
                    st.markdown('<div class="al a-err"><span class="al-icon">🚫</span><div>Corrige la suma proximal antes de predecir.</div></div>', unsafe_allow_html=True)
                    st.stop()

                with st.spinner("Analizando todas las muestras..."):
                    try:
                        df_res = predecir(df_raw)
                    except Exception as e:
                        st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>{e}</div></div>', unsafe_allow_html=True)
                        st.stop()

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-label"><div class="sec-num">2</div>Resultados — todas las muestras</div>', unsafe_allow_html=True)

                lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
                cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
                df_tabla  = df_res.rename(
                    columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}}
                )[cols_show].copy()
                df_tabla.index = [f"M{i+1}" for i in range(len(df_tabla))]
                df_tabla['Descripción grupo'] = df_tabla['Cluster'].apply(
                    lambda c: GRUPOS.get(int(c), {}).get('summary','—'))

                st.dataframe(df_tabla.style.format(precision=2)
                    .set_properties(**{'color':'#3d2b1f','font-weight':'500'}),
                    use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    df_tabla.to_excel(w, sheet_name='Resultados', index=True)
                cd, _ = st.columns([1, 3])
                cd.download_button("📥  Descargar Excel completo",
                    data=buf.getvalue(), file_name="predicciones_biomasa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div><b>Error:</b> {e}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #e8e2d9;padding:1.5rem 2.5rem;
     display:flex;justify-content:space-between;align-items:center;
     background:#fff;margin-top:2rem;">
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:28px;height:28px;background:linear-gradient(135deg,#F6BE2C,#e8a820);
             border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">🌾</div>
        <span style="font-weight:800;color:#3d2b1f;font-size:0.9rem;">BiomassIQ</span>
        <span style="color:#9ca3af;font-size:0.75rem;">— Herramienta para la predicción del análisis elemental de biomasa</span>
    </div>
    <span style="font-size:0.72rem;color:#9ca3af;">Powered by Machine Learning · v3.0</span>
</div>
""", unsafe_allow_html=True)
