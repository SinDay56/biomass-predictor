import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, plotly.graph_objects as go
from io import BytesIO
import time

st.set_page_config(
    page_title="BiomassIQ — Elemental Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# CSS — Diseño mejorado con animaciones e interactividad
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif !important; }

/* ── ANIMACIONES ── */
@keyframes fadeInUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
@keyframes bounceIn { 0%{opacity:0;transform:scale(0.5)} 70%{transform:scale(1.04)} 100%{opacity:1;transform:scale(1)} }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }

/* ── FONDO CÁLIDO ── */
.stApp {
    background-color: #faf7f2;
    background-image:
        radial-gradient(circle at 15% 85%, rgba(251,191,36,0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(249,115,22,0.06) 0%, transparent 45%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23f59e0b' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

.block-container { padding:0 !important; max-width:100% !important; position:relative; z-index:1; }

/* ── Ocultar barra Streamlit ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"],
[data-testid="stStatusWidget"] { display:none !important; }

/* ── HEADER CÁLIDO ── */
.app-header {
    background: linear-gradient(135deg, #78350f 0%, #92400e 40%, #b45309 75%, #d97706 100%);
    padding: 1.8rem 2.5rem;
    display: flex; align-items:center; justify-content:center; gap:20px;
    box-shadow: 0 8px 32px rgba(120,53,15,0.25);
    border-bottom: 4px solid #fbbf24;
    position: relative; overflow: hidden;
}
.app-header::before {
    content:'';
    position:absolute; top:0; left:0; right:0; bottom:0;
    background: radial-gradient(ellipse at 70% 50%, rgba(251,191,36,0.15) 0%, transparent 60%);
    pointer-events:none;
}
.header-icon { font-size:3rem; line-height:1; animation: float 3s ease-in-out infinite; }
.header-content { text-align:center; }
.header-title {
    font-size:2rem !important; font-weight:800 !important;
    color:#fffbeb !important; margin:0 !important;
    letter-spacing:-0.02em; text-shadow:0 2px 12px rgba(0,0,0,0.2);
}
.header-sub {
    font-size:0.72rem; color:rgba(254,243,199,0.85); margin:0.3rem 0 0 0 !important;
    letter-spacing:0.18em; text-transform:uppercase; font-weight:600;
}
.header-badge {
    position:absolute; top:14px; right:20px;
    background:rgba(255,255,255,0.12); border:1px solid rgba(251,191,36,0.4);
    border-radius:20px; padding:5px 12px; font-size:0.68rem;
    color:#fef3c7; font-weight:700; letter-spacing:0.05em;
}

/* ── BODY ── */
.body-wrap { padding:2rem 2.5rem 3rem; animation:fadeInUp 0.7s ease-out; }

/* ── CARDS ── */
.card {
    background:#ffffff;
    border-radius:20px;
    padding:1.8rem 2rem;
    box-shadow: 0 2px 20px rgba(120,53,15,0.07), 0 1px 4px rgba(120,53,15,0.04);
    margin-bottom:1.5rem;
    border:1px solid rgba(251,191,36,0.15);
    border-top:4px solid #f59e0b;
    position:relative; overflow:hidden;
    transition:transform 0.3s, box-shadow 0.3s;
    animation:fadeInUp 0.6s ease-out backwards;
}
.card:hover { transform:translateY(-3px); box-shadow:0 8px 32px rgba(120,53,15,0.1); }
.card::after {
    content:'';
    position:absolute; top:0; right:0;
    width:140px; height:140px;
    background:radial-gradient(circle, rgba(251,191,36,0.06) 0%, transparent 70%);
    pointer-events:none;
}

.card-title {
    font-size:1.1rem !important; font-weight:800 !important;
    color:#78350f !important; margin:0 0 1.4rem 0 !important;
    padding-bottom:0.9rem; border-bottom:2px solid #fef3c7;
    letter-spacing:-0.01em; display:flex; align-items:center; gap:10px;
}
.card-title-icon {
    background:linear-gradient(135deg,#92400e,#d97706);
    color:#fff; width:34px; height:34px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; box-shadow:0 4px 12px rgba(146,64,14,0.3);
    flex-shrink:0;
}

/* ── INPUTS ── */
.col-header { font-size:0.7rem; font-weight:800; color:#92400e;
    text-transform:uppercase; letter-spacing:0.1em; text-align:center; padding-bottom:5px; }
.col-range { font-size:0.58rem; color:#fff; font-family:'DM Mono',monospace;
    text-align:center; margin-bottom:7px; background:linear-gradient(135deg,#92400e,#d97706);
    padding:3px 7px; border-radius:5px; display:inline-block; }
.col-unit { font-size:0.56rem; color:#d97706; font-family:'DM Mono',monospace;
    text-align:center; display:block; margin-top:2px; }
.sample-label { font-size:0.88rem; font-weight:800; color:#fff; font-family:'DM Mono',monospace;
    padding:7px 11px; background:linear-gradient(135deg,#b45309,#f59e0b);
    border-radius:8px; display:inline-block; margin-bottom:7px;
    box-shadow:0 3px 10px rgba(180,83,9,0.3); }
.oor-hint { font-size:0.58rem; color:#dc2626; text-align:center;
    font-family:'DM Mono',monospace; margin-top:3px; }

/* ── CLUSTER ── */
.cluster-section { text-align:center; padding:0.8rem 0; }
.cluster-pill {
    display:inline-flex; align-items:center; gap:10px;
    background:linear-gradient(135deg,#78350f,#d97706);
    color:#fff; border-radius:50px; padding:10px 22px;
    font-size:1rem; font-weight:800;
    box-shadow:0 6px 18px rgba(120,53,15,0.35), 0 0 0 3px rgba(251,191,36,0.2);
    animation:bounceIn 0.6s ease-out;
}
.cluster-pill-icon { font-size:1.2rem; }
.cluster-desc {
    background:linear-gradient(135deg,#fffbeb,#fef3c7);
    border:1px solid #fde68a; border-left:4px solid #f59e0b;
    border-radius:12px; padding:12px 16px;
    font-size:0.85rem; color:#78350f; font-weight:500;
    margin-top:12px; line-height:1.6;
    animation:fadeInUp 0.5s ease-out 0.2s backwards;
}

/* ── BADGES ELEMENTOS ── */
.elem-row-wrap { display:flex; gap:8px; margin-top:0.8rem; }
.ebadge { flex:1; border:2px solid; border-radius:14px; padding:12px 6px;
    text-align:center; transition:transform 0.25s, box-shadow 0.25s; cursor:default; }
.ebadge:hover { transform:translateY(-5px); box-shadow:0 8px 20px rgba(0,0,0,0.09); }
.eb-sym { font-size:1.2rem; font-weight:900; line-height:1; margin-bottom:3px; }
.eb-pct { font-size:0.85rem; font-weight:700; font-family:'DM Mono',monospace; }
.e-C { border-color:#d97706; color:#78350f; background:linear-gradient(135deg,#fffbeb,#fef3c7); }
.e-H { border-color:#0891b2; color:#0e4f6b; background:linear-gradient(135deg,#ecfeff,#cffafe); }
.e-O { border-color:#7c3aed; color:#4c1d95; background:linear-gradient(135deg,#f5f3ff,#ede9fe); }
.e-N { border-color:#059669; color:#064e3b; background:linear-gradient(135deg,#ecfdf5,#d1fae5); }
.e-S { border-color:#dc2626; color:#991b1b; background:linear-gradient(135deg,#fef2f2,#fee2e2); }

/* ── RESULT ROWS ── */
.res-row { display:flex; align-items:center; gap:12px; padding:9px 0;
    border-bottom:1px solid #fef3c7; transition:all 0.2s; }
.res-row:hover { background:rgba(254,243,199,0.4); padding-left:8px; border-radius:8px; }
.res-row:last-child { border-bottom:none; }
.res-sym { font-size:1rem; font-weight:900; min-width:22px; }
.res-val { color:#1c1917; font-family:'DM Mono',monospace; font-size:0.95rem; font-weight:700; }
.rc{color:#d97706;} .rh{color:#0891b2;} .ro{color:#7c3aed;} .rn{color:#059669;} .rs{color:#dc2626;}

/* ── SUMA ── */
.suma-ok { background:linear-gradient(135deg,#78350f,#d97706); color:#fff;
    border-radius:10px; padding:7px 13px; font-size:0.82rem; font-weight:800;
    font-family:'DM Mono',monospace; display:inline-flex; align-items:center; gap:5px;
    box-shadow:0 3px 12px rgba(120,53,15,0.3); animation:bounceIn 0.5s ease-out; }
.suma-err { background:linear-gradient(135deg,#991b1b,#dc2626); color:#fff;
    border-radius:10px; padding:7px 13px; font-size:0.82rem; font-weight:800;
    font-family:'DM Mono',monospace; display:inline-flex; align-items:center; gap:5px;
    box-shadow:0 3px 12px rgba(220,38,38,0.3); animation:pulse 1s ease-in-out infinite; }

/* ── BOTONES ── */
.stButton > button {
    background:linear-gradient(135deg,#92400e,#d97706) !important;
    color:#fff !important; border:none !important; border-radius:14px !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
    font-weight:800 !important; padding:0.85rem 1.5rem !important;
    width:100% !important; letter-spacing:0.01em !important;
    box-shadow:0 6px 20px rgba(146,64,14,0.35) !important;
    transition:all 0.3s !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#b45309,#fbbf24) !important;
    transform:translateY(-3px) !important;
    box-shadow:0 10px 28px rgba(146,64,14,0.45) !important;
}

/* ── NUMBER INPUTS ── */
[data-testid="stNumberInput"] input {
    background:#fffbeb !important; border:2px solid #fde68a !important;
    border-radius:12px !important; color:#78350f !important;
    font-family:'DM Mono',monospace !important; font-size:0.95rem !important;
    font-weight:700 !important; text-align:center !important;
    transition:all 0.25s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:#f59e0b !important;
    box-shadow:0 0 0 4px rgba(245,158,11,0.15) !important;
    background:#fff !important;
}
[data-testid="stNumberInput"] button {
    background:#fef3c7 !important; border:1px solid #fde68a !important;
    color:#92400e !important; border-radius:8px !important;
}
.stNumberInput label { color:#57534e !important; font-size:0.78rem !important; font-weight:700 !important; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#fff !important; border-radius:14px !important; padding:5px !important;
    gap:5px !important; box-shadow:0 2px 12px rgba(120,53,15,0.07) !important;
    border:1px solid rgba(251,191,36,0.2) !important; margin-bottom:1.5rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'DM Sans',sans-serif !important; font-size:0.88rem !important;
    font-weight:700 !important; color:#78716c !important;
    border-radius:10px !important; padding:8px 18px !important; transition:all 0.25s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,#92400e,#d97706) !important; color:#fff !important;
    box-shadow:0 4px 14px rgba(146,64,14,0.3) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#1c1412 0%,#292013 100%) !important; }
[data-testid="stSidebar"] * { color:#fde68a !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background:rgba(245,158,11,0.06) !important;
    border:2px dashed rgba(251,191,36,0.4) !important; border-radius:16px !important;
}
[data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg,#fffbeb,#fef3c7) !important;
    border:2px solid #d97706 !important; color:#78350f !important;
    font-weight:800 !important; border-radius:14px !important; width:100% !important;
}
[data-testid="stDataFrame"] * { color:#1c1917 !important; }
[data-testid="stDataFrame"] { border-radius:16px !important; overflow:hidden; }

/* ── ALERTAS ── */
.al { border-radius:14px; padding:1rem 1.2rem; margin:0.7rem 0;
    font-size:0.86rem; line-height:1.7; border-left:4px solid; font-weight:500;
    display:flex; align-items:flex-start; gap:12px; animation:fadeInUp 0.4s ease-out; }
.al-icon { font-size:1.2rem; line-height:1; }
.a-ok   { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border-color:#22c55e; color:#166534; }
.a-warn { background:linear-gradient(135deg,#fffbeb,#fef3c7); border-color:#f59e0b; color:#92400e; }
.a-err  { background:linear-gradient(135deg,#fef2f2,#fee2e2); border-color:#f87171; color:#991b1b; }
.a-info { background:linear-gradient(135deg,#fffbeb,#fef3c7); border-color:#fbbf24; color:#78350f; }

/* ── PROGRESS ── */
.progress-wrap { background:#fff; border-radius:16px; padding:1.5rem;
    text-align:center; margin:1rem 0; box-shadow:0 2px 12px rgba(120,53,15,0.07); }
.progress-icon { font-size:2.5rem; animation:float 2s ease-in-out infinite; }
.progress-text { font-size:0.95rem; color:#92400e; font-weight:700; margin-top:0.5rem; }

/* ── DIVIDER ── */
.sample-divider { height:2px;
    background:linear-gradient(90deg,transparent,#fde68a 20%,#fde68a 80%,transparent);
    margin:1.8rem 0; position:relative; }
.sample-divider::after { content:'◆'; position:absolute; left:50%; top:50%;
    transform:translate(-50%,-50%); background:#faf7f2;
    padding:0 10px; color:#fde68a; font-size:0.75rem; }

/* ── SAMPLE HEADER / BADGE ── */
.sample-header { text-align:center; margin-bottom:1rem; padding-bottom:1rem;
    border-bottom:2px solid #fef3c7; }
.sample-badge { display:inline-block; background:linear-gradient(135deg,#b45309,#f59e0b);
    color:#fff; padding:7px 18px; border-radius:25px; font-family:'DM Mono',monospace;
    font-size:0.82rem; font-weight:800; box-shadow:0 4px 14px rgba(180,83,9,0.3); }

/* ── RESPONSIVE ── */
@media (max-width:768px) {
    .app-header { padding:1.4rem; } .header-title { font-size:1.5rem !important; }
    .body-wrap { padding:1rem; } .card { padding:1.2rem; }
    .elem-row-wrap { flex-direction:column; }
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

CLUSTER_DESC = {
    0: "🌱 Biomasa con características favorables para la producción de hidrógeno. Alta eficiencia en procesos de gasificación.",
    1: "⚡ Biomasa con alto potencial para la generación de energía termoquímica. Óptima para combustión directa.",
    2: "♻️ Biomasa con propiedades equilibradas para múltiples rutas de valorización energética.",
    3: "🍂 Biomasa con menor aptitud para conversión termoquímica directa. Requiere pretratamiento.",
}

CLUSTER_ICONS = {
    0: "🌱",
    1: "⚡", 
    2: "♻️",
    3: "🍂"
}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
clf_path = os.path.join(RUTA, "modelo_final.pkl")
nn_path  = RUTA

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 2rem;">
        <div style="font-size:3rem; animation: float 3s ease-in-out infinite;">🌾</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:1.3rem;
             color:#fbbf24; font-weight:800; margin-top:10px;">BiomassIQ</div>
        <div style="font-size:0.7rem; color:#92400e; margin-top:4px;
             letter-spacing:0.15em; text-transform:uppercase;">Elemental Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color:rgba(52,211,153,0.15); margin:0.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding:1rem 0;">
        <div style="font-size:0.65rem; color:#34d399; text-transform:uppercase;
             letter-spacing:0.12em; margin-bottom:0.8rem; font-weight:700;">
            📊 Rangos válidos (AR)
        </div>
    """, unsafe_allow_html=True)
    
    for _,(nm,un,vmin,vmax,_d) in META.items():
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; padding:8px 0;'
            f'border-bottom:1px solid rgba(52,211,153,0.1); font-size:0.75rem;">'
            f'<span style="color:#6ee7b7; font-weight:600;">{nm}</span>'
            f'<span style="color:#a7f3d0; font-family:JetBrains Mono,monospace; font-size:0.7rem;">'
            f'{vmin} – {vmax} {un}</span></div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color:rgba(52,211,153,0.15); margin:1rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding:0.5rem 0; text-align:center;">
        <div style="font-size:0.6rem; color:#065f46; text-transform:uppercase;
             letter-spacing:0.1em; margin-bottom:0.5rem;">Versión</div>
        <div style="font-size:0.75rem; color:#34d399; font-weight:700;">v2.0 - Enhanced UI</div>
    </div>
    """, unsafe_allow_html=True)

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
    labels = [t[5] for t in ELEM_ORDER]
    vals   = [row[OUT_DISP[t[0]]] for t in ELEM_ORDER]
    colors = [t[4] for t in ELEM_ORDER]
    
    fig = go.Figure()
    
    for lbl,val,clr in zip(labels,vals,colors):
        fig.add_trace(go.Bar(
            name=lbl, x=[lbl], y=[val],
            marker_color=clr,
            marker_line=dict(color='rgba(255,255,255,0.4)', width=2),
            text=[f'<b>{val:.2f}%</b>'], 
            textposition='outside',
            textfont=dict(size=13, color='#064e3b', family='JetBrains Mono'),
            width=0.65,
            hovertemplate=f'<b>{lbl}</b><br>Valor: {val:.2f}%<extra></extra>',
        ))
    
    fig.update_layout(
        barmode='group', 
        showlegend=False, 
        height=280,
        margin=dict(l=25,r=15,t=25,b=65),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(255,251,235,0.3)',
        font=dict(family='DM Sans', color='#57534e', size=12),
        xaxis=dict(
            showgrid=False, 
            zeroline=False,
            tickfont=dict(size=11, color='#374151', family='Outfit')
        ),
        yaxis=dict(
            gridcolor='rgba(253,230,138,0.5)', 
            zeroline=False,
            ticksuffix='%', 
            tickfont=dict(size=10, color='#9ca3af'),
            title=dict(text='Composición (%)', font=dict(size=10, color='#6b7280'))
        ),
    )
    
    return fig

def render_cluster_info(cluster_val):
    icon = CLUSTER_ICONS.get(int(cluster_val), "🔬")
    desc = CLUSTER_DESC.get(int(cluster_val), f"Cluster {cluster_val}")
    
    st.markdown(
        f'<div class="cluster-section">'
        f'<div class="cluster-pill">'
        f'<span class="cluster-pill-icon">{icon}</span>'
        f'<span>Cluster {cluster_val}</span>'
        f'</div>'
        f'</div>'
        f'<div class="cluster-desc">{desc}</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-icon">🌾</div>
    <div class="header-content">
        <p class="header-title">BiomassIQ Elemental Predictor</p>
        <p class="header-sub">Machine Learning · Análisis Elemental de Biomasa</p>
    </div>
    <div class="header-badge">🔬 AI-Powered</div>
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
            f'<div class="al a-ok" style="margin-top:1.5rem;">'
            f'<span class="al-icon">✏️</span>'
            f'<div>Ingresa los datos de <b>{n} muestra{"s" if n>1 else ""}</b> y presiona '
            f'<b>Clasificar Biomasa</b>.<br>'
            f'<span style="font-size:0.8rem; opacity:0.85;">'
            f'⚠️ La suma de Humedad + Cenizas + Mat. Volátil + C. Fijo debe ser 100%</span></div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="al a-info" style="margin-top:1.5rem;">'
            '<span class="al-icon">📂</span>'
            '<div>Para más de <b>5 muestras</b>, carga un archivo Excel con los datos.</div></div>',
            unsafe_allow_html=True)

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FLUJO MANUAL ≤ 5 muestras
# ══════════════════════════════════════════════════════════════════
if n <= 5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        <p class="card-title">
            <span class="card-title-icon">📊</span>
            Datos de Entrada
        </p>
    """, unsafe_allow_html=True)

    col_headers = st.columns([0.7] + [1]*5)
    col_headers[0].markdown("", unsafe_allow_html=True)
    meta_items = list(META.items())
    
    for hcol, (col_key,(nm,un,vmin,vmax,_)) in zip(col_headers[1:], meta_items):
        hcol.markdown(
            f'<div class="col-header">{nm}</div>'
            f'<div style="text-align:center;"><span class="col-range">{vmin}–{vmax}</span></div>'
            f'<div class="col-unit">{un}</div>',
            unsafe_allow_html=True)

    all_vals = []
    fuera_rango = []
    errores_suma = []

    for i in range(n):
        if i > 0:
            st.markdown('<div class="sample-divider"></div>', unsafe_allow_html=True)
        
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
                    f'<div class="oor-hint">⚠️ Fuera de rango</div>',
                    unsafe_allow_html=True)
                fuera_rango.append(f"M{i+1}·{nm}: {v:.2f}")
            
            vals.append(v)
            vals_dict[col_key] = v
        
        all_vals.append(vals)

        suma = sum(vals_dict[c] for c in COLS_SUMA)
        ok_suma = abs(suma - 100) <= 1.0
        
        if not ok_suma:
            errores_suma.append((i+1, suma))
        
        row_cols[0].markdown(
            f'<div style="margin-top:8px;">'
            f'<span class="{"suma-ok" if ok_suma else "suma-err"}">'
            f'{"✓" if ok_suma else "✗"} Σ={suma:.1f}%</span>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    if fuera_rango:
        st.markdown(
            f'<div class="al a-warn">'
            f'<span class="al-icon">⚠️</span>'
            f'<div><b>Valores fuera del rango de entrenamiento:</b><br>'
            f'{" · ".join(fuera_rango)}</div></div>',
            unsafe_allow_html=True)

    if errores_suma:
        muestras_err = ", ".join([f"M{idx} (Σ={s:.1f}%)" for idx,s in errores_suma])
        st.markdown(
            f'<div class="al a-err">'
            f'<span class="al-icon">❌</span>'
            f'<div><b>Error en suma proximal:</b> {muestras_err}<br>'
            f'La suma debe ser 100% (±1%). Verifica tus datos.</div></div>',
            unsafe_allow_html=True)

    cb1, cb2 = st.columns([1, 3])
    ejecutar = cb1.button("🔬  Clasificar Biomasa")
    st.markdown('</div>', unsafe_allow_html=True)

    if ejecutar:
        if errores_suma:
            st.markdown(
                '<div class="al a-err">'
                '<span class="al-icon">🚫</span>'
                '<div>No se puede predecir: <b>corrige la suma proximal</b> primero.</div></div>',
                unsafe_allow_html=True)
            st.stop()

        err = []
        if not os.path.exists(clf_path): err.append(f"Clasificador no encontrado: {clf_path}")
        if not os.path.isdir(nn_path):   err.append(f"Carpeta no encontrada: {nn_path}")
        
        for e in err:
            st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>{e}</div></div>', 
                       unsafe_allow_html=True)
        if err:
            st.stop()

        df_input = pd.DataFrame(all_vals, columns=COLS)
        
        # Progress indicator mejorado
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = ["Cargando modelo de clasificación...", 
                 "Procesando datos de entrada...",
                 "Ejecutando predicción con IA...",
                 "Generando resultados..."]
        
        for i, step in enumerate(steps):
            icono = '🔄' if i < 3 else '✅'
            status_text.markdown(
                f'<div class="progress-wrap">'
                f'<div class="progress-icon">{icono}</div>'
                f'<div class="progress-text">{step}</div></div>',
                unsafe_allow_html=True)
            progress_bar.progress((i + 1) * 25)
            time.sleep(0.3)
        
        try:
            df_res = predecir(df_input)
        except Exception as e:
            st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>{e}</div></div>', 
                       unsafe_allow_html=True)
            st.stop()
        
        progress_bar.empty()
        status_text.empty()

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
            <p class="card-title">
                <span class="card-title-icon">🎯</span>
                Resultados de Predicción
            </p>
        """, unsafe_allow_html=True)

        cols_ch = st.columns(len(df_res))
        
        for i, col in enumerate(cols_ch):
            row = df_res.iloc[i]
            cluster_val = row.get('Cluster', row.get('cluster','—'))
            
            with col:
                st.markdown(
                    f'<div class="sample-header">'
                    f'<span class="sample-badge">MUESTRA {i+1}</span>'
                    f'</div>',
                    unsafe_allow_html=True)
                
                render_cluster_info(cluster_val)
                st.plotly_chart(chart_muestra(row, f"M{i+1}"), width='stretch')
                
                st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
                for key,sym,_,rcls,_,_ in ELEM_ORDER:
                    val = row[OUT_DISP[key]]
                    st.markdown(
                        f'<div class="res-row">'
                        f'<span class="res-sym {rcls}">{sym}</span>'
                        f'<span class="res-val">{val:.2f}%</span>'
                        f'</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Descarga
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
        cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
        df_dl     = df_res.rename(columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}})[cols_show].copy()
        df_dl.index = [f"M{i+1}" for i in range(len(df_dl))]
        
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df_dl.to_excel(w, sheet_name='Resultados', index=True)
        
        cd, _ = st.columns([1, 4])
        cd.download_button(
            "📥  Descargar Resultados (Excel)",
            data=buf.getvalue(),
            file_name="predicciones_biomasa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════
# FLUJO EXCEL > 5 muestras
# ══════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        <p class="card-title">
            <span class="card-title-icon">📂</span>
            Cargar Archivo Excel
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown(
        '<div class="al a-info" style="margin-bottom:1rem;">'
        '<span class="al-icon">📋</span>'
        '<div><b>Columnas requeridas:</b><br>'
        '<code style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
        'background:#d1fae5;padding:3px 8px;border-radius:6px;color:#064e3b;">'
        'AR_Moisturecontent · AR_Ashcontent · AR_Volatilematter · '
        'AR_Fixedcarbon · AR_Netcalorificvalue(LHV)</code></div></div>',
        unsafe_allow_html=True)

    archivo = st.file_uploader("Seleccionar archivo Excel (.xlsx)",
                               type=["xlsx"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo:
        try:
            df_raw = pd.read_excel(archivo, engine='openpyxl')
            miss   = [c for c in COLS if c not in df_raw.columns]
            
            if miss:
                st.markdown(
                    f'<div class="al a-err"><span class="al-icon">❌</span>'
                    f'<div><b>Columnas faltantes:</b> {", ".join(miss)}</div></div>',
                    unsafe_allow_html=True)
                st.stop()

            df_raw['_suma'] = df_raw[COLS_SUMA].sum(axis=1)
            bad_suma = df_raw[abs(df_raw['_suma'] - 100) > 1.0]
            df_raw = df_raw.drop(columns=['_suma'])

            probs = []
            for ck,(nm,un,vmin,vmax,_) in META.items():
                bad = df_raw[(df_raw[ck]<vmin)|(df_raw[ck]>vmax)]
                if not bad.empty: probs.append(f"{nm}: {len(bad)} muestra(s)")

            if not bad_suma.empty:
                st.markdown(
                    f'<div class="al a-err"><span class="al-icon">❌</span>'
                    f'<div><b>{len(bad_suma)} muestras con suma proximal ≠ 100%</b><br>'
                    f'Filas: {", ".join(str(x+2) for x in bad_suma.index[:10].tolist())}...</div></div>',
                    unsafe_allow_html=True)

            if probs:
                st.markdown(
                    f'<div class="al a-warn"><span class="al-icon">⚠️</span>'
                    f'<div><b>{len(df_raw)} muestras cargadas.</b><br>'
                    f'Fuera de rango → {" · ".join(probs)}</div></div>', 
                    unsafe_allow_html=True)
            elif bad_suma.empty:
                st.markdown(
                    f'<div class="al a-ok"><span class="al-icon">✅</span>'
                    f'<div><b>{len(df_raw)} muestras cargadas</b> correctamente.<br>'
                    f'Todos los valores dentro del rango y suma proximal correcta.</div></div>',
                    unsafe_allow_html=True)

            cb1, _ = st.columns([1, 3])
            if cb1.button("🔬  Predecir todas las muestras"):
                if not bad_suma.empty:
                    st.markdown(
                        '<div class="al a-err"><span class="al-icon">🚫</span>'
                        '<div>Corrige las muestras con suma proximal incorrecta antes de predecir.</div></div>',
                        unsafe_allow_html=True)
                    st.stop()

                err = []
                if not os.path.exists(clf_path): err.append("Clasificador no encontrado.")
                if not os.path.isdir(nn_path):   err.append("Carpeta no encontrada.")
                
                for e in err:
                    st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>{e}</div></div>',
                               unsafe_allow_html=True)
                if err: st.stop()

                # Progress para batch
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(4):
                    icono2 = '🔄' if i < 3 else '✅'
                    status_text.markdown(
                        f'<div class="progress-wrap">'
                        f'<div class="progress-icon">{icono2}</div>'
                        f'<div class="progress-text">Procesando {len(df_raw)} muestras...</div></div>',
                        unsafe_allow_html=True)
                    progress_bar.progress((i + 1) * 25)
                    time.sleep(0.2)

                try:
                    df_res = predecir(df_raw)
                except Exception as e:
                    st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div>{e}</div></div>',
                               unsafe_allow_html=True)
                    st.stop()
                
                progress_bar.empty()
                status_text.empty()

                st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("""
                    <p class="card-title">
                        <span class="card-title-icon">🎯</span>
                        Resultados — Todas las Muestras
                    </p>
                """, unsafe_allow_html=True)
                
                lbl_map   = {k:nm for k,(nm,un,_,_,_) in META.items()}
                cols_show = list(lbl_map.values())+['Cluster']+list(OUT_DISP.values())
                df_tabla  = df_res.rename(
                    columns={**lbl_map,**{k:OUT_DISP[k] for k in OUT_KEYS}}
                )[cols_show].copy()
                df_tabla.index = [f"M{i+1}" for i in range(len(df_tabla))]

                df_tabla['Descripción'] = df_tabla['Cluster'].apply(
                    lambda c: CLUSTER_DESC.get(int(c), f"Cluster {c}"))

                st.dataframe(
                    df_tabla.style
                        .format(precision=3)
                        .set_properties(**{'color':'#064e3b','font-weight':'500'}),
                    width='stretch', 
                    use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    df_tabla.to_excel(w, sheet_name='Resultados', index=True)
                
                cd, _ = st.columns([1, 3])
                cd.download_button(
                    "📥  Descargar Excel Completo", 
                    data=buf.getvalue(),
                    file_name="predicciones_biomasa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.markdown(f'<div class="al a-err"><span class="al-icon">❌</span><div><b>Error:</b> {e}</div></div>',
                       unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)