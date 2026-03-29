"""
================================================================================
AQUAGUARD REAL-TIME DASHBOARD — FINAL STABLE VERSION
Flask + Colab Built-in Tunnel (No ngrok, No Gradio)
Change The Future Team | Amman Arab University | IEEE RAS & CS Hackathon
================================================================================
HOW TO RUN IN GOOGLE COLAB:
  Cell 1: !pip install -q flask scikit-learn pandas numpy joblib
  Cell 2: exec(open('/content/data/aquaguard_final.py').read())
================================================================================
"""

import subprocess
subprocess.run(["pip","install","-q","flask","scikit-learn","pandas","numpy","joblib"], check=False)
print("✅ Packages ready.")

import os, math, random, warnings, threading, time
import numpy as np, pandas as pd, joblib
from collections import deque
from datetime import datetime
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTDIR = "colab_outputs"
MODELS_LOADED = False
iso_model = rf_model = scaler_if = scaler_rf = scaler_lstm = scaler_lstm_y = None

try:
    iso_model     = joblib.load(f"{OUTDIR}/model_isolation_forest.pkl")
    rf_model      = joblib.load(f"{OUTDIR}/model_random_forest.pkl")
    scaler_if     = joblib.load(f"{OUTDIR}/scaler_if.pkl")
    scaler_rf     = joblib.load(f"{OUTDIR}/scaler_rf.pkl")
    scaler_lstm   = joblib.load(f"{OUTDIR}/scaler_lstm.pkl")
    scaler_lstm_y = joblib.load(f"{OUTDIR}/scaler_lstm_y.pkl")
    MODELS_LOADED = True
    print("✅ ML models loaded.")
except Exception as e:
    print(f"⚠️  Rule-Based mode ({e})")

LSTM_LOADED = False
lstm_model = None
try:
    import tensorflow as tf
    lstm_model  = tf.keras.models.load_model(f"{OUTDIR}/model_lstm.keras")
    LSTM_LOADED = True
    print("✅ LSTM loaded.")
except Exception:
    print("⚠️  LSTM not found — residuals will be zero.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASS_NAMES = ["normal", "leak", "burst", "theft"]

GOVS = {
    "Amman":   {"label":"عمّان",    "zones":["ZN-AM-01","ZN-AM-02","ZN-AM-03","ZN-AM-04","ZN-AM-05"],"nrw":.43,"age":18,"D":.05,"eps":.26,  "fR":(12,75),"pR":(42,85),"env":0,"pop":541},
    "Zarqa":   {"label":"الزرقاء", "zones":["ZN-ZQ-01","ZN-ZQ-02","ZN-ZQ-03"],                        "nrw":.52,"age":35,"D":.05,"eps":.26,  "fR":(8,60), "pR":(32,72),"env":1,"pop":305},
    "Irbid":   {"label":"إربد",    "zones":["ZN-IR-01","ZN-IR-02","ZN-IR-03","ZN-IR-04"],             "nrw":.45,"age":24,"D":.05,"eps":.26,  "fR":(10,65),"pR":(36,78),"env":0,"pop":732},
    "Balqa":   {"label":"البلقاء", "zones":["ZN-BQ-01","ZN-BQ-02"],                                   "nrw":.47,"age":16,"D":.05,"eps":.0015,"fR":(5,45), "pR":(28,68),"env":2,"pop":455},
    "Karak":   {"label":"الكرك",   "zones":["ZN-KR-01","ZN-KR-02"],                                   "nrw":.56,"age":48,"D":.04,"eps":.30,  "fR":(3,28), "pR":(18,55),"env":3,"pop":86},
    "Mafraq":  {"label":"المفرق",  "zones":["ZN-MF-01","ZN-MF-02","ZN-MF-03"],                        "nrw":.50,"age":13,"D":.05,"eps":.007, "fR":(5,35), "pR":(22,62),"env":4,"pop":13},
    "Aqaba":   {"label":"العقبة",  "zones":["ZN-AQ-01","ZN-AQ-02"],                                   "nrw":.41,"age":22,"D":.05,"eps":.26,  "fR":(8,50), "pR":(32,78),"env":5,"pop":31},
    "Madaba":  {"label":"مادبا",   "zones":["ZN-MD-01","ZN-MD-02"],                                   "nrw":.51,"age":22,"D":.05,"eps":.0015,"fR":(4,32), "pR":(22,58),"env":3,"pop":202},
    "Jerash":  {"label":"جرش",     "zones":["ZN-JR-01","ZN-JR-02"],                                   "nrw":.48,"age":19,"D":.04,"eps":.0015,"fR":(3,30), "pR":(24,60),"env":2,"pop":585},
    "Ajloun":  {"label":"عجلون",   "zones":["ZN-AJ-01","ZN-AJ-02"],                                   "nrw":.54,"age":40,"D":.04,"eps":.30,  "fR":(2,25), "pR":(20,55),"env":3,"pop":393},
    "Tafilah": {"label":"الطفيلة", "zones":["ZN-TF-01"],                                              "nrw":.53,"age":25,"D":.04,"eps":.0015,"fR":(2,22), "pR":(18,52),"env":3,"pop":48},
    "Maan":    {"label":"معان",    "zones":["ZN-MN-01","ZN-MN-02"],                                   "nrw":.55,"age":15,"D":.04,"eps":.007, "fR":(2,25), "pR":(18,55),"env":4,"pop":4},
}

IF_FEATURES = [
    "Delta_Flow","Delta_Pressure","Delta_Flow_Pct","Delta_Pressure_Pct",
    "Flow_Ratio","Pressure_Ratio","Loss_Per_100m","DP_Predicted",
    "DP_Deviation","Flow_DP_Ratio","Distance_AB",
]
LSTM_FEATURES = [
    "Flow_A","Pressure_A","Distance_AB","Temp_Water","hour","day_of_week",
    "month","is_ramadan","is_eid","is_summer","is_winter","is_weekend",
    "supply_on","population_density","environment_code","pipe_age_years",
    "hw_coefficient","elevation_m","nrw_rate",
]
RF_FEATURES = IF_FEATURES + [
    "hour","day_of_week","month","is_ramadan","is_eid","is_summer",
    "is_winter","is_weekend","supply_on","population_density",
    "environment_code","pipe_age_years","hw_coefficient","elevation_m",
    "nrw_rate","Flow_A","Pressure_A",
    "IF_Score","IF_Confidence","IF_Flag","LSTM_Res_Flow","LSTM_Res_Pressure",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHYSICS ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def darcy_psi(f, d, D=0.05, e=0.26):
    a  = math.pi * (D / 2) ** 2
    v  = (f / 60000) / (a + 1e-12)
    Re = max(v * D / 1.004e-6, 1.0)
    eD = (e * 1e-3) / D
    A  = (-2.457 * math.log((7 / Re) ** 0.9 + 0.27 * eD)) ** 16
    B  = (37530 / Re) ** 16
    dp = 8 * ((8/Re)**12 + (A+B)**(-1.5))**(1/12) * (d/D) * (1000*v**2/2) * 1.08 * 0.000145038
    return min(max(dp, 0.02), 50.0)


def generate_reading(gov, zone, atype=None):
    g   = GOVS[gov]
    now = datetime.now()
    h, dow, m = now.hour, now.weekday(), now.month

    if atype is None:
        p = g["nrw"] * 0.25
        r = random.random()
        atype = ("leak"   if r < p * 0.45 else
                 "burst"  if r < p * 0.70 else
                 "theft"  if r < p        else "normal")

    dk = (1.38 if 7 <= h <= 9 else
          1.22 if 18 <= h <= 21 else
          0.42 if h < 5 else 1.0)
    if m in [6, 7, 8]:  dk *= 1.35
    if dow in [3, 4]:   dk *= 1.12

    flo, fhi = g["fR"]
    plo, phi = g["pR"]
    dist = round(random.uniform(60, 520), 1)
    fA   = round(max(flo*0.3, min(random.uniform(flo,fhi)*dk + random.gauss(0,.8), fhi*1.5)), 3)
    pA   = round(max(plo*0.65, min(random.uniform(plo,phi) + random.gauss(0,1.0), phi)), 3)
    dp   = darcy_psi(fA, dist, g["D"], g["eps"])
    temp = round(18 + (7 if m in [6,7,8] else -5 if m in [12,1,2] else 0) + random.gauss(0,1.5), 2)
    sev  = 0.0

    if atype == "normal":
        bg = min(0.004*(dist/100)*(1+(g["age"]-20)/80)*(g["nrw"]/0.40), 0.02)
        fB = max(0.01, min(fA*(1-bg) + random.gauss(0,.2), fA))
        pB = max(0.5,  min(pA - dp + random.gauss(0,.35), pA))
    elif atype == "leak":
        sev = random.uniform(0.03, 0.18)
        fB  = max(0.01, min(fA*(1-sev) + random.gauss(0,.3), fA))
        pB  = max(0.5,  min(pA - dp - sev*pA*random.uniform(.35,.75) + random.gauss(0,.45), pA))
    elif atype == "burst":
        sev = random.uniform(0.25, 0.72)
        fB  = max(0.01, min(fA*(1-sev) + random.gauss(0,.9), fA))
        pB  = max(0.5,  min(pA - dp - sev*pA*random.uniform(.70,.92) + random.gauss(0,.8), pA))
    else:  # theft
        sev = random.uniform(0.05, 0.14)
        fB  = max(0.01, min(fA*(1-sev) + random.gauss(0,.2), fA))
        pB  = max(0.5,  min(pA - dp - sev*pA*random.uniform(.04,.16) + random.gauss(0,.3), pA))

    fB  = round(fB, 3);  pB = round(pB, 3)
    dF  = round(fA - fB, 4)
    dP  = round(pA - pB, 4)
    hw  = max(65, (140 if g["eps"] < 0.1 else 120) - 0.5 * g["age"])

    return {
        "Timestamp": now.strftime("%H:%M:%S"), "Governorate": gov, "Zone_ID": zone,
        "Distance_AB": dist, "Flow_A": fA, "Pressure_A": pA, "Flow_B": fB, "Pressure_B": pB,
        "Temp_Water": temp, "hour": h, "day_of_week": dow, "month": m,
        "is_ramadan": int(m == 3), "is_eid": 0,
        "is_summer": int(m in [6,7,8]), "is_winter": int(m in [12,1,2]),
        "is_weekend": int(dow in [3,4]), "supply_on": 1,
        "population_density": g["pop"], "environment_code": g["env"],
        "pipe_age_years": g["age"], "hw_coefficient": round(hw, 1),
        "elevation_m": 750, "nrw_rate": g["nrw"],
        "Delta_Flow": dF, "Delta_Pressure": dP,
        "Delta_Flow_Pct":      round(dF / max(fA, 0.01) * 100, 4),
        "Delta_Pressure_Pct":  round(dP / max(pA, 0.01) * 100, 4),
        "Flow_Ratio":          round(fB / max(fA, 0.01), 5),
        "Pressure_Ratio":      round(pB / max(pA, 0.01), 5),
        "Loss_Per_100m":       round(dF / max(dist / 100, 0.01), 4),
        "DP_Predicted":        round(dp, 4),
        "DP_Deviation":        round(dP - dp, 4),
        "Flow_DP_Ratio":       round(dF / max(abs(dP), 0.10), 4),
        "True_Type": atype, "Severity": round(sev, 4),
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ML INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_inference(row):
    # LSTM residuals
    row["LSTM_Res_Flow"] = 0.0
    row["LSTM_Res_Pressure"] = 0.0
    if LSTM_LOADED and MODELS_LOADED:
        try:
            avail = [c for c in LSTM_FEATURES if c in row]
            xs    = scaler_lstm.transform(pd.DataFrame([row])[avail])
            pred  = scaler_lstm_y.inverse_transform(
                        lstm_model.predict(xs.reshape(1, 1, len(avail)), verbose=0))[0]
            row["LSTM_Res_Flow"]     = round(float(row["Delta_Flow"]     - pred[0]), 4)
            row["LSTM_Res_Pressure"] = round(float(row["Delta_Pressure"] - pred[1]), 4)
        except Exception:
            pass

    # Isolation Forest
    row["IF_Score"] = -0.3;  row["IF_Flag"] = 0;  row["IF_Confidence"] = 0.0
    if MODELS_LOADED:
        try:
            avail = [c for c in IF_FEATURES if c in row]
            x     = scaler_if.transform(pd.DataFrame([row])[avail].fillna(0))
            sc    = float(iso_model.score_samples(x)[0])
            fl    = int(iso_model.predict(x)[0] == -1)
            row["IF_Score"]      = round(sc, 4)
            row["IF_Flag"]       = fl
            row["IF_Confidence"] = round(min(1.0, max(0.0, (-sc - 0.1) / 0.5)), 4)
        except Exception:
            pass

    # Random Forest + Decision Engine
    if MODELS_LOADED:
        try:
            avail = [c for c in RF_FEATURES if c in row]
            x     = scaler_rf.transform(pd.DataFrame([row])[avail].fillna(0))
            lbl   = int(rf_model.predict(x)[0])
            proba = rf_model.predict_proba(x)[0]
            conf  = round(0.30 * row["IF_Confidence"] + 0.70 * (1 - float(proba[0])), 4)
            alert = ((row["IF_Flag"] == 1 or lbl > 0) and conf >= 0.45)
            row["Predicted_Type"]   = CLASS_NAMES[lbl] if alert else "normal"
            row["Final_Confidence"] = conf
        except Exception:
            row["Predicted_Type"]   = "normal"
            row["Final_Confidence"] = 0.5
    else:
        dFP = abs(row["Delta_Flow_Pct"])
        dpD = abs(row["DP_Deviation"])
        if   dFP > 20:              t = "burst"; c = min(0.95, 0.70 + dFP / 100)
        elif dFP > 3 and dpD > 0.5: t = "leak";  c = min(0.90, 0.55 + dFP / 20)
        elif dFP > 5 and dpD <= 0.5:t = "theft"; c = min(0.85, 0.50 + dFP / 25)
        else:                        t = "normal"; c = max(0.05, 1 - dFP / 30)
        row["Predicted_Type"]   = t
        row["Final_Confidence"] = round(c, 4)

    row["Alert"] = int(row["Predicted_Type"] != "normal")
    return row

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIST = 30
S = {
    "gov":  "Amman",
    "fa":   deque([0] * HIST, maxlen=HIST),
    "fb":   deque([0] * HIST, maxlen=HIST),
    "pa":   deque([0] * HIST, maxlen=HIST),
    "pb":   deque([0] * HIST, maxlen=HIST),
    "zones":  [],
    "alerts": [],
    "total":  0,
    "tick":   0,
    "upd":    "",
}


def run_cycle(gov=None, force=None):
    gn = gov or S["gov"]
    S["gov"] = gn
    zones = []
    for i, z in enumerate(GOVS[gn]["zones"]):
        ft  = force if (force and i == 0) else None
        row = generate_reading(gn, z, ft)
        row = run_inference(row)
        zones.append(row)
    S["zones"] = zones
    S["tick"] += 1
    S["upd"]   = datetime.now().strftime("%H:%M:%S")
    r = zones[0]
    S["fa"].append(r["Flow_A"]);    S["fb"].append(r["Flow_B"])
    S["pa"].append(r["Pressure_A"]); S["pb"].append(r["Pressure_B"])
    for row in zones:
        if row.get("Alert"):
            S["total"] += 1
            S["alerts"].insert(0, {
                "time": row["Timestamp"], "gov": gn, "zone": row["Zone_ID"],
                "type": row["Predicted_Type"], "conf": row["Final_Confidence"],
                "dfp":  round(row["Delta_Flow_Pct"], 2),
                "dpd":  round(row["DP_Deviation"], 3),
                "dist": row["Distance_AB"],
            })
    S["alerts"] = S["alerts"][:50]


def background_loop():
    while True:
        try:
            run_cycle()
        except Exception as e:
            print(f"[bg] {e}")
        time.sleep(35)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLASK ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = Flask(__name__)


@app.route("/api/data")
def api_data():
    return jsonify({
        "gov":           S["gov"],
        "gov_label":     GOVS[S["gov"]]["label"],
        "zones":         S["zones"],
        "alerts":        S["alerts"][:15],
        "total_alerts":  S["total"],
        "tick":          S["tick"],
        "last_update":   S["upd"],
        "flow_a_hist":   list(S["fa"]),
        "flow_b_hist":   list(S["fb"]),
        "press_a_hist":  list(S["pa"]),
        "press_b_hist":  list(S["pb"]),
        "govs":          {k: {"label": v["label"], "zones": v["zones"]} for k, v in GOVS.items()},
        "model_mode":    "ML (RF+IF+LSTM)" if MODELS_LOADED else "Rule-Based",
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    d = request.get_json(silent=True) or {}
    run_cycle(d.get("gov"), d.get("force"))
    return jsonify({"ok": True})


@app.route("/")
def index():
    return DASHBOARD_HTML

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DASHBOARD HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AquaGuard AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#080c14;--bg2:#0d1320;--bg3:#111927;--bg4:#162033;--b:rgba(56,189,248,.10);--b2:rgba(56,189,248,.22);--cy:#38bdf8;--gr:#22c55e;--am:#f59e0b;--re:#ef4444;--pu:#a855f7;--tx:#e2e8f0;--mu:#94a3b8;--fa:#475569;--mn:'IBM Plex Mono',monospace;--sn:'Tajawal',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:var(--sn)}
.hdr{display:flex;align-items:center;justify-content:space-between;padding:13px 22px;background:var(--bg2);border-bottom:1px solid var(--b2)}
.logo{font-size:17px;font-weight:700;color:#f8fafc}.logo-s{font-size:10px;color:var(--mu);font-family:var(--mn)}
.hr{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.live{display:flex;align-items:center;gap:6px;padding:4px 11px;border-radius:20px;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);font-size:11px;color:var(--gr);font-family:var(--mn)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--gr);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.tmr{padding:4px 11px;border-radius:20px;background:var(--bg3);border:1px solid var(--b2);font-size:11px;color:var(--cy);font-family:var(--mn)}
select,button{padding:5px 11px;border-radius:8px;background:var(--bg3);border:1px solid var(--b2);color:var(--tx);font-family:var(--sn);font-size:12px;cursor:pointer}
button:hover{background:var(--bg4);border-color:var(--cy)}
.main{padding:16px 18px}
.kpis{display:grid;grid-template-columns:repeat(7,1fr);gap:9px;margin-bottom:14px}
.kpi{background:var(--bg2);border:1px solid var(--b);border-radius:10px;padding:11px}
.kl{font-size:10px;color:var(--mu);font-family:var(--mn);margin-bottom:3px}
.kv{font-size:21px;font-weight:700;font-family:var(--mn)}
.ks{font-size:9px;color:var(--fa);margin-top:2px}
.g2{display:grid;grid-template-columns:1fr 310px;gap:12px;margin-bottom:12px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.pnl{background:var(--bg2);border:1px solid var(--b);border-radius:12px;padding:15px}
.pt{font-size:10px;color:var(--mu);font-family:var(--mn);letter-spacing:.5px;margin-bottom:11px}
.cw{height:190px;position:relative}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:11px}
.sc{background:var(--bg3);border-radius:8px;border:1px solid var(--b);padding:9px;cursor:pointer;transition:.2s}
.sc:hover{border-color:var(--b2)}
.sh{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
.sid{font-size:10px;color:var(--fa);font-family:var(--mn)}
.bdg{font-size:9px;font-weight:600;padding:2px 7px;border-radius:8px;font-family:var(--mn)}
.sv{display:grid;grid-template-columns:1fr 1fr;gap:3px}
.svl{font-size:9px;color:var(--fa)}.svv{font-size:13px;font-weight:600;font-family:var(--mn)}
.bb{height:3px;background:var(--bg);border-radius:2px;margin-top:6px}
.bf{height:3px;border-radius:2px;transition:width .6s}
.abr{display:flex;align-items:center;gap:7px;margin-bottom:7px}
.abn{flex:1;text-align:center;background:var(--bg);border:1px solid var(--b);border-radius:6px;padding:7px}
.abnl{font-size:9px;color:var(--mu);font-family:var(--mn);margin-bottom:2px}
.abnv{font-size:15px;font-weight:700;font-family:var(--mn)}
.abnu{font-size:9px;color:var(--fa);font-family:var(--mn)}
.aba{display:flex;flex-direction:column;align-items:center;gap:3px;min-width:42px}
.abl{width:36px;height:2px;background:var(--b2);position:relative}
.abl::after{content:'';position:absolute;right:-1px;top:-4px;border:5px solid transparent;border-left-color:var(--b2)}
.abls{font-size:9px;color:var(--re);font-family:var(--mn);white-space:nowrap}
.dg{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px;padding-top:7px;border-top:1px solid var(--b)}
.dl{font-size:9px;color:var(--mu);font-family:var(--mn);margin-bottom:2px}
.dv{font-size:12px;font-weight:600;font-family:var(--mn)}
.sb{background:var(--bg3);border-radius:8px;border:1px solid var(--b);border-right:3px solid var(--gr);padding:11px;margin-top:9px}
.st{font-size:13px;font-weight:600;margin-bottom:7px}
.ss{font-size:11px;color:var(--mu);padding:4px 0;border-bottom:1px solid var(--b)}
.ss:last-child{border-bottom:none}
.sbtn{margin-top:7px;width:100%;padding:7px;background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.3);border-radius:7px;color:var(--cy);font-size:11px;font-family:var(--mn);cursor:pointer;text-align:center}
.als{display:flex;flex-direction:column;gap:5px;max-height:250px;overflow-y:auto}
.al{background:var(--bg3);border-radius:7px;padding:8px 10px;border-right:3px solid;animation:si .3s}
@keyframes si{from{transform:translateX(-7px);opacity:0}to{transform:none;opacity:1}}
.at{display:flex;justify-content:space-between;margin-bottom:3px}
.az{font-size:11px;font-weight:500;font-family:var(--mn)}
.atm{font-size:9px;color:var(--fa);font-family:var(--mn)}
.am{font-size:10px;color:var(--mu)}
.ac{font-size:9px;font-family:var(--mn);margin-top:2px}
.stbl{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mn)}
.stbl th{color:var(--fa);font-weight:400;padding:5px 6px;text-align:right;border-bottom:1px solid var(--b);font-size:9px}
.stbl td{padding:6px 6px;border-bottom:1px solid var(--b);color:var(--tx)}
.stbl tr:last-child td{border-bottom:none}
.stbl tr:hover td{background:var(--bg3)}
</style></head><body>
<div class="hdr">
  <div><div class="logo">💧 AquaGuard AI — شبكة المياه الذكية · الأردن</div>
  <div class="logo-s">SPATIAL A-B SENSOR MONITORING · REAL-TIME INFERENCE · 35s CYCLE</div></div>
  <div class="hr">
    <select id="gSel" onchange="swGov()"></select>
    <select id="fSel">
      <option value="">عشوائي</option>
      <option value="leak">leak</option>
      <option value="burst">burst</option>
      <option value="theft">theft</option>
      <option value="normal">normal</option>
    </select>
    <button onclick="manRef()">🔄 تحديث</button>
    <div class="tmr" id="tmr">التحديث في: 35s</div>
    <div class="live"><div class="dot"></div>بث مباشر</div>
  </div>
</div>
<div class="main">
  <div class="kpis">
    <div class="kpi"><div class="kl">FLOW A</div><div class="kv" id="kFA" style="color:var(--cy)">—</div><div class="ks">L/min</div></div>
    <div class="kpi"><div class="kl">FLOW B</div><div class="kv" id="kFB" style="color:var(--gr)">—</div><div class="ks">L/min</div></div>
    <div class="kpi"><div class="kl">ΔFlow%</div><div class="kv" id="kDF" style="color:var(--am)">—</div><div class="ks">فقد التدفق</div></div>
    <div class="kpi"><div class="kl">DP_Deviation</div><div class="kv" id="kDP" style="color:var(--pu)">—</div><div class="ks">PSI</div></div>
    <div class="kpi"><div class="kl">Distance A→B</div><div class="kv" id="kDst" style="color:var(--cy)">—</div><div class="ks">متر</div></div>
    <div class="kpi"><div class="kl">التنبيهات</div><div class="kv" id="kAl" style="color:var(--re)">0</div><div class="ks">الجلسة</div></div>
    <div class="kpi" id="kSt"><div class="kl">PREDICTION</div><div class="kv" id="kPr" style="color:var(--gr)">—</div><div class="ks" id="kCf">—</div></div>
  </div>
  <div class="g2">
    <div style="display:flex;flex-direction:column;gap:11px">
      <div class="pnl">
        <div class="pt">الأقسام النشطة · <span id="gLbl">عمّان</span></div>
        <div class="sg" id="sgGrid"></div>
      </div>
      <div class="pnl">
        <div class="pt">سجل التدفق · Flow A vs B (L/min)</div>
        <div class="cw"><canvas id="fC"></canvas></div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:9px">
      <div class="pnl" style="padding:12px">
        <div class="pt">المقارنة المكانية A ← → B</div>
        <div class="abr">
          <div class="abn"><div class="abnl">حساس A · تدفق</div><div class="abnv" id="aFA" style="color:var(--cy)">—</div><div class="abnu">L/min</div></div>
          <div class="aba"><div class="abl"></div><div class="abls" id="aDFL">ΔQ: —</div></div>
          <div class="abn"><div class="abnl">حساس B · تدفق</div><div class="abnv" id="aFB" style="color:var(--gr)">—</div><div class="abnu">L/min</div></div>
        </div>
        <div class="abr" style="margin-bottom:0">
          <div class="abn"><div class="abnl">ضغط A</div><div class="abnv" id="aPA" style="color:var(--pu);font-size:13px">—</div><div class="abnu">PSI</div></div>
          <div class="aba"><div class="abl"></div><div class="abls" id="aDPL">ΔP: —</div></div>
          <div class="abn"><div class="abnl">ضغط B</div><div class="abnv" id="aPB" style="color:var(--am);font-size:13px">—</div><div class="abnu">PSI</div></div>
        </div>
        <div class="dg">
          <div><div class="dl">ΔFlow%</div><div class="dv" id="dFP" style="color:var(--am)">—</div></div>
          <div><div class="dl">DP_Deviation</div><div class="dv" id="dPD" style="color:var(--pu)">—</div></div>
          <div><div class="dl">Flow_DP_Ratio</div><div class="dv" id="dFDR" style="color:var(--cy)">—</div></div>
          <div><div class="dl">Distance</div><div class="dv" id="dDst" style="color:var(--gr)">—</div></div>
        </div>
        <div class="sb" id="solBox">
          <div class="st" id="solT">✅ الحالة طبيعية</div>
          <div id="solS"></div>
          <div class="sbtn" id="solB" style="display:none">⚡ إنشاء تذكرة</div>
        </div>
      </div>
      <div class="pnl" style="flex:1">
        <div class="pt">التنبيهات · <span id="alCnt">0</span></div>
        <div class="als" id="alList">
          <div style="color:var(--fa);font-size:11px;text-align:center;padding:16px">لا توجد تنبيهات...</div>
        </div>
      </div>
    </div>
  </div>
  <div class="g3">
    <div class="pnl"><div class="pt">الضغط · Pressure A vs B (PSI)</div><div class="cw"><canvas id="pC"></canvas></div></div>
    <div class="pnl"><div class="pt">ΔFlow% لكل قسم</div><div class="cw"><canvas id="bC"></canvas></div></div>
    <div class="pnl">
      <div class="pt">جدول الأقسام</div>
      <table class="stbl" id="sTbl">
        <thead><tr><th>القسم</th><th>ΔFlow%</th><th>DP_Dev</th><th>الحالة</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const C={normal:"#22c55e",leak:"#f59e0b",burst:"#ef4444",theft:"#a855f7"};
const SOLS={
  normal:{t:"✅ الحالة طبيعية",s:["جميع القراءات ضمن النطاق المقبول.","لا إجراء مطلوب."],b:null,c:"#22c55e"},
  leak:  {t:"🟡 تسرب مكتشف — تدخل مطلوب",s:["نشر فريق الكشف الميداني بجهاز صوتي.","تخفيض الضغط مؤقتاً.","رفع تذكرة إصلاح — أولوية متوسطة."],b:"إنشاء تذكرة إصلاح",c:"#f59e0b"},
  burst: {t:"🔴 انفجار حرج — استجابة فورية",s:["عزل القسم فوراً — أغلق الصمامات.","إبلاغ فرق الطوارئ.","إرسال فريق طارئ < 2 ساعة."],b:"إطلاق بروتوكول طوارئ",c:"#ef4444"},
  theft: {t:"🟣 استهلاك غير مرخص",s:["فحص الوصلات الجانبية ميدانياً.","مراجعة سجلات الاستهلاك 72 ساعة.","التنسيق مع قسم الحماية الإيرادية."],b:"إبلاغ قسم الحماية",c:"#a855f7"},
};
const H=30;
Chart.defaults.color="#94a3b8";
Chart.defaults.font.family="IBM Plex Mono,monospace";
Chart.defaults.font.size=10;
const opts={responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{boxWidth:8,padding:9}}},scales:{x:{grid:{color:"rgba(56,189,248,.05)"},ticks:{maxTicksLimit:8}},y:{grid:{color:"rgba(56,189,248,.07)"}}}};
const fCh=new Chart(document.getElementById("fC"),{type:"line",data:{labels:Array(H).fill(""),datasets:[
  {label:"Flow A",data:Array(H).fill(0),borderColor:"#38bdf8",backgroundColor:"rgba(56,189,248,.07)",borderWidth:1.5,pointRadius:2,tension:.35,fill:true},
  {label:"Flow B",data:Array(H).fill(0),borderColor:"#22c55e",backgroundColor:"rgba(34,197,94,.04)",borderWidth:1.5,pointRadius:2,tension:.35,borderDash:[4,3],fill:true}
]},options:opts});
const pCh=new Chart(document.getElementById("pC"),{type:"line",data:{labels:Array(H).fill(""),datasets:[
  {label:"Pressure A",data:Array(H).fill(0),borderColor:"#a855f7",borderWidth:1.5,pointRadius:2,tension:.35},
  {label:"Pressure B",data:Array(H).fill(0),borderColor:"#f59e0b",borderWidth:1.5,pointRadius:2,tension:.35,borderDash:[4,3]}
]},options:opts});
const bCh=new Chart(document.getElementById("bC"),{type:"bar",data:{labels:[],datasets:[{data:[],backgroundColor:[],borderRadius:4}]},options:{...opts,plugins:{legend:{display:false}}}});

let cd=35;

function post(gov,force){
  return fetch("/api/refresh",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({gov:gov,force:force||null})});
}
function swGov(){post(document.getElementById("gSel").value,null).then(()=>fD());}
function manRef(){post(document.getElementById("gSel").value,document.getElementById("fSel").value).then(()=>{fD();cd=35;});}

function fD(){
  fetch("/api/data").then(r=>r.json()).then(d=>{
    const sel=document.getElementById("gSel");
    if(!sel.options.length)
      Object.entries(d.govs).forEach(([k,v])=>{
        const o=document.createElement("option");
        o.value=k; o.textContent=v.label;
        if(k===d.gov) o.selected=true;
        sel.appendChild(o);
      });
    document.getElementById("gLbl").textContent=d.gov_label;
    const r=d.zones[0]||{};
    const typ=r.Predicted_Type||"normal";
    const col=C[typ]||"#22c55e";
    const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
    set("kFA",(r.Flow_A||0).toFixed(1));
    set("kFB",(r.Flow_B||0).toFixed(1));
    set("kDF",(r.Delta_Flow_Pct||0).toFixed(1)+"%");
    set("kDP",(r.DP_Deviation||0).toFixed(2));
    set("kDst",(r.Distance_AB||0).toFixed(0));
    set("kAl",d.total_alerts);
    set("kPr",typ.toUpperCase()); document.getElementById("kPr").style.color=col;
    set("kCf",d.model_mode+" · "+((r.Final_Confidence||0)*100).toFixed(0)+"%");
    set("aFA",(r.Flow_A||0).toFixed(2));   set("aFB",(r.Flow_B||0).toFixed(2));
    set("aPA",(r.Pressure_A||0).toFixed(2)); set("aPB",(r.Pressure_B||0).toFixed(2));
    set("aDFL","ΔQ: "+(r.Delta_Flow||0).toFixed(2)+" L/m");
    set("aDPL","ΔP: "+(r.Delta_Pressure||0).toFixed(2)+" PSI");
    set("dFP",(r.Delta_Flow_Pct||0).toFixed(2)+"%");
    set("dPD",(r.DP_Deviation||0).toFixed(3)+" PSI");
    set("dFDR",(r.Flow_DP_Ratio||0).toFixed(3));
    set("dDst",(r.Distance_AB||0).toFixed(0)+" m");
    const sol=SOLS[typ]||SOLS.normal;
    document.getElementById("solBox").style.borderRightColor=sol.c;
    set("solT",sol.t); document.getElementById("solT").style.color=sol.c;
    document.getElementById("solS").innerHTML=sol.s.map(s=>`<div class="ss">${s}</div>`).join("");
    const sb=document.getElementById("solB");
    if(sol.b){sb.style.display="block"; sb.textContent="⚡ "+sol.b;} else sb.style.display="none";
    document.getElementById("sgGrid").innerHTML=d.zones.map(z=>{
      const c=C[z.Predicted_Type]||"#22c55e";
      const bw=Math.min((z.Delta_Flow_Pct||0)*4,100);
      const bd=z.Alert?`2px solid ${c}`:`1px solid rgba(56,189,248,.1)`;
      return `<div class="sc" style="border:${bd}">
        <div class="sh"><div class="sid">${z.Zone_ID}</div>
        <div class="bdg" style="background:${c}20;color:${c};border:1px solid ${c}50">${(z.Predicted_Type||"normal").toUpperCase()}</div></div>
        <div class="sv">
          <div><div class="svl">Flow A</div><div class="svv" style="color:#38bdf8">${(z.Flow_A||0).toFixed(1)}</div></div>
          <div><div class="svl">Flow B</div><div class="svv" style="color:#22c55e">${(z.Flow_B||0).toFixed(1)}</div></div>
          <div><div class="svl">ΔFlow%</div><div class="svv" style="color:${c}">${(z.Delta_Flow_Pct||0).toFixed(1)}%</div></div>
          <div><div class="svl">Conf.</div><div class="svv" style="color:${c}">${((z.Final_Confidence||0)*100).toFixed(0)}%</div></div>
        </div>
        <div class="bb"><div class="bf" style="width:${bw}%;background:${c}"></div></div>
      </div>`;
    }).join("");
    fCh.data.datasets[0].data=[...d.flow_a_hist];
    fCh.data.datasets[1].data=[...d.flow_b_hist];
    fCh.update("active");
    pCh.data.datasets[0].data=[...d.press_a_hist];
    pCh.data.datasets[1].data=[...d.press_b_hist];
    pCh.update("active");
    bCh.data.labels=d.zones.map(z=>z.Zone_ID.split("-").pop());
    bCh.data.datasets[0].data=d.zones.map(z=>z.Delta_Flow_Pct||0);
    bCh.data.datasets[0].backgroundColor=d.zones.map(z=>C[z.Predicted_Type]||"#38bdf8");
    bCh.update("active");
    set("alCnt",d.total_alerts);
    const al=document.getElementById("alList");
    if(!d.alerts.length){
      al.innerHTML="<div style='color:var(--fa);font-size:11px;text-align:center;padding:16px'>لا توجد تنبيهات...</div>";
    } else {
      al.innerHTML=d.alerts.map(a=>{
        const c=C[a.type]||"#888";
        return `<div class="al" style="border-right-color:${c}">
          <div class="at"><div class="az">${a.zone} · ${a.gov}</div><div class="atm">${a.time}</div></div>
          <div class="am">ΔFlow: ${a.dfp}%  DP_Dev: ${a.dpd} PSI  |  ${a.dist}m</div>
          <div class="ac" style="color:${c}">${a.type.toUpperCase()} · ${(a.conf*100).toFixed(0)}%</div>
        </div>`;
      }).join("");
    }
    const tb=document.querySelector("#sTbl tbody");
    tb.innerHTML=d.zones.map(z=>{
      const c=C[z.Predicted_Type]||"#22c55e";
      return `<tr>
        <td>${z.Zone_ID.split("-").pop()}</td>
        <td style="color:${c}">${(z.Delta_Flow_Pct||0).toFixed(1)}%</td>
        <td style="color:var(--pu)">${(z.DP_Deviation||0).toFixed(2)}</td>
        <td style="color:${c};font-size:9px">${(z.Predicted_Type||"normal").toUpperCase()}</td>
      </tr>`;
    }).join("");
  }).catch(e=>console.error(e));
}

setInterval(()=>{
  cd--;
  document.getElementById("tmr").textContent="التحديث في: "+cd+"s";
  if(cd<=0){
    post(document.getElementById("gSel").value, null).then(()=>{fD(); cd=35;});
  }
},1000);

fD();
</script></body></html>"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAUNCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("✅ All ready. Launching...")
run_cycle()
threading.Thread(target=background_loop, daemon=True).start()
print("✅ Background thread started (35s cycle).")

def run_flask():
    app.run(port=5000, use_reloader=False, debug=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(3)

from google.colab.output import eval_js
url = eval_js("google.colab.kernel.proxyPort(5000)")

print("\n" + "="*55)
print("🚀  AQUAGUARD DASHBOARD IS LIVE")
print("="*55)
print(f"🔗  {url}")
print("="*55)
print(f"    وضع النموذج: {'ML (RF+IF+LSTM)' if MODELS_LOADED else 'Rule-Based'}")
print("    التحديث التلقائي كل 35 ثانية")
print("="*55)