import numpy as np, pandas as pd, random, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED); random.seed(SEED)

RHO=1000.0; NU=1.004e-6; PA_TO_PSI=0.000145038; EPSILON_PSI=0.10

ROUGHNESS_MM={"ductile_iron":0.26,"cast_iron":0.60,"pvc":0.0015,"hdpe":0.007,"asbestos_cement":0.12,"steel":0.046}
ENV_CODE={"urban":0,"urban_industrial":1,"mixed":2,"rural":3,"desert":4,"coastal_desert":5}
MAT_CODE={"ductile_iron":0,"cast_iron":1,"pvc":2,"hdpe":3,"asbestos_cement":4,"steel":5}
LABEL_MAP={"normal":0,"leak":1,"burst":2,"theft":3}

GOVERNORATES={
  "Amman":{"population":4100000,"area_km2":7579,"environment":"urban","elevation_m":790,"pipe_material":"ductile_iron","pipe_age_years":18,"pipe_diameter_m":0.100,"supply_hours":12,"nrw_rate":0.43,"flow_range":(15.0,110.0),"pressure_range":(45.0,82.0),"summer_mult":1.40,"winter_mult":0.80,"ramadan_day":0.58,"ramadan_eve":1.55,"eid_mult":1.45,"zones":["AM-Z01","AM-Z02","AM-Z03","AM-Z04","AM-Z05","AM-Z06"]},
  "Zarqa":{"population":1400000,"area_km2":4761,"environment":"urban_industrial","elevation_m":620,"pipe_material":"cast_iron","pipe_age_years":36,"pipe_diameter_m":0.080,"supply_hours":8,"nrw_rate":0.52,"flow_range":(8.0,75.0),"pressure_range":(32.0,68.0),"summer_mult":1.32,"winter_mult":0.83,"ramadan_day":0.55,"ramadan_eve":1.50,"eid_mult":1.38,"zones":["ZQ-Z01","ZQ-Z02","ZQ-Z03","ZQ-Z04"]},
  "Irbid":{"population":1100000,"area_km2":1572,"environment":"urban","elevation_m":620,"pipe_material":"ductile_iron","pipe_age_years":22,"pipe_diameter_m":0.090,"supply_hours":10,"nrw_rate":0.44,"flow_range":(8.0,72.0),"pressure_range":(38.0,75.0),"summer_mult":1.28,"winter_mult":0.86,"ramadan_day":0.58,"ramadan_eve":1.48,"eid_mult":1.36,"zones":["IR-Z01","IR-Z02","IR-Z03","IR-Z04","IR-Z05"]},
  "Balqa":{"population":480000,"area_km2":1120,"environment":"mixed","elevation_m":950,"pipe_material":"pvc","pipe_age_years":16,"pipe_diameter_m":0.075,"supply_hours":9,"nrw_rate":0.47,"flow_range":(4.0,45.0),"pressure_range":(30.0,65.0),"summer_mult":1.22,"winter_mult":0.88,"ramadan_day":0.62,"ramadan_eve":1.42,"eid_mult":1.33,"zones":["BQ-Z01","BQ-Z02","BQ-Z03"]},
  "Karak":{"population":260000,"area_km2":3217,"environment":"rural","elevation_m":1100,"pipe_material":"asbestos_cement","pipe_age_years":48,"pipe_diameter_m":0.075,"supply_hours":6,"nrw_rate":0.56,"flow_range":(2.0,30.0),"pressure_range":(20.0,52.0),"summer_mult":1.18,"winter_mult":0.90,"ramadan_day":0.65,"ramadan_eve":1.38,"eid_mult":1.28,"zones":["KR-Z01","KR-Z02","KR-Z03"]},
  "Mafraq":{"population":310000,"area_km2":26550,"environment":"desert","elevation_m":700,"pipe_material":"hdpe","pipe_age_years":12,"pipe_diameter_m":0.075,"supply_hours":7,"nrw_rate":0.49,"flow_range":(2.0,28.0),"pressure_range":(22.0,58.0),"summer_mult":1.48,"winter_mult":0.78,"ramadan_day":0.50,"ramadan_eve":1.60,"eid_mult":1.52,"zones":["MF-Z01","MF-Z02","MF-Z03"]},
  "Aqaba":{"population":210000,"area_km2":6905,"environment":"coastal_desert","elevation_m":5,"pipe_material":"ductile_iron","pipe_age_years":20,"pipe_diameter_m":0.090,"supply_hours":14,"nrw_rate":0.40,"flow_range":(3.0,42.0),"pressure_range":(35.0,74.0),"summer_mult":1.52,"winter_mult":0.84,"ramadan_day":0.60,"ramadan_eve":1.46,"eid_mult":1.55,"zones":["AQ-Z01","AQ-Z02","AQ-Z03"]},
  "Madaba":{"population":180000,"area_km2":940,"environment":"rural","elevation_m":800,"pipe_material":"pvc","pipe_age_years":20,"pipe_diameter_m":0.075,"supply_hours":8,"nrw_rate":0.50,"flow_range":(2.0,28.0),"pressure_range":(24.0,56.0),"summer_mult":1.20,"winter_mult":0.88,"ramadan_day":0.64,"ramadan_eve":1.40,"eid_mult":1.30,"zones":["MD-Z01","MD-Z02"]},
  "Jerash":{"population":205000,"area_km2":410,"environment":"mixed","elevation_m":580,"pipe_material":"pvc","pipe_age_years":18,"pipe_diameter_m":0.075,"supply_hours":9,"nrw_rate":0.46,"flow_range":(2.0,26.0),"pressure_range":(26.0,60.0),"summer_mult":1.24,"winter_mult":0.86,"ramadan_day":0.62,"ramadan_eve":1.42,"eid_mult":1.32,"zones":["JR-Z01","JR-Z02"]},
  "Ajloun":{"population":152000,"area_km2":420,"environment":"rural","elevation_m":1200,"pipe_material":"steel","pipe_age_years":28,"pipe_diameter_m":0.075,"supply_hours":8,"nrw_rate":0.48,"flow_range":(1.5,22.0),"pressure_range":(22.0,55.0),"summer_mult":1.20,"winter_mult":0.88,"ramadan_day":0.63,"ramadan_eve":1.40,"eid_mult":1.29,"zones":["AJ-Z01","AJ-Z02"]},
  "Tafilah":{"population":97000,"area_km2":2210,"environment":"rural","elevation_m":1100,"pipe_material":"asbestos_cement","pipe_age_years":42,"pipe_diameter_m":0.075,"supply_hours":6,"nrw_rate":0.54,"flow_range":(1.5,18.0),"pressure_range":(18.0,48.0),"summer_mult":1.16,"winter_mult":0.90,"ramadan_day":0.66,"ramadan_eve":1.37,"eid_mult":1.26,"zones":["TF-Z01","TF-Z02"]},
  "Maan":{"population":130000,"area_km2":32832,"environment":"desert","elevation_m":1070,"pipe_material":"hdpe","pipe_age_years":14,"pipe_diameter_m":0.075,"supply_hours":7,"nrw_rate":0.51,"flow_range":(1.5,20.0),"pressure_range":(20.0,52.0),"summer_mult":1.42,"winter_mult":0.80,"ramadan_day":0.52,"ramadan_eve":1.58,"eid_mult":1.48,"zones":["MN-Z01","MN-Z02"]},
}

RAMADAN_WINDOWS=[(datetime(2020,4,24),datetime(2020,5,23)),(datetime(2021,4,13),datetime(2021,5,12)),(datetime(2022,4,2),datetime(2022,5,1)),(datetime(2023,3,23),datetime(2023,4,20)),(datetime(2024,3,11),datetime(2024,4,9))]
EID_DAYS=[datetime(2020,5,24),datetime(2021,5,13),datetime(2022,5,2),datetime(2023,4,21),datetime(2024,4,10),datetime(2020,7,31),datetime(2021,7,20),datetime(2022,7,9),datetime(2023,6,28),datetime(2024,6,16)]

def _is_ramadan(ts):
    for s,e in RAMADAN_WINDOWS:
        if s<=ts<=e: return True
    return False

def _is_eid(ts):
    d=ts.date()
    return any(abs((d-e.date()).days)<=2 for e in EID_DAYS)

def hw_coeff(mat,age):
    base={"ductile_iron":140,"cast_iron":130,"pvc":150,"hdpe":150,"asbestos_cement":120,"steel":135}[mat]
    decay={"ductile_iron":0.4,"cast_iron":0.6,"pvc":0.15,"hdpe":0.15,"asbestos_cement":0.5,"steel":0.4}[mat]
    return max(65, base - decay*age)

def darcy_dp_psi(flow_lpm, dist_m, D, rough_mm):
    area=np.pi*(D/2)**2
    v=(flow_lpm/1000/60)/max(area,1e-9)
    if v<1e-4: return 0.01
    Re=v*D/NU; eps_D=(rough_mm*1e-3)/D
    A=(-2.457*np.log((7/Re)**0.9+0.27*eps_D))**16
    B=(37530/Re)**16
    f=8*((8/Re)**12+(A+B)**(-1.5))**(1/12)
    f=np.clip(f,0.008,0.09)
    dp_pa=f*(dist_m/D)*(RHO*v**2/2)
    return float(np.clip(dp_pa*PA_TO_PSI*1.08, 0.01, 40.0))

def demand_mult(ts, cfg):
    h,m,dow=ts.hour,ts.month,ts.weekday()
    ram=_is_ramadan(ts); eid=_is_eid(ts)
    season=cfg["summer_mult"] if m in [6,7,8] else cfg["winter_mult"] if m in [12,1,2] else 1.00 if m in [3,4,5] else 0.95
    if eid: return float(np.clip(season*cfg["eid_mult"],0.3,3.0))
    if ram:
        if 5<=h<=15: return float(np.clip(season*cfg["ramadan_day"],0.3,3.0))
        if h>=18 or h<=3: return float(np.clip(season*cfg["ramadan_eve"],0.3,3.0))
        return float(np.clip(season*0.85,0.3,3.0))
    diurnal=1.40 if 7<=h<=9 else 1.25 if 18<=h<=21 else 0.40 if 0<=h<=4 else 0.82 if 12<=h<=14 else 1.00
    wend=1.12 if dow in [3,4] else 1.00
    return float(np.clip(season*diurnal*wend,0.25,2.80))

def inject_normal(fa,pa,dp,cfg,d):
    af=1.0+(cfg["pipe_age_years"]-15)*0.012
    bg=float(np.clip(0.004*(d/100)*af*(cfg["nrw_rate"]/0.40),0.001,0.020))
    fb=float(np.clip(fa*(1-bg)+np.random.normal(0,0.25), fa*0.97, fa*0.999))
    pb=float(np.clip(pa-dp+np.random.normal(0,0.35), 1.0, pa))
    return fb,pb,0.0

def inject_leak(fa,pa,dp,cfg,d):
    s=float(np.random.uniform(0.03,0.18))
    fb=float(np.clip(fa*(1-s)+np.random.normal(0,0.30), fa*0.75, fa*0.97))
    edp=s**0.5*pa*np.random.uniform(0.18,0.45)
    pb=float(np.clip(pa-dp-edp+np.random.normal(0,0.40), 1.0, pa-0.5))
    return fb,pb,s

def inject_burst(fa,pa,dp,cfg,d):
    s=float(np.random.uniform(0.22,0.72))
    fb=float(np.clip(fa*(1-s)+np.random.normal(0,0.80), fa*0.20, fa*0.78))
    edp=s*pa*np.random.uniform(0.55,0.92)
    pb=float(np.clip(pa-dp-edp+np.random.normal(0,0.80), 0.5, pa-1.0))
    return fb,pb,s

def inject_theft(fa,pa,dp,cfg,d):
    s=float(np.random.uniform(0.04,0.13))
    fb=float(np.clip(fa*(1-s)+np.random.normal(0,0.20), fa*0.84, fa*0.97))
    edp=s*pa*np.random.uniform(0.02,0.12)
    pb=float(np.clip(pa-dp-edp+np.random.normal(0,0.30), 1.0, pa-0.1))
    return fb,pb,s

INJECTORS={"normal":inject_normal,"leak":inject_leak,"burst":inject_burst,"theft":inject_theft}

def make_sample(ts, gname, zone, atype, dist_m):
    cfg=GOVERNORATES[gname]; mat=cfg["pipe_material"]; D=cfg["pipe_diameter_m"]
    rough=ROUGHNESS_MM[mat]
    h,dow,m=ts.hour,ts.weekday(),ts.month
    ram=_is_ramadan(ts); eid=_is_eid(ts)
    is_s=int(m in [6,7,8]); is_w=int(m in [12,1,2]); is_wk=int(dow in [3,4])
    supply_on=int(h<cfg["supply_hours"] or h>=22)
    mult=demand_mult(ts,cfg)
    fl,fh=cfg["flow_range"]
    fa=float(np.clip(np.random.uniform(fl,fh)*mult+np.random.normal(0,1.2), fl*0.25, fh*1.5))
    pl,ph=cfg["pressure_range"]
    pa=float(np.clip(np.random.uniform(pl,ph)+np.random.normal(0,1.0), pl*0.7, ph))
    dp_pred=darcy_dp_psi(fa,dist_m,D,rough)
    fb,pb,sev=INJECTORS[atype](fa,pa,dp_pred,cfg,dist_m)
    fb=float(np.clip(fb,0.10,fa)); pb=float(np.clip(pb,0.50,pa))
    # Recompute all derived values from raw A/B
    df_=fa-fb; dp_=pa-pb
    dfp=(df_/fa*100) if fa>0 else 0.0
    dpp=(dp_/pa*100) if pa>0 else 0.0
    fr=fb/fa if fa>0 else 0.0
    pr=pb/pa if pa>0 else 0.0
    l100=df_/(dist_m/100) if dist_m>0 else 0.0
    dpdev=dp_-dp_pred
    fdpr=float(np.clip(df_/max(abs(dp_),EPSILON_PSI),-50,50))
    temp_w=float(np.clip((35+np.random.normal(0,2) if is_s else 8+np.random.normal(0,2))*0.6+9,8,32))
    hwc=hw_coeff(mat,cfg["pipe_age_years"])
    return {
        "timestamp":ts.isoformat(),"Governorate":gname,"Zone_ID":zone,
        "Segment_ID":f"{gname[:3].upper()}-{zone[-3:]}-{int(dist_m)}m",
        "hour":h,"day_of_week":dow,"month":m,
        "is_ramadan":int(ram),"is_eid":int(eid),
        "is_summer":is_s,"is_winter":is_w,"is_weekend":is_wk,"supply_on":supply_on,
        "population_density":round(cfg["population"]/cfg["area_km2"],2),
        "environment_code":ENV_CODE[cfg["environment"]],
        "pipe_material_code":MAT_CODE[mat],
        "pipe_age_years":cfg["pipe_age_years"],
        "pipe_diameter_mm":round(D*1000,0),
        "hw_coefficient":round(hwc,1),
        "elevation_m":cfg["elevation_m"],"nrw_rate":cfg["nrw_rate"],
        "Flow_A":round(fa,3),"Pressure_A":round(pa,3),
        "Flow_B":round(fb,3),"Pressure_B":round(pb,3),
        "Distance_AB":round(dist_m,1),"Temp_Water":round(temp_w,2),
        "Delta_Flow":round(df_,4),"Delta_Pressure":round(dp_,4),
        "Delta_Flow_Pct":round(dfp,4),"Delta_Pressure_Pct":round(dpp,4),
        "Flow_Ratio":round(fr,5),"Pressure_Ratio":round(pr,5),
        "Loss_Per_100m":round(l100,4),
        "DP_Predicted":round(dp_pred,4),"DP_Deviation":round(dpdev,4),
        "Flow_DP_Ratio":round(fdpr,4),
        # TARGET COLUMNS ONLY - never use as model input features
        "Anomaly_Type":atype,"Anomaly_Label":LABEL_MAP[atype],
        "Anomaly_Binary":0 if atype=="normal" else 1,"Severity":round(sev,4),
    }

def _build_pool():
    pool=[]
    for gn,cfg in GOVERNORATES.items():
        for zone in cfg["zones"]:
            for _ in range(3):
                dist=float(np.random.uniform(60,580))
                pool.append((gn,zone,dist))
    return pool

def generate(n, anom_ratio, start_date, tag):
    pool=_build_pool(); ts0=datetime.strptime(start_date,"%Y-%m-%d"); interval=timedelta(minutes=15)
    n_norm=int(n*(1-anom_ratio)); n_anom=n-n_norm
    n_leak=int(n_anom*0.44); n_burst=int(n_anom*0.24); n_theft=n_anom-n_leak-n_burst
    types=(["normal"]*n_norm+["leak"]*n_leak+["burst"]*n_burst+["theft"]*n_theft)
    random.shuffle(types)
    print(f"\n[{tag}] {n:,} rows  |  normal={n_norm:,} leak={n_leak:,} burst={n_burst:,} theft={n_theft:,}")
    records=[make_sample(ts0+interval*i, *random.choice(pool)[:2], types[i], random.choice(pool)[2]) for i in range(n)]
    df=pd.DataFrame(records)
    assert (df.Flow_B<=df.Flow_A+0.01).all(),"Flow_B>Flow_A"
    assert (df.Pressure_B<=df.Pressure_A+0.01).all(),"Pres_B>Pres_A"
    assert df.DP_Predicted.between(0.005,41).all(),"DP_Pred range"
    assert df.Flow_DP_Ratio.between(-50.1,50.1).all(),"FlowDPRatio overflow"
    print(f"  Shape:{df.shape}  Anom:{df.Anomaly_Binary.mean():.1%}  Govs:{df.Governorate.nunique()}")
    print(f"  DP_Pred: {df.DP_Predicted.min():.3f}–{df.DP_Predicted.max():.2f} PSI ✓")
    print(f"  Flow_DP_Ratio: {df.Flow_DP_Ratio.min():.2f}–{df.Flow_DP_Ratio.max():.2f} ✓")
    return df

if __name__=="__main__":
    os.makedirs("/mnt/user-data/outputs",exist_ok=True)
    print("="*55+"\nJORDAN WATER NETWORK DATASET v2.0 — FULL SCALE\n"+"="*55)
    df_tr=generate(80000,0.20,"2020-01-01","TRAIN 80k")
    df_tr.to_csv("/mnt/user-data/outputs/jordan_train_v2.csv",index=False)
    df_te=generate(20000,0.05,"2023-07-01","TEST 20k")
    df_te.to_csv("/mnt/user-data/outputs/jordan_test_v2.csv",index=False)
    print(f"\nTOTAL: {len(df_tr)+len(df_te):,} | COLS: {df_tr.shape[1]}")
    print("TRAIN distribution:"); print(df_tr.Anomaly_Type.value_counts().to_string())
    print("TEST distribution:"); print(df_te.Anomaly_Type.value_counts().to_string())
    for f in ["jordan_train_v2.csv","jordan_test_v2.csv"]:
        sz=os.path.getsize(f"/mnt/user-data/outputs/{f}")/1e6
        print(f"  {f}: {sz:.1f} MB")
