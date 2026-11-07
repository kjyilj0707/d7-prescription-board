# 팀 Advisor 회귀(품질·매출)의 계수 표준오차와 셀 중간값(위시리스트·앱 용량·광고비)을 engine.json에 추가 → 화면의 피처별 영향(포레스트 플롯)용
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "outputs/mvp")
import priority_pipeline as P
D = "d-art-b-x-parrot-data-tone"
adv = P.Advisor(f"{D}/train.csv")
tr = P.prep(pd.read_csv(f"{D}/train.csv")); tr["cell"] = [adv._cell_of(c, s) for c, s in zip(tr.cell, tr.monetization_model)]
trn = tr[~tr.zero].reset_index(drop=True); y = np.log1p(trn.total_revenue_usd.values)
def cov_of(X, b):
    A = np.c_[np.ones(len(X)), X.values.astype(float)]
    r = y - A @ b; rank = np.linalg.matrix_rank(A); s2 = (r @ r) / (len(y) - rank)
    return s2 * np.linalg.pinv(A.T @ A), ["_int"] + list(X.columns), len(y) - rank
Cq, nq, dfq = cov_of(adv._design(trn), adv.bq)
Cf, nf, dff = cov_of(adv._design(trn, True), adv.bf)
iq, jf = {k: i for i, k in enumerate(nq)}, {k: i for i, k in enumerate(nf)}
st = {"q": {}, "f": {}, "df": int(dfq)}
for k in ["lpt", "lsm", "crash", "upd", "lwl", "lapk"]: st["q"][k] = {"var": float(Cq[iq[k], iq[k]])}
for g in P.GENRES:
    k = "pt*" + g; st["q"][k] = {"var": float(Cq[iq[k], iq[k]]), "cov_lpt": float(Cq[iq[k], iq["lpt"]])}
st["f"]["lmk"] = {"var": float(Cf[jf["lmk"], jf["lmk"]])}
for p in ["Mid-size", "AAA", "Unknown"]:
    k = "mk*" + p; st["f"][k] = {"var": float(Cf[jf[k], jf[k]]), "cov_lmk": float(Cf[jf[k], jf["lmk"]])}
med = trn.groupby("cell").agg(wish=("wishlist_prelaunch", "median"), apk=("apk_size_mb", "median"), mkt=("marketing_spend_usd", "median"))
for f in ["outputs/mvp/engine.json", "/Users/yeobby/DartB_8th/d7-prescription-board/data/engine.json"]:
    e = json.load(open(f))
    e["coef_stats"] = st
    for c in e["cells"]:
        if c in med.index: e["cells"][c].update({k: float(med.loc[c, k]) for k in ["wish", "apk", "mkt"]})
    json.dump(e, open(f, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
se = lambda v: v ** .5
print("품질 계수 ± SE:", {k: f'{adv.coefq[k]:.4f} ± {se(st["q"][k]["var"]):.4f}' for k in ["lpt", "lsm", "crash", "upd", "lwl", "lapk"]})
print("광고비 계수 ± SE:", f'{dict(zip(["_int"] + adv.cols_f, adv.bf))["lmk"]:.4f} ± {se(st["f"]["lmk"]["var"]):.4f}')
