# 팀원 파이프라인(priority_pipeline.py, Advisor)을 그대로 사용해 MVP 데이터·브라우저 엔진·train 검증을 만든다
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "outputs/mvp")
import priority_pipeline as P
from sklearn.model_selection import KFold
from scipy import stats
D = "d-art-b-x-parrot-data-tone"
adv = P.Advisor(f"{D}/train.csv")

# ---- test 게임 진단 (예측 매출 = 팀 공식 0.6741) ----
te = pd.read_csv(f"{D}/test.csv")
sub = pd.read_csv(f"{D}/submission_통합모델_cv0.6741.csv").set_index("game_id").total_revenue_usd
te["pred_log"] = np.log1p(te.game_id.map(sub))
res = adv.diagnose(te.to_dict("records"))
R = {g["game_id"]: g for g in res}
EMO = {v: k for k, v in P.QT.items()}
for g in res: g["사분면_코드"] = EMO.get(g["사분면"], g["사분면"])
GRADE = lambda q, m: "판단 보류" if m is None else ("고" if abs(q - m) >= 55 else "중" if abs(q - m) >= 30 else "저")

# 세부 진단용 셀 중앙값(세션 길이·하루 세션 수)은 팀 코드에 없어 같은 기준(train POS, 같은 셀)으로 추가 계산
tr = P.prep(pd.read_csv(f"{D}/train.csv")); tr["cell"] = [adv._cell_of(c, s) for c, s in zip(tr.cell, tr.monetization_model)]
trn = tr[~tr.zero]
med_sess = trn.groupby("cell").avg_session_minutes.median(); med_daily = trn.groupby("cell").avg_daily_sessions.median()

rows = []
for _, r in P.prep(te).iterrows():
    g = R[r.game_id]
    if g["사분면_코드"] == "⛔": continue
    m, q, rx = g["마케팅"], g["품질"], g["처방"]
    sol = rx["솔루션"]; qv = rx.get("품질개선변수")
    gains = {"충돌률": q["충돌률"]["중앙값까지개선시_추가매출_usd"], "플레이타임": q["하루플레이타임_분"]["중앙값까지개선시_추가매출_usd"], "업데이트": q["업데이트간격_일"]["중앙값까지개선시_추가매출_usd"]}
    best_q = max(gains, key=lambda k: gains[k] * P.FEAS[k]) if max(gains.values()) > 0 else "없음"
    E = m["예측매출_usd"]
    rows.append({"game_id": r.game_id, "split": "test", "genre": r.genre, "monetization_model": r.monetization_model, "채널": r.struct,
        "셀": r.genre + " × " + r.monetization_model, "비교셀": g["셀"], "비교셀_n": int(adv.med.loc[g["셀"], "n"]), "publisher_tier": r.publisher_tier,
        "pred_usd": E, "marketing_spend_usd": m["마케팅비_usd"], "roas": m["예측ROAS"], "한계roas": m["한계ROAS"], "추가매출_예산": m["마케팅비50%증액시_추가매출_usd"],
        "품질_pct": q["품질_셀백분위"], "마케팅_pct": m["마케팅비_셀백분위"],
        "불균형": None if m["마케팅비_셀백분위"] is None else round(q["품질_셀백분위"] - m["마케팅비_셀백분위"], 1),
        "사분면": g["사분면_코드"], "사분면_이름": g["사분면"], "등급": GRADE(q["품질_셀백분위"], m["마케팅비_셀백분위"]), "우선순위점수": g["우선순위점수"],
        "점검": rx.get("점검"), "축소_usd": (rx.get("집행축소") or {}).get("감액_usd"), "축소_매출감소_usd": (rx.get("집행축소") or {}).get("예상매출감소_usd"),
        "playtime": q["하루플레이타임_분"]["현재"], "avg_session_minutes": r.avg_session_minutes, "avg_daily_sessions": r.avg_daily_sessions,
        "crash_rate_pct": q["충돌률"]["현재"], "update_frequency_days": r.update_frequency_days, "wishlist_prelaunch": r.wishlist_prelaunch, "apk_size_mb": r.apk_size_mb,
        "med_playtime": q["하루플레이타임_분"]["셀중앙값"], "med_crash_rate_pct": q["충돌률"]["셀중앙값"], "med_update_frequency_days": q["업데이트간격_일"]["셀중앙값"],
        "med_avg_session_minutes": float(med_sess[g["셀"]]), "med_avg_daily_sessions": float(med_daily[g["셀"]]),
        "추가매출_플레이타임": gains["플레이타임"], "추가매출_충돌률": gains["충돌률"], "추가매출_업데이트": gains["업데이트"],
        "품질솔루션": best_q, "품질솔루션_배수": round(1 + gains[best_q] / E, 3) if best_q != "없음" and E > 0 else 1.0,
        "처방축": rx["축"], "솔루션": sol if sol in ("충돌률", "플레이타임", "업데이트", "예산") else "유지",
        "예상추가매출": m["마케팅비50%증액시_추가매출_usd"] if sol == "예산" else (qv["예상추가매출_usd"] if qv else 0.0)})
out = pd.DataFrame(rows)
out.to_csv("outputs/mvp/games_mvp.csv", index=False, encoding="utf-8-sig")
zero_test = sum(1 for g in res if g["사분면_코드"] == "⛔")
print(f"test 진단 {len(out)}개 + 수익 수단 없음 {zero_test}개")
print("사분면", out.사분면.value_counts().to_dict(), "| 등급", out.등급.value_counts().to_dict(), "| 솔루션", out.솔루션.value_counts().to_dict())
json.dump({"설명": P.NOTE, "games": res}, open("outputs/mvp/팀파이프라인_test결과.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- 웹 보드용 games_test.json ----
cells = out.groupby("비교셀").agg(n=("비교셀_n", "first"), roas=("roas", "median"), mroas=("한계roas", "median"), pred=("pred_usd", "median"), mkt=("marketing_spend_usd", "median")).reset_index()
cells["quality"] = None
payload = {"quality_coef": {k: round(float(adv.coefq[k]), 4) for k in ["lpt", "lsm", "crash", "upd", "lwl", "lapk"]},
           "cols": list(out.columns), "rows": out.replace({np.nan: None}).values.tolist(), "cells": cells.replace({np.nan: None}).to_dict(orient="records"), "zero_test": zero_test}
json.dump(payload, open("outputs/mvp/games_test.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# ---- 브라우저 진단 엔진 (팀 Advisor 계수·분포 그대로) ----
r6 = lambda v: round(float(v), 6)
eng = {"coefq": {k: r6(v) for k, v in zip(["_int"] + adv.cols_q, adv.bq)}, "coeff": {k: r6(v) for k, v in zip(["_int"] + adv.cols_f, adv.bf)},
       "med_mk": adv.med_mk, "small": sorted(adv.small), "elasticity": P.ELASTICITY, "feas": P.FEAS, "grade": [55, 30], "cut": P.CUT, "q_cut": P.Q_CUT, "m_cut": P.M_CUT, "cell_basis": "monetization_model",
       "cells": {c: {"n": int(adv.med.loc[c, "n"]), "q": np.round(adv.ref_q[c], 4).tolist(), "m": np.round(adv.ref_mk[c], 2).tolist(),
                     "crash": float(adv.med.loc[c, "crash"]), "upd": float(adv.med.loc[c, "upd"]), "play": float(adv.med.loc[c, "play"]),
                     "sess": float(med_sess[c]), "daily": float(med_daily[c]), "roas": None} for c in adv.ref_q},
       "levels": {k: sorted(te[k].dropna().unique().tolist()) for k in ["genre", "publisher_tier", "region_focus", "multiplayer_support"]}}
for c in eng["cells"]:
    s = out[out.비교셀 == c]; eng["cells"][c]["roas"] = float(s.roas.median()) if len(s) else None; eng["cells"][c]["mroas"] = float(s.한계roas.median()) if len(s) else None
excols = ["game_id", "genre", "monetization_model", "contains_ads", "has_in_app_purchases", "publisher_tier", "region_focus", "multiplayer_support",
          "marketing_spend_usd", "avg_session_minutes", "avg_daily_sessions", "crash_rate_pct", "update_frequency_days", "wishlist_prelaunch", "apk_size_mb", "pred_log"]
eng["example_cols"] = excols
eng["examples"] = json.loads(te[excols].round(4).to_json(orient="values", force_ascii=False))
eng["demo"] = []
json.dump(eng, open("outputs/mvp/engine.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("engine.json 저장")

# ---- train 사후 검증 (팀 품질 모델의 교차적합 값, 실제 매출은 채점용) ----
y = np.log1p(trn.total_revenue_usd.values); trn = trn.reset_index(drop=True); Xq = adv._design(trn); Q = np.zeros(len(trn))
for a, b in KFold(5, shuffle=True, random_state=1).split(trn):
    Q[b] = P.predict(Xq.iloc[b], P.ols(Xq.iloc[a], y[a]))
trn["Q"] = Q
trn["qp"] = [adv._pct(adv.ref_q[c], v) for c, v in zip(trn.cell, trn.Q)]
trn["mp"] = [adv._pct(adv.ref_mk[c], v) if v == v else None for c, v in zip(trn.cell, trn.marketing_spend_usd)]
v = trn[trn.mp.notna()].copy()
v["quad"] = np.where(v.qp >= 50, np.where(v.mp < 50, "🟢", "🔵"), np.where(v.mp < 50, "🟡", "🔴"))
v["roas"] = v.total_revenue_usd / v.marketing_spend_usd
v["rel"] = v.roas / v.groupby("cell").roas.transform("median")
resid = pd.Series(index=v.index, dtype=float)
for c, d in v.groupby("cell"):
    b = np.polyfit(np.log(d.marketing_spend_usd), np.log1p(d.total_revenue_usd), 1)
    resid[d.index] = np.log1p(d.total_revenue_usd) - np.polyval(b, np.log(d.marketing_spend_usd))
print(f"\n[train 검증] {len(v):,}개")
print(v.groupby("quad").agg(게임수=("roas", "size"), 실제ROAS=("roas", "median"), 셀대비=("rel", "median")).reindex(["🟢", "🔵", "🟡", "🔴"]).round(2).to_string())
print("품질 백분위 ↔ 마케팅비 영향 뺀 실제 매출 ρ = %.2f" % stats.spearmanr(v.qp, resid)[0])
print("우선순위 점수 ↔ 셀 대비 실제 효율 ρ = %.2f" % stats.spearmanr(v.qp * (100 - v.mp), v.rel)[0])
