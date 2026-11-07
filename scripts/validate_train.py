# train 사후 검증: 진단(예측 기반)이 실제 90일 매출과 맞는지 채점. 실제 매출은 채점용으로만 사용.
import pandas as pd, numpy as np
from scipy import stats
g = pd.read_csv("outputs/mvp/games_mvp.csv"); t = g[g.split == "train"].copy()
act = pd.read_csv("d-art-b-x-parrot-data-tone/train.csv").set_index("game_id").total_revenue_usd
t["실제매출"] = t.game_id.map(act)
t = t[t.marketing_spend_usd.notna()]
t["실제ROAS"] = t.실제매출 / t.marketing_spend_usd
t["실제매출_셀pct"] = t.groupby("비교셀").실제매출.rank(pct=True) * 100
t["실제ROAS_셀대비"] = t.실제ROAS / t.groupby("비교셀").실제ROAS.transform("median")
print(f"대상: train 게임 {len(t):,}개 (수익 수단 있음 + 마케팅비 기록 있음)\n")

q = t.groupby("사분면").agg(게임수=("game_id", "size"), 실제ROAS_중앙값=("실제ROAS", "median"),
                         셀대비_실제효율=("실제ROAS_셀대비", "median"), 실제매출_중앙값=("실제매출", "median")).reindex(["🟢", "🔵", "🟡", "🔴"])
print("① 사분면별 실제 성과"); print(q.round(2).to_string(), "\n")

rho_q = stats.spearmanr(t["품질_pct"], t["실제매출_셀pct"])[0]
resid = t.groupby("비교셀", group_keys=False).apply(lambda d: pd.Series(
    np.log1p(d.실제매출) - np.polyval(np.polyfit(np.log1p(d.marketing_spend_usd), np.log1p(d.실제매출), 1), np.log1p(d.marketing_spend_usd)), index=d.index))
resid = resid.reindex(t.index); rho_q2 = stats.spearmanr(t["품질_pct"], resid)[0]
qd = pd.qcut(t["품질_pct"], 5, labels=["하위20%", "20-40", "40-60", "60-80", "상위20%"])
print(f"② 품질 백분위 ↔ 실제 매출: 셀 안 순위 상관 ρ={rho_q:.2f} / 마케팅비 영향을 뺀 실제 매출과 ρ={rho_q2:.2f}")
print(t.groupby(qd, observed=True).실제ROAS_셀대비.median().round(2).to_string(), "\n")

md = pd.qcut(t["마케팅_pct"], 5, labels=["하위20%", "20-40", "40-60", "60-80", "상위20%"])
print("③ 마케팅비 백분위별 실제 ROAS 중앙값 (낮을수록 투자 여지가 큰지)")
print(t.groupby(md, observed=True).실제ROAS.median().round(1).to_string(), "\n")

gr = t.groupby("등급").agg(게임수=("game_id", "size"), 셀대비_실제효율=("실제ROAS_셀대비", "median")).reindex(["고", "중", "저"])
print("④ 등급별 (고 = 두 축이 크게 어긋남)"); print(gr.round(2).to_string())
hi = t[t.등급 == "고"]
print("  등급 고 안에서: 🟢", hi[hi.사분면 == "🟢"].실제ROAS_셀대비.median().round(2), "/ 🔴", hi[hi.사분면 == "🔴"].실제ROAS_셀대비.median().round(2), "\n")

# ⑤ 예측 ROAS가 실제 ROAS 순서를 지키나
rho_r = stats.spearmanr(t.roas, t.실제ROAS)[0]
print(f"⑤ 예측 ROAS ↔ 실제 ROAS 순위 상관 ρ={rho_r:.2f}")
q.to_csv("outputs/mvp/검증_사분면별.csv", encoding="utf-8-sig")
