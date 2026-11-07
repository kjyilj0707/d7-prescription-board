"""
D7 게임 진단·솔루션 추천기  (입력 → 팀 결과 JSON과 같은 형식으로 출력)

입력한 게임에 대해서만 결과를 낸다. 게임 1개든 여러 개든 비교군 기준은 train.csv에서 한 번 만든다.
여러 개를 넣고 --budget을 주면 그 게임들 안에서만 마케팅 증액 예산을 나눈다.

흐름
 1) 클래스 분류 : 장르 × 수익화 구조(광고+IAP / 광고만 / IAP만 / 없음) 셀, 50개 미만 셀은 같은 구조 전체로 대체
 2) 우선순위   : 셀 안 품질 백분위(마케팅비 제외 D7 지표 회귀) + 마케팅비 백분위 → 사분면·등급
                 🟢 품질↑ 마케팅↓ / 🔵 둘 다↑ / 🟡 둘 다↓ / 🔴 품질↓ 마케팅↑ / ⚪ 마케팅비 없음 / ⛔ 수익 수단 없음
                 등급 = |품질 백분위 − 마케팅비 백분위| 55 이상 고, 30 이상 중, 그 외 저
 3) 솔루션     : 🟢 마케팅 증액(예산) / 🔵 유지 / 그 외 품질 개선
                 품질 개선 = 충돌률·플레이타임·업데이트 중 셀 중앙값까지 개선했을 때
                             (예상 추가 매출 × 실행 가능성)이 가장 큰 변수

사용
  python priority_pipeline.py --train train.csv --game example_game.json
  python priority_pipeline.py --train train.csv --games games.csv --budget 5000 --out result.json
입력 열(test.csv와 동일): genre, monetization_model, contains_ads(Yes/No), has_in_app_purchases(Yes/No),
  publisher_tier, region_focus, multiplayer_support, avg_session_minutes, avg_daily_sessions,
  crash_rate_pct, update_frequency_days, wishlist_prelaunch, apk_size_mb, marketing_spend_usd
  (선택) game_id, pred_log (팀 앙상블 예측의 log1p 값. 없으면 내부 회귀 모델로 대체)
주의: 모든 예상치는 상관 기반이며 인과 효과가 아니다.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ---------------- 설정(가정) ----------------
MIN_CELL = 50                      # 셀 최소 표본(매출 0 규칙 제외). 미만이면 '구조 (전체)'로 대체
ELASTICITY = 0.555                 # 마케팅비 탄력성 (+50% 시나리오 계산에 사용)
RAISE = 0.5                        # 마케팅비 증액 비율
FEAS = {"충돌률": 1.0, "업데이트": 0.7, "플레이타임": 0.6}   # 실행 가능성 가중
GRADE_HI, GRADE_MID = 55, 30       # 등급 기준: |품질 백분위 − 마케팅비 백분위|
QUALITY_COLS = ["lpt", "lsm", "crash", "upd", "lwl", "lapk"]  # 품질에는 마케팅비를 넣지 않는다
QUALITY_CATS = ["mon", "pub", "genre", "region_focus", "multiplayer_support"]
GENRES = ["Casual", "Puzzle", "RPG", "Racing", "Simulation", "Sports", "Strategy"]  # Action이 기준
NOTE = {
    "대상": "입력한 게임 (D7 시점)",
    "셀": "장르 × 수익화 구조(광고/IAP 조합: 광고+IAP·광고만·IAP만·없음), 50개 미만 셀은 같은 수익화 구조 전체로 대체",
    "예측매출": "pred_log 입력 시 팀 앙상블 예측, 없으면 내부 회귀 모델(마케팅 포함) 예측",
    "품질": "마케팅비를 뺀 D7 지표로 예측한 로그 매출(선형회귀, train 교차적합)의 셀 안 백분위",
    "한계ROAS": f"마케팅비 +50% 시 추가 $1당 예측 매출 (탄력성 {ELASTICITY})",
    "사분면": "품질·마케팅비 셀 백분위 50 기준: 🟢 품질↑마케팅↓ / 🔵 둘 다↑ / 🟡 둘 다↓ / 🔴 품질↓마케팅↑ / ⚪ 마케팅비 없음 / ⛔ 수익 수단 없음",
    "품질개선변수": "충돌률·플레이타임·업데이트 중 셀 중앙값까지 개선 시 (예상 추가 매출 × 실행 가능성)이 가장 큰 것. 🟢는 마케팅비 증액, 🔵는 유지",
    "주의": "모든 예상치는 상관 기반이며 인과 효과가 아님",
}


# ---------------- 전처리 ----------------
def prep(d):
    d = d.copy()
    d["lpt"] = np.log(d.avg_session_minutes * d.avg_daily_sessions)
    d["lsm"] = np.log(d.avg_session_minutes)
    d["crash"] = d.crash_rate_pct.clip(lower=0)                        # -1 같은 이상치 보정
    d["upd"] = d.update_frequency_days
    d["lwl"] = np.log(d.wishlist_prelaunch)
    d["lapk"] = np.log(d.apk_size_mb)
    d["play"] = d.avg_session_minutes * d.avg_daily_sessions           # 하루 플레이타임(분)
    d["lmk"] = np.log(d.marketing_spend_usd)                           # 결측은 NaN 유지
    d["mon"] = d.monetization_model + "|" + d.contains_ads + "|" + d.has_in_app_purchases
    d["pub"] = d.publisher_tier
    ads, iap = d.contains_ads == "Yes", d.has_in_app_purchases == "Yes"
    d["struct"] = np.select([ads & iap, ads, iap], ["광고+IAP", "광고만", "IAP만"], "없음")
    d["zero"] = (d.monetization_model.isin(["Free", "Freemium"]) & ~ads & ~iap)   # 매출 0 규칙
    d["cell"] = d.genre + " × " + d.struct
    return d


def ols(X, y):
    A = np.c_[np.ones(len(X)), X.values.astype(float)]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def predict(X, b):
    return np.c_[np.ones(len(X)), X.values.astype(float)] @ b


class Advisor:
    def __init__(self, train_csv):
        tr = prep(pd.read_csv(train_csv))
        cnt = tr[~tr.zero].groupby("cell").size()
        self.small = set(cnt[cnt < MIN_CELL].index)
        tr["cell"] = [self._cell_of(c, s) for c, s in zip(tr.cell, tr.struct)]
        trn = tr[~tr.zero].reset_index(drop=True)
        self.med_mk = float(trn.marketing_spend_usd.median())
        y = np.log1p(trn.total_revenue_usd.values)

        # 품질 지수: 마케팅 제외 회귀. train은 5-fold 교차적합 (자기 매출이 새지 않게)
        Xq = self._design(trn); self.cols_q = list(Xq.columns); trn["Q"] = 0.0
        for a, b in KFold(5, shuffle=True, random_state=1).split(trn):
            trn.loc[trn.index[b], "Q"] = predict(Xq.iloc[b], ols(Xq.iloc[a], y[a]))
        self.bq = ols(Xq, y); self.coefq = dict(zip(["_int"] + self.cols_q, self.bq))
        # 예측 매출 대체용 회귀(마케팅 포함)
        Xf = self._design(trn, True); self.cols_f = list(Xf.columns); self.bf = ols(Xf, y)

        # 비교군 기준
        self.ref_q = {k: np.sort(g.Q.values) for k, g in trn.groupby("cell")}
        self.ref_mk = {k: np.sort(g.marketing_spend_usd.dropna().values) for k, g in trn.groupby("cell")}
        grp = trn.groupby("cell")
        self.med = pd.concat({"crash": grp.crash.median(), "upd": grp.upd.median(), "play": grp.play.median(), "n": grp.size()}, axis=1)

    def _cell_of(self, cell, struct):
        return f"{struct} (전체)" if cell in self.small else cell

    def _design(self, d, with_mk=False):
        X = d[QUALITY_COLS].copy()
        if with_mk:
            X["lmk"] = d.lmk.fillna(np.log(getattr(self, "med_mk", d.marketing_spend_usd.median())))
            for p in ["Mid-size", "AAA", "Unknown"]:
                X["mk*" + p] = (d.pub == p) * X.lmk
        for g in GENRES:
            X["pt*" + g] = (d.genre == g) * d.lpt                      # 장르별 플레이타임 기울기
        X = pd.concat([X, pd.get_dummies(d[QUALITY_CATS]).astype(float)], axis=1)
        cols = getattr(self, "cols_f" if with_mk else "cols_q", None)
        return X.reindex(columns=cols, fill_value=0.0) if cols else X

    # ---------------- 진단 ----------------
    def diagnose(self, games, budget=None):
        """games: dict / list[dict] / DataFrame. 입력한 게임들에 대해서만 결과를 낸다."""
        if isinstance(games, dict): games = [games]
        df = pd.DataFrame(games) if not isinstance(games, pd.DataFrame) else games.copy()
        df = df.reset_index(drop=True)
        if "game_id" not in df: df["game_id"] = [f"input_{i}" for i in range(len(df))]
        if "marketing_spend_usd" not in df: df["marketing_spend_usd"] = np.nan
        d = prep(df)
        d["cell"] = [self._cell_of(c, s) for c, s in zip(d.cell, d.struct)]
        d["Q"] = predict(self._design(d), self.bq)
        base = d.pred_log.values if "pred_log" in d else predict(self._design(d, True), self.bf)
        d["E"] = np.expm1(base)
        d.loc[d.zero, "E"] = 0.0
        out = [self._one(r) for _, r in d.iterrows()]

        # 여러 개 + 예산: 입력한 게임들 안에서만 🟢 증액을 점수 순으로 배정
        if budget is not None:
            left = float(budget)
            green = [g for g in out if g["사분면"] == "🟢"]
            for g in sorted(green, key=lambda g: g["우선순위점수"], reverse=True):
                amt = g["처방"]["증액_usd"]
                ok = amt <= left
                g["처방"]["예산내"] = bool(ok)
                if ok: left -= amt
            self.last_budget_used = float(budget) - left
        return out

    def _pct(self, ref, v):
        return round(float(np.searchsorted(ref, v) / len(ref) * 100), 1) if len(ref) else None

    def _one(self, r):
        cs = self.med.loc[r.cell]
        ref_q, ref_mk = self.ref_q[r.cell], self.ref_mk[r.cell]
        qp = self._pct(ref_q, r.Q)
        S = r.marketing_spend_usd; has_mk = S == S and S > 0
        mkp = self._pct(ref_mk, S) if has_mk else None
        E = float(r.E)
        add = E * (1.5 ** ELASTICITY - 1)                              # +50% 증액 시 추가 매출
        mk = {"마케팅비_usd": round(S, 0) if has_mk else None, "마케팅비_셀백분위": mkp,
              "예측매출_usd": round(E, 0),
              "예측ROAS": round(E / S, 2) if has_mk else None,
              "한계ROAS": round(add / (RAISE * S), 2) if has_mk else None,
              "마케팅비50%증액시_추가매출_usd": round(add, 0)}
        # 품질 변수별 '셀 중앙값까지 개선 시' 추가 매출
        slope = self.coefq["lpt"] + self.coefq.get("pt*" + r.genre, 0.0)
        gains = {}
        c = max(r.crash - cs.crash, 0); gains["충돌률"] = E * (np.exp(-self.coefq["crash"] * c) - 1)
        u = max(r.upd - cs.upd, 0); gains["업데이트"] = E * (np.exp(-self.coefq["upd"] * u) - 1)
        p = np.log(cs.play / r.play) if r.play < cs.play else 0.0; gains["플레이타임"] = E * (np.exp(slope * p) - 1)
        ql = {"품질_셀백분위": qp,
              "충돌률": {"현재": round(float(r.crash), 2), "셀중앙값": round(float(cs.crash), 2), "중앙값까지개선시_추가매출_usd": round(gains["충돌률"], 0)},
              "하루플레이타임_분": {"현재": round(float(r.play), 1), "셀중앙값": round(float(cs.play), 1), "중앙값까지개선시_추가매출_usd": round(gains["플레이타임"], 0)},
              "업데이트간격_일": {"현재": round(float(r.upd), 1), "셀중앙값": round(float(cs.upd), 1), "중앙값까지개선시_추가매출_usd": round(gains["업데이트"], 0)}}

        # 사분면·등급·점수
        score = None
        if r.zero:
            quad, grade = "⛔", "대상 제외"
        elif not has_mk:
            quad, grade = "⚪", "판단 보류"
        else:
            hi_q, hi_m = qp >= 50, mkp >= 50
            quad = "🟢" if hi_q and not hi_m else "🔵" if hi_q and hi_m else "🔴" if (not hi_q and hi_m) else "🟡"
            dd = abs(qp - mkp)
            grade = "고" if dd >= GRADE_HI else "중" if dd >= GRADE_MID else "저"
            score = round(qp * (100 - mkp) / 10000, 3)                 # 품질 백분위 × (1 − 마케팅비 백분위)

        # 처방
        if r.zero:
            rx = {"축": "차단", "솔루션": "집행 보류", "품질개선변수": None,
                  "사유": "Free/Freemium인데 광고도 IAP도 없어 90일 매출이 0으로 확정"}
        elif quad == "🟢":
            rx = {"축": "예산", "솔루션": "예산", "품질개선변수": None, "증액_usd": round(RAISE * S, 0)}
        elif quad == "🔵":
            rx = {"축": "유지", "솔루션": "유지", "품질개선변수": None}
        else:
            best = max(gains, key=lambda k: gains[k] * FEAS[k])
            if gains[best] <= 0:
                rx = {"축": "품질", "솔루션": "유지", "품질개선변수": None}
            else:
                cur, tgt, unit = {"충돌률": (r.crash, cs.crash, "%"), "업데이트": (r.upd, cs.upd, "일"), "플레이타임": (r.play, cs.play, "분")}[best]
                rx = {"축": "품질", "솔루션": best,
                      "품질개선변수": {"변수": best, "현재값": round(float(cur), 2), "목표값_셀중앙값": round(float(tgt), 2), "단위": unit,
                                  "예상매출배수": round(1 + gains[best] / E, 2) if E > 0 else None,
                                  "예상추가매출_usd": round(gains[best], 0)}}
        return {"game_id": r.game_id, "장르": r.genre, "수익화구조_광고IAP": r.struct, "과금방식": r.monetization_model,
                "셀": r.cell, "마케팅": mk, "품질": ql, "사분면": quad, "등급": grade, "우선순위점수": score, "처방": rx}


def summary(g):
    m, q, rx = g["마케팅"], g["품질"], g["처방"]
    mp = f"{m['마케팅비_셀백분위']}" if m["마케팅비_셀백분위"] is not None else "없음"
    lines = [f"\n[{g['game_id']}] {g['셀']} ({g['과금방식']}) {g['사분면']} 등급 {g['등급']}",
             f"  품질 백분위 {q['품질_셀백분위']} · 마케팅비 백분위 {mp} · 우선순위 점수 {g['우선순위점수']} · 예측 90일 매출 ${m['예측매출_usd']:,.0f}"]
    if g["사분면"] == "🟢":
        lines.append(f"  솔루션: 마케팅비 +50% (+${rx['증액_usd']:,.0f}) → 추가 매출 약 ${m['마케팅비50%증액시_추가매출_usd']:,.0f}, 한계 ROAS {m['한계ROAS']}"
                     + (f"  [예산 {'내' if rx['예산내'] else '밖'}]" if "예산내" in rx else ""))
    elif rx["솔루션"] == "집행 보류":
        lines.append(f"  솔루션: 마케팅 집행 보류 ({rx['사유']})")
    elif rx["품질개선변수"]:
        v = rx["품질개선변수"]
        lines.append(f"  솔루션: {v['변수']} {v['현재값']}{v['단위']} → {v['목표값_셀중앙값']}{v['단위']} (셀 중앙값) · 예상 매출 ×{v['예상매출배수']} (+${v['예상추가매출_usd']:,.0f})")
    else:
        lines.append(f"  솔루션: {rx['솔루션']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train.csv (비교군 기준을 만드는 데 사용)")
    ap.add_argument("--game", help="게임 1개 JSON 파일 경로 (예: example_game.json) 또는 JSON 문자열")
    ap.add_argument("--games", help="여러 게임 CSV 또는 JSON(list 또는 {games:[...]})")
    ap.add_argument("--budget", type=float, default=None, help="마케팅 증액 예산(USD). 입력한 게임들끼리만 나눔")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    a = ap.parse_args()
    adv = Advisor(a.train)
    if a.game:
        games = json.load(open(a.game, encoding='utf-8')) if a.game.strip().lower().endswith('.json') else json.loads(a.game)
    elif a.games.endswith(".json"):
        j = json.load(open(a.games, encoding='utf-8')); games = j["games"] if isinstance(j, dict) else j
    else: games = pd.read_csv(a.games)
    res = adv.diagnose(games, a.budget)
    for g in res: print(summary(g))
    if a.budget is not None and len(res) > 1:
        n = sum(1 for g in res if g["처방"].get("예산내"))
        print(f"\n예산 ${a.budget:,.0f} 중 증액 {n}건에 ${adv.last_budget_used:,.0f} 배정 (우선순위 점수 높은 순)")
    if a.out:
        json.dump({"설명": NOTE, "games": res}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
