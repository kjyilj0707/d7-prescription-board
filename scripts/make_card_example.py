import json, sys, pandas as pd
g=pd.read_csv('outputs/mvp/games_mvp.csv').set_index('game_id')
Q={"🟢":"기회 · 예산 증액","🔵":"유지","🟡":"품질 먼저","🔴":"위험 · 품질 개선 + 예산 재검토"}
usd=lambda v:f"${v:,.0f}"
def card(i):
    r=g.loc[i]; s=r['솔루션']; c={"game_id":i,"quadrant":r['사분면'],"grade":r['등급'],"cell":r['비교셀'],"대상":f"{i} ({r['비교셀']}, {r['사분면']} {Q[r['사분면']]})","레버":s}
    if s=="충돌률": c.update(가설=f"충돌률을 {r.crash_rate_pct:.2f}% → 셀 중앙값 {r.med_crash_rate_pct:.2f}%로 낮추면 이용시간과 매출이 개선될 것 (품질 모델 기준 예상 ×{r['품질솔루션_배수']:.2f}, 상관 기반)",실행="상위 크래시 원인 3개 핫픽스 → 일부 이용자 단계적 배포",성공지표=f"충돌률 {r.med_crash_rate_pct:.1f}% 이하 · 하루 플레이타임 · D30 매출",관찰기간="2주",예상효과=f"+{usd(r['예상추가매출'])} (예상치)")
    elif s=="플레이타임": c.update(가설=f"하루 플레이타임 {r.playtime:.0f}분 → 셀 중앙값 {r.med_playtime:.0f}분으로 늘리면 매출이 개선될 것 (예상 ×{r['품질솔루션_배수']:.2f}, 상관 기반)",실행="데일리 보상·푸시 알림으로 하루 세션 수 늘리기 A/B" if r.avg_daily_sessions<r.med_avg_daily_sessions else "세션당 콘텐츠(퀘스트·이벤트) 깊이 강화 A/B",성공지표="하루 세션 수 · 하루 플레이타임 · D30 매출",관찰기간="4주",예상효과=f"+{usd(r['예상추가매출'])} (예상치)")
    elif s=="업데이트": c.update(가설=f"업데이트 간격 {r.update_frequency_days:.0f}일 → 셀 중앙값 {r.med_update_frequency_days:.0f}일로 줄이면 매출이 개선될 것 (예상 ×{r['품질솔루션_배수']:.2f}, 상관 기반)",실행="소규모 콘텐츠 업데이트 주기 단축 (격주 패치)",성공지표="업데이트 간격 · 복귀 이용자 · D30 매출",관찰기간="4주",예상효과=f"+{usd(r['예상추가매출'])} (예상치)")
    elif s=="예산": c.update(가설=f"품질은 셀 상위인데 마케팅비는 하위 → 마케팅비 +50% 시 추가 $1당 예측 매출 {r['한계roas']:.1f}",실행=f"마케팅비 {usd(r.marketing_spend_usd)} → {usd(r.marketing_spend_usd*1.5)} 단계 집행 (1주 단위 점검)",성공지표="실제 ROAS · 유료 설치 수 · D30 매출",관찰기간="2–4주",예상효과=f"+{usd(r['추가매출_예산'])} (예상치)")
    else: c.update(가설="현재 품질·예산 균형 유지 시 매출 유지",실행="주간 지표 모니터링 (충돌률·플레이타임·ROAS)",성공지표="충돌률 · 플레이타임 · ROAS",관찰기간="4주",예상효과="–")
    c["근거"]={"pred_revenue_usd":round(r.pred_usd),"marketing_spend_usd":round(r.marketing_spend_usd),"roas":round(r.roas,2),"marginal_roas":round(r['한계roas'],2),"quality_pct":round(r['품질_pct'],1),"marketing_pct":round(r['마케팅_pct'],1)}
    return c
ids=sys.argv[1].split(',')
out={"description":f"D7 처방 보드 실험카드 예시 (시연 셀: {g.loc[ids[0],'비교셀']}, 셀 = 장르 × 수익화 구조[광고/IAP 조합]). 숫자는 팀 최종 모델 예측·품질 모델에서 계산, 모든 효과는 상관 기반 예상치","cards":[card(i) for i in ids]}
json.dump(out,open('outputs/mvp/실험카드_예시.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print("실험카드_예시.json:", ids)
