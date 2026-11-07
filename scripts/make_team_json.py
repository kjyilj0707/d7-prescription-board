import json, numpy as np, pandas as pd
g=pd.read_csv('outputs/mvp/games_mvp.csv'); g=g[g.split=='test']
LV={"충돌률":("crash_rate_pct","med_crash_rate_pct","%"),"플레이타임":("playtime","med_playtime","분"),"업데이트":("update_frequency_days","med_update_frequency_days","일")}
r2=lambda v,n=2: None if pd.isna(v) else round(float(v),n)
recs=[]
for _,r in g.iterrows():
    sol=r['솔루션']
    if sol in LV:
        c,m,u=LV[sol]; imp={"변수":sol,"현재값":r2(r[c]),"목표값_셀중앙값":r2(r[m]),"단위":u,"예상매출배수":r2(r['품질솔루션_배수']),"예상추가매출_usd":r2(r['예상추가매출'],0)}
    else: imp=None
    recs.append({
      "game_id":r.game_id,"장르":r.genre,"수익화구조_광고IAP":r["채널"],"과금방식":r.monetization_model,"셀":r['비교셀'],
      "마케팅":{"마케팅비_usd":r2(r.marketing_spend_usd,0),"마케팅비_셀백분위":r2(r['마케팅_pct'],1),"예측매출_usd":r2(r.pred_usd,0),
               "예측ROAS":r2(r.roas),"한계ROAS":r2(r['한계roas']),"마케팅비50%증액시_추가매출_usd":r2(r['추가매출_예산'],0)},
      "품질":{"품질_셀백분위":r2(r['품질_pct'],1),
             "충돌률":{"현재":r2(r.crash_rate_pct),"셀중앙값":r2(r.med_crash_rate_pct),"중앙값까지개선시_추가매출_usd":r2(r['추가매출_충돌률'],0)},
             "하루플레이타임_분":{"현재":r2(r.playtime,1),"셀중앙값":r2(r.med_playtime,1),"중앙값까지개선시_추가매출_usd":r2(r['추가매출_플레이타임'],0)},
             "업데이트간격_일":{"현재":r2(r.update_frequency_days,0),"셀중앙값":r2(r.med_update_frequency_days,0),"중앙값까지개선시_추가매출_usd":r2(r['추가매출_업데이트'],0)}},
      "사분면":r['사분면'],"등급":r['등급'],
      "처방":{"축":r['처방축'],"솔루션":sol,"품질개선변수":imp}})
out={"설명":{"대상":"test 게임 1,831개 (D7 시점, 수익 수단 없는 169개 제외)","셀":"장르 × 수익화 구조(광고/IAP 조합: 광고+IAP·광고만·IAP만·없음), 50개 미만 셀은 같은 수익화 구조 전체로 대체",
 "예측매출":"팀 최종 앙상블 CV RMSLE 0.6741","품질":"마케팅비를 뺀 D7 지표(플레이타임·충돌률·업데이트·위시리스트·앱 용량 + 수익원 조합·퍼블리셔·장르·지역·멀티플레이)로 예측한 로그 매출(선형회귀, train 교차적합)의 셀 안 백분위",
 "한계ROAS":"마케팅비 +50% 시 추가 $1당 예측 매출 (탄력성 0.555)","사분면":"품질·마케팅비 셀 백분위 50 기준: 🟢 품질↑마케팅↓ / 🔵 둘 다↑ / 🟡 둘 다↓ / 🔴 품질↓마케팅↑ / ⚪ 마케팅비 없음",
 "품질개선변수":"충돌률·플레이타임·업데이트 중 셀 중앙값까지 개선 시 (예상 추가 매출 × 실행 가능성)이 가장 큰 것. 🟢는 마케팅비 증액, 🔵는 유지",
 "주의":"모든 예상치는 상관 기반이며 인과 효과가 아님"},"games":recs}
json.dump(out,open('outputs/mvp/팀원용_게임별_처방.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(len(recs)); print(json.dumps(recs[[x['game_id'] for x in recs].index('REV4_003355')],ensure_ascii=False,indent=1))
