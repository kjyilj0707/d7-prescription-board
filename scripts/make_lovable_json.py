import json
d=json.load(open('outputs/mvp/games_test.json',encoding='utf-8'))
ren={"채널":"channel","셀":"cell","비교셀":"ref_cell","비교셀_n":"ref_cell_n","한계roas":"marginal_roas","추가매출_예산":"gain_budget",
"품질지수":"quality_score","품질_pct":"quality_pct","마케팅_pct":"marketing_pct","불균형":"gap","사분면":"quadrant","등급":"grade",
"우선순위점수":"priority","추가매출_플레이타임":"gain_playtime","추가매출_충돌률":"gain_crash","추가매출_업데이트":"gain_update",
"품질솔루션":"quality_solution","품질솔루션_배수":"quality_multiplier","처방축":"axis","솔루션":"solution","예상추가매출":"expected_gain"}
vmap={"플레이타임":"playtime","충돌률":"crash","업데이트":"update","예산":"budget","유지":"keep","없음":"none","품질":"quality",
"고":"high","중":"mid","저":"low","판단 보류":"pending","🟢":"green","🔵":"blue","🟡":"yellow","🔴":"red","⚪":"gray"}
cols=[ren.get(c,c) for c in d["cols"]]; games=[]
for r in d["rows"]:
    g=dict(zip(cols,r))
    for k in ["quadrant","grade","axis","solution","quality_solution"]: g[k]=vmap.get(g[k],g[k])
    games.append(g)
cells=[{"ref_cell":c["비교셀"],"n":c["n"],"median_roas":c["roas"],"median_marginal_roas":c["mroas"],"median_quality":c["quality"],"median_pred_usd":c["pred"],"median_marketing":c["mkt"]} for c in d["cells"]]
out={"meta":{"model":"Team final ensemble, CV RMSLE 0.6741","zero_rule_games_excluded":d["zero_test"],"quality_coef":d["quality_coef"],
 "marketing_elasticity":0.555,"quadrant_rule":"pct 50 split within ref_cell","grade_rule":"|gap|>=55 high, >=30 mid, else low; gray=pending"},"cells":cells,"games":games}
json.dump(out,open('outputs/mvp/lovable_games.json','w',encoding='utf-8'),ensure_ascii=False)
print("lovable_games.json 재생성:", len(games), "games")
