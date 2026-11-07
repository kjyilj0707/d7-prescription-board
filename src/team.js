// 팀 앙상블 모델(team_model.py)을 브라우저에서 그대로 계산 (HGB 트리 + Ridge)
// TEAM.load(meta, arrayBuffer) 후 TEAM.predictLog(rows) → log1p(90일 매출)
const TEAM = (() => {
  let M = null, A, F, L, R, FL, BS, TS;
  function load(meta, buf) {
    M = meta;
    const h = new Int32Array(buf, 0, 3), nn = h[0], nt = h[1], nb = h[2];
    let o = 12;
    TS = new Int32Array(buf, o, nt); o += nt * 4;
    A = new Float64Array(buf.slice(o, o + nn * 8)); o += nn * 8;
    F = new Int16Array(buf.slice(o, o + nn * 2)); o += nn * 2;
    L = new Uint8Array(buf, o, nn); o += nn;
    R = new Uint8Array(buf, o, nn); o += nn;
    FL = new Uint8Array(buf, o, nn); o += nn;
    BS = new Uint32Array(buf.slice(o, o + nb * 32));
    M.catIdx = Object.fromEntries(M.cats.map(c => [c, Object.fromEntries(M.catlv[c].map((v, i) => [v, i]))]));
  }
  const inBits = (arr, row, v) => (arr[row * 8 + (v >>> 5)] >>> (v & 31)) & 1;
  function tree(t, x, known, fmap) {
    let i = TS[t];
    const base = i;
    for (;;) {
      const fl = FL[i];
      if (fl & 1) return A[i];
      const v = x[F[i]], ml = fl & 2;
      let left;
      if (Number.isNaN(v)) left = ml;
      else if (fl & 4) {
        if (v < 0) left = ml;
        else if (inBits(BS, A[i], v)) left = 1;
        else if ((known[fmap[F[i]]][v >>> 5] >>> (v & 31)) & 1) left = 0;
        else left = ml;
      } else left = v <= A[i];
      i = base + (left ? L[i] : R[i]);
    }
  }
  function model(k, x0) {
    const m = M.models[k];
    // sklearn 내부 전처리: 범주 열을 앞으로 모으고 학습 때 본 값의 순번으로 바꿈 (처음 보는 값 = 결측)
    const x = m.order.map((j, i) => { const v = x0[j]; if (i >= m.ncat) return v; if (Number.isNaN(v) || v < 0) return NaN; const r = m.ordc[i].indexOf(v); return r < 0 ? NaN : r; });
    let s = m.base;
    for (let t = m.t0; t < m.t0 + m.nt; t++) s += tree(t, x, m.known, m.fmap);
    return s;
  }
  const days = (a, b) => Math.round((a - b) / 864e5);
  function prep(g) {
    const d = { ...g };
    d.price_usd = g.price_usd == null || g.price_usd < 0 ? NaN : +g.price_usd;
    d.crash_rate_pct = g.crash_rate_pct == null || g.crash_rate_pct < 0 ? NaN : +g.crash_rate_pct;
    const rel = new Date(g.release_date + "T00:00:00Z"), soft = g.soft_launch_date ? new Date(g.soft_launch_date + "T00:00:00Z") : null;
    d.has_soft_launch = soft ? 1 : 0;
    d.days_soft_to_release = soft ? days(rel, soft) : -1;
    d.release_year = rel.getUTCFullYear(); d.release_month = rel.getUTCMonth() + 1;
    d.release_dayofweek = (rel.getUTCDay() + 6) % 7; d.release_day = rel.getUTCDate();
    const yn = v => (v === "Yes" || v === 1 || v === true) ? 1 : 0;
    d.contains_ads = yn(g.contains_ads); d.has_in_app_purchases = yn(g.has_in_app_purchases);
    for (const c of ["is_weekend_release", "cross_platform_available", "cloud_save_support"]) d[c] = g[c] ? 1 : 0;
    if (M.rare.includes(g.sub_genre)) d.sub_genre = "Other";
    d.monet_combo = `${g.monetization_model}_a${d.contains_ads}_i${d.has_in_app_purchases}`;
    const mk = g.marketing_spend_usd;
    d.log_marketing = mk == null || mk === "" || Number.isNaN(+mk) ? NaN : Math.log1p(+mk);
    d.log_wishlist = Math.log1p(+g.wishlist_prelaunch); d.log_apk = Math.log1p(+g.apk_size_mb);
    d.log_playtime = Math.log1p(g.avg_session_minutes * g.avg_daily_sessions);
    return d;
  }
  function hgbX(d) {
    return M.feats.map(c => {
      if (M.cats.includes(c)) { const v = M.catIdx[c][String(d[c])]; return v == null ? -1 : v; }
      const v = d[c]; return v == null ? NaN : +v;
    });
  }
  function ridge(p, d) {
    let s = p.b, j = 0;
    M.nums.forEach((c, i) => { let v = d[c]; v = (v == null || Number.isNaN(+v)) ? p.med[i] : +v; s += p.coef[j++] * (v - p.mean[i]) / p.scale[i]; });
    M.cats.forEach((c, i) => { const lv = p.cats[i]; const k = lv.indexOf(String(d[c])); if (k >= 0) s += p.coef[j + k]; j += lv.length; });
    return s;
  }
  function predictLog(g) {
    const d = prep(g), x = hgbX(d), N = M.N, S = M.seeds;
    const idx = name => M.models.findIndex(m => m.name === name);
    let ph = 0; for (let i = 0; i < N * S; i++) ph += model(i, x); ph /= N * S;
    let pr = 0;
    for (let k = 0; k < N; k++) { const bt = ridge(M.ridges[k], d); for (let s = 0; s < S; s++) pr += (bt + model(idx(`resid${k}_${s}`), x)) / N / S; }
    const mask = { paid: d.monetization_model === "Paid", sub: d.monetization_model === "Subscription", ad: d.contains_ads === 1, iap: d.has_in_app_purchases === 1 };
    let usd = 0;
    for (const n of ["paid", "sub", "ad", "iap"]) {
      if (!mask[n]) continue;
      for (let k = 0; k < N; k++) { let pt = 0; for (let s = 0; s < S; s++) pt += model(idx(`ch_${n}${k}_${s}`), x); pt /= S; usd += Math.expm1(Math.max(pt, 0)) / N; }
    }
    let p = Math.max(M.W.hgb * ph + M.W.resid * pr + M.W.ch * Math.log1p(usd), 0);
    if ((d.monetization_model === "Free" || d.monetization_model === "Freemium") && !d.contains_ads && !d.has_in_app_purchases) p = 0;
    return p;
  }
  return { load, predictLog, prep, get meta() { return M; } };
})();
if (typeof module !== "undefined") module.exports = TEAM;
