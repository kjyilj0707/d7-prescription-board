"""src/mvp_template.html + data/*.json → index.html (GitHub Pages 배포본)"""
from pathlib import Path
root = Path(__file__).parent
t = (root / "src/mvp_template.html").read_text(encoding="utf-8")
page = t.replace("__DATA__", (root / "data/games_test.json").read_text(encoding="utf-8")).replace("__ENGINE__", (root / "data/engine.json").read_text(encoding="utf-8"))
(root / "index.html").write_text("<!doctype html><html lang='ko' style='color-scheme:light'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'></head><body>" + page + "</body></html>", encoding="utf-8")
print("index.html 생성:", round(len(page) / 1e6, 2), "MB")
