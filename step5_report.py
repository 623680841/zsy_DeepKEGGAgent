import os, glob, base64, pandas as pd
from datetime import datetime

BASE = "/tmp/DeepKEGG-agent"
RUNS = f"{BASE}/runs"
OUT_HTML = os.path.join(RUNS, "summary.html")
ALL_CSV = os.path.join(RUNS, "all_models_metrics.csv")

def latest_dir_with_suffix(suffix):
    # 找 runs 下以 suffix 结尾的目录，按时间排序取最后一个
    cands = [p for p in glob.glob(os.path.join(RUNS, "*"+suffix)) if os.path.isdir(p)]
    if not cands:
        return None
    cands.sort(key=os.path.getmtime)
    return cands[-1]

def b64img(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

def main():
    # 读取汇总表（如果没有就只生成空表壳）
    if os.path.exists(ALL_CSV):
        df = pd.read_csv(ALL_CSV)
    else:
        # 尝试从各目录抓取 metrics.csv 组装
        rows = []
        for suf in ["_LR","_SVM","_XGB","_DeepKEGG"]:
            d = latest_dir_with_suffix(suf)
            if d and os.path.exists(os.path.join(d,"metrics.csv")):
                t = pd.read_csv(os.path.join(d,"metrics.csv"))
                t.insert(0, "Model", suf.strip("_"))
                rows.append(t)
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # 捕获图像
    blocks = []
    for name, suf in [("LR","_LR"), ("SVM","_SVM"), ("XGBoost","_XGB")]:
        d = latest_dir_with_suffix(suf)
        if not d: 
            continue
        roc64 = b64img(os.path.join(d,"roc.png"))
        pr64  = b64img(os.path.join(d,"pr.png"))
        blocks.append((name, d, roc64, pr64))

    # 渲染 HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_html = (df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
                  if not df.empty else "<p><i>暂无汇总表</i></p>")

    imgs_html = ""
    for name, d, roc64, pr64 in blocks:
        imgs_html += f"""
        <div class="card">
          <div class="card-title">{name} · {os.path.basename(d)}</div>
          <div class="row">
            <div class="col">
              <div class="img-title">ROC</div>
              {'<img src="'+roc64+'" />' if roc64 else '<div class="placeholder">无 roc.png</div>'}
            </div>
            <div class="col">
              <div class="img-title">PR</div>
              {'<img src="'+pr64+'" />' if pr64 else '<div class="placeholder">无 pr.png</div>'}
            </div>
          </div>
        </div>
        """

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>DeepKEGG-Agent Summary</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 24px; }}
h1 {{ margin: 0 0 8px 0; }}
.subtitle {{ color:#555; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background: #f6f6f6; }}
.card {{ border:1px solid #eee; border-radius:12px; padding:16px; margin-bottom:18px; box-shadow:0 2px 6px rgba(0,0,0,0.04); }}
.card-title {{ font-weight:600; margin-bottom:8px; }}
.row {{ display:flex; gap:16px; }}
.col {{ flex:1; }}
.img-title {{ font-size:14px; color:#444; margin: 6px 0; }}
img {{ width:100%; height:auto; border:1px solid #ddd; border-radius:8px; }}
.placeholder {{ height:200px; border:1px dashed #ccc; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#888; }}
.footer {{ color:#777; font-size:12px; margin-top:12px; }}
</style>
</head>
<body>
  <h1>DeepKEGG-Agent · 一页式报告</h1>
  <div class="subtitle">生成时间：{now} · 目录：{RUNS}</div>

  <h2>汇总表</h2>
  {table_html}

  <h2>各模型 ROC / PR</h2>
  {imgs_html}

  <div class="footer">说明：表格来源 runs/all_models_metrics.csv；图像取各模型最新一次运行目录中的 roc.png/pr.png。</div>
</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ 已生成：", OUT_HTML)

if __name__ == "__main__":
    main()
