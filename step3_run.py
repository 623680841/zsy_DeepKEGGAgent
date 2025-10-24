import os, json, time, hashlib, tempfile, shutil, subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG_PATH = "/tmp/DeepKEGG-agent/last_run_config.json"
HOOK_PY  = "/tmp/DeepKEGG-agent/hooks/run_deepkegg_fold.py"

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_matrix_auto_id(csv_path):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} 列数不足，无法作为矩阵")
    id_col = df.columns[0]
    df = df.set_index(id_col)
    df = df.dropna(axis=1, how="all")
    return df

def load_omics(paths, omics_selected):
    mats = []
    if "mRNA" in omics_selected and paths.get("mrna"):
        m = read_matrix_auto_id(paths["mrna"]); m.columns=[f"mRNA::{c}" for c in m.columns]; mats.append(m)
    if "miRNA" in omics_selected and paths.get("mirna"):
        m = read_matrix_auto_id(paths["mirna"]); m.columns=[f"miRNA::{c}" for c in m.columns]; mats.append(m)
    if "SNV" in omics_selected and paths.get("snv"):
        m = read_matrix_auto_id(paths["snv"]); m.columns=[f"SNV::{c}" for c in m.columns]; mats.append(m)
    if not mats: raise ValueError("未选择任何组学或路径为空")
    common = set(mats[0].index)
    for m in mats[1:]:
        common &= set(m.index)
    if not common: raise ValueError("多组学样本交集为空，请检查数据")
    common = sorted(common)
    mats = [m.loc[common] for m in mats]
    return pd.concat(mats, axis=1)

def load_labels(clinical_path, id_hint=None, label_col_hint="response"):
    df = pd.read_csv(clinical_path)
    id_col = id_hint if (id_hint in df.columns) else df.columns[0]
    if label_col_hint not in df.columns:
        for cand in ["label","y","outcome","Response","STATUS","status"]:
            if cand in df.columns: label_col_hint = cand; break
    if label_col_hint not in df.columns:
        raise ValueError(f"找不到标签列，请确认 clinical 中是否包含 {label_col_hint}")
    lab = df[[id_col, label_col_hint]].dropna().copy()
    lab[label_col_hint] = lab[label_col_hint].astype(int)
    return lab.set_index(id_col)

def ensure_outdir(root, cfg, model_tag):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    key = json.dumps({
        "cancer": cfg["cancer"], "omics": cfg["omics"], "model": cfg["model"],
        "cv_folds": cfg["cv_folds"], "cv_repeats": cfg["cv_repeats"], "k": cfg.get("k",64)
    }, sort_keys=True).encode("utf-8")
    tag = hashlib.md5(key).hexdigest()[:8]
    outdir = os.path.join(root, f'{cfg["cancer"]}_{stamp}_{tag}_{model_tag}')
    os.makedirs(outdir, exist_ok=True)
    return outdir

def model_short(name: str) -> str:
    if "LR" in name: return "LR"
    if "SVM" in name: return "SVM"
    if "XGBoost" in name or "XGB" in name: return "XGB"
    if "DeepKEGG" in name: return "DeepKEGG"
    return "UNK"

def pos_weight_from_y(y: np.ndarray) -> float:
    pos = (y==1).sum(); neg = (y==0).sum()
    return float(neg)/max(1.0, float(pos))

def make_model(name: str, y: np.ndarray):
    name = str(name)
    if "LR" in name:
        return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    if "SVM" in name:
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True)
    if "XGBoost" in name or "XGB" in name:
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("需要安装 xgboost： pip install xgboost") from e
        return XGBClassifier(
            booster="gbtree",
            n_estimators=600, max_depth=2, min_child_weight=5, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=2.0,
            tree_method="hist", eval_metric="logloss",
            scale_pos_weight=pos_weight_from_y(y),
            random_state=42
        )
    if "DeepKEGG" in name:
        return None  # 用 hook 处理
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")

def plot_roc_pr(y_true, y_prob, outdir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(); plt.plot(fpr,tpr,lw=2); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"roc.png")); plt.close()
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rec,prec,lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"pr.png")); plt.close()

def run_deepkegg_fold(tmpdir, k=64, r=1e-4, epochs=200, seed=42):
    cmd = f'python "{HOOK_PY}" --workdir "{tmpdir}" --k {k} --r {r} --epochs {epochs} --seed {seed}'
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError("DeepKEGG hook 运行失败")
    out = os.path.join(tmpdir, "y_prob.csv")
    if not os.path.exists(out):
        raise RuntimeError("DeepKEGG hook 未产出 y_prob.csv")
    df = pd.read_csv(out)
    if not {"sample_id","y_prob"}.issubset(set(df.columns)):
        raise RuntimeError("y_prob.csv 格式错误，需包含 sample_id,y_prob")
    return df

def main():
    cfg = load_cfg(CFG_PATH)
    paths = cfg["paths"]
    out_root = cfg.get("output_root", "/tmp/DeepKEGG-agent/runs")
    model_name = cfg["model"]
    tag = model_short(model_name)
    outdir = ensure_outdir(out_root, cfg, tag)

    print("[1/5] 读取标签…")
    ydf = load_labels(paths["clinical"], id_hint=paths.get("id_column","index"), label_col_hint=paths.get("label_column","response"))

    print("[2/5] 读取并拼接组学…")
    X = load_omics(paths, cfg["omics"])

    print("[3/5] 对齐样本…")
    common = X.index.intersection(ydf.index)
    X = X.loc[common].copy()
    y = ydf.loc[common].iloc[:,0].values.astype(int)
    print(f"样本数: {len(common)}, 特征数: {X.shape[1]}")

    folds, reps = cfg["cv_folds"], cfg["cv_repeats"]
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=reps, random_state=42)

    aucs, auprs, accs, f1s = [], [], [], []
    per_fold_rows, all_pred = [], []
    fold_id = 0

    if "DeepKEGG" not in model_name:
        print(f"[4/5] 交叉验证内标准化 + 训练（模型：{model_name}）…")
        for train_idx, test_idx in rskf.split(X.values, y):
            fold_id += 1
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(X.values[train_idx])
            X_test  = scaler.transform(X.values[test_idx])

            clf = make_model(model_name, y[train_idx])
            clf.fit(X_train, y[train_idx])
            prob = clf.predict_proba(X_test)[:,1]
            pred = (prob >= 0.5).astype(int)
            yt = y[test_idx]

            aucs.append(roc_auc_score(yt, prob))
            auprs.append(average_precision_score(yt, prob))
            accs.append(accuracy_score(yt, pred))
            f1s.append(f1_score(yt, pred))
            all_pred.append(pd.DataFrame({"sample_id": X.index[test_idx], "y_true": yt, "y_prob": prob, "y_pred": pred, "fold": fold_id}))
            per_fold_rows.append({"fold": fold_id, "AUC": aucs[-1], "AUPR": auprs[-1], "ACC": accs[-1], "F1": f1s[-1]})
    else:
        print("[4/5] 交叉验证内调用 DeepKEGG hook…")
        for train_idx, test_idx in rskf.split(X.values, y):
            fold_id += 1
            X_train_df = X.iloc[train_idx].copy()
            X_test_df  = X.iloc[test_idx].copy()
            yt = y[test_idx]

            tmpdir = tempfile.mkdtemp(prefix="dk_fold_")
            try:
                # 按组学拆分写出 train/test
                def write_by_group(df, name, outname):
                    cols = [c for c in df.columns if c.startswith(f"{name}::")]
                    if not cols: return
                    sub = df[cols].copy()
                    sub.columns = [c.split("::",1)[1] for c in cols]
                    out = sub.copy()
                    out.insert(0, "sample_id", df.index)
                    out.to_csv(os.path.join(tmpdir, outname), index=False)

                write_by_group(X_train_df, "mRNA",  "train_X_mRNA.csv")
                write_by_group(X_train_df, "miRNA", "train_X_miRNA.csv")
                write_by_group(X_train_df, "SNV",   "train_X_SNV.csv")
                write_by_group(X_test_df,  "mRNA",  "test_X_mRNA.csv")
                write_by_group(X_test_df,  "miRNA", "test_X_miRNA.csv")
                write_by_group(X_test_df,  "SNV",   "test_X_SNV.csv")
                pd.DataFrame({"sample_id": X_train_df.index, "y": y[train_idx]}).to_csv(os.path.join(tmpdir, "train_y.csv"), index=False)

                # 复制 KEGG 资源
                if not os.path.exists(paths.get("kegg_gmt","")) or not os.path.exists(paths.get("kegg_map_long","")):
                    raise ValueError("DeepKEGG 需要 KEGG_gmt 和 kegg_map_long，请在上传目录提供并在配置里填写路径。")
                shutil.copy(paths["kegg_gmt"], os.path.join(tmpdir, "kegg_gmt.gmt"))
                shutil.copy(paths["kegg_map_long"], os.path.join(tmpdir, "kegg_map.txt"))

                # 调 hook，拿 test 概率
                yprob_df = run_deepkegg_fold(tmpdir, k=cfg.get("k",64))
                yprob_df = yprob_df.set_index("sample_id").loc[X_test_df.index]  # 对齐顺序
                prob = yprob_df["y_prob"].values
                pred = (prob >= 0.5).astype(int)

                aucs.append(roc_auc_score(yt, prob))
                auprs.append(average_precision_score(yt, prob))
                accs.append(accuracy_score(yt, pred))
                f1s.append(f1_score(yt, pred))
                all_pred.append(pd.DataFrame({"sample_id": X_test_df.index, "y_true": yt, "y_prob": prob, "y_pred": pred, "fold": fold_id}))
                per_fold_rows.append({"fold": fold_id, "AUC": aucs[-1], "AUPR": auprs[-1], "ACC": accs[-1], "F1": f1s[-1]})
            finally:
                if os.environ.get('KEEP_FOLDS','0')!='1':
                    shutil.rmtree(tmpdir, ignore_errors=True)
                else:
                    print('[DEBUG] 保留本折目录:', tmpdir)

    pred_df = pd.concat(all_pred, axis=0)
    per_fold_df = pd.DataFrame(per_fold_rows)
    metrics = pd.DataFrame({
        "AUC":[np.mean(aucs)], "AUC_std":[np.std(aucs)],
        "AUPR":[np.mean(auprs)], "AUPR_std":[np.std(auprs)],
        "ACC":[np.mean(accs)], "ACC_std":[np.std(accs)],
        "F1":[np.mean(f1s)], "F1_std":[np.std(f1s)],
        "n_samples":[len(common)], "n_features":[X.shape[1]]
    })

    os.makedirs(outdir, exist_ok=True)
    metrics.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    per_fold_df.to_csv(os.path.join(outdir, "metrics_per_fold.csv"), index=False)
    pred_df.to_csv(os.path.join(outdir, "predictions.csv"), index=False)

    print("[5/5] 绘制曲线…")
    fpr, tpr, _ = roc_curve(pred_df["y_true"].values, pred_df["y_prob"].values)
    plt.figure(); plt.plot(fpr,tpr,lw=2); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"roc.png")); plt.close()
    prec, rec, _ = precision_recall_curve(pred_df["y_true"].values, pred_df["y_prob"].values)
    plt.figure(); plt.plot(rec,prec,lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"pr.png")); plt.close()

    with open(os.path.join(outdir,"run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    X.to_csv(os.path.join(outdir,"X_features_used.csv"))
    ydf.loc[X.index].to_csv(os.path.join(outdir,"y_labels_used.csv"))

    print("\n✅ 完成！输出目录：", outdir)
    print("  - metrics.csv")
    print("  - metrics_per_fold.csv")
    print("  - predictions.csv")
    print("  - roc.png / pr.png")
    print("  - X_features_used.csv / y_labels_used.csv")
    print("  - run_config.json")

if __name__ == "__main__":
    main()
