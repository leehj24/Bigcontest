# Model/build_rl_dataset.py
# - 원천 데이터(임베딩/월지표/트렌드)를 병합하여 rl_dataset.csv 생성
# - 견고한 CSV/엑셀 로더(인코딩 자동), 숫자 컬럼 정제, 랭크/트렌드 주입, RL 라벨 생성

from pathlib import Path
import pandas as pd
import numpy as np
import json

# -------------------------
# 경로
# -------------------------
ROOT = Path(__file__).resolve().parents[1]   # .../Bigcontest
MODEL_DIR = ROOT / "Model"
DATA_DIR  = ROOT / "Bigcontest_Data"
CROLL_DIR = ROOT / "Crolling"

emb_path   = MODEL_DIR / "node_embeddings.csv"
set2_path  = DATA_DIR / "big_data_set2_f.csv"   # 월별 성과(매출, 배달비중 등)
set3_path  = DATA_DIR / "big_data_set3_f.csv"   # 월별 고객 지표(재방문/신규 등)
trend_path = CROLL_DIR / "trend_results.json"
out_path   = MODEL_DIR / "rl_dataset.csv"

# -------------------------
# 유틸
# -------------------------
def _read_table(p: Path) -> pd.DataFrame:
    """엑셀/CSV 모두 안전 로드. 인코딩/구분자 자동."""
    if not p.exists():
        raise FileNotFoundError(f"필수 파일 누락: {p}")
    suffix = p.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        print(f"✅ loaded {p.name} as EXCEL")
        return pd.read_excel(p)
    # CSV
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    last_err = None
    for enc in encodings:
        # 1) C 엔진 기본 시도
        try:
            df = pd.read_csv(p, encoding=enc)
            print(f"✅ loaded {p.name} with encoding={enc} (engine=C)")
            return df
        except Exception as e:
            last_err = e
            # 2) 파이썬 엔진 + sep 자동 추정(복잡한 구분자/따옴표 케이스)
            try:
                df = pd.read_csv(p, encoding=enc, engine="python", sep=None)
                print(f"✅ loaded {p.name} with encoding={enc} (engine=python, sep=auto)")
                return df
            except Exception as e2:
                last_err = e2
                continue
    raise last_err

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """엑셀 저장 시 생긴 Unnamed 인덱스 컬럼 제거."""
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """쉼표/공백/%/원 등의 문자가 섞인 숫자 문자열을 강제로 float로 변환."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(r"[^0-9\.\-\+eE]", "", regex=True),
                errors="coerce"
            )
    return df

# -------------------------
# 로드
# -------------------------
df_emb = _read_table(emb_path)
df2    = _read_table(set2_path)
df3    = _read_table(set3_path)

df_emb = _drop_unnamed(df_emb)
df2    = _drop_unnamed(df2)
df3    = _drop_unnamed(df3)

# -------------------------
# 숫자/ID 정제
# -------------------------
latent_cols = [c for c in df_emb.columns if c.startswith("latent_")]
if len(latent_cols) < 16:
    raise ValueError(f"node_embeddings.csv에 latent_0..latent_15 컬럼이 부족합니다. 현재: {latent_cols[:8]} ...")

df_emb = _coerce_numeric(df_emb, latent_cols)

num_cols_set2 = ["RC_M1_SAA", "DLV_SAA_RAT"]
num_cols_set3 = ["MCT_UE_CLN_REU_RAT", "MCT_UE_CLN_NEW_RAT"]

df2 = _coerce_numeric(df2, num_cols_set2)
df3 = _coerce_numeric(df3, num_cols_set3)

# ENCODED_MCT 타입 통일(조인 안정)
for d in (df_emb, df2, df3):
    if "ENCODED_MCT" in d.columns:
        d["ENCODED_MCT"] = d["ENCODED_MCT"].astype(str)

# -------------------------
# 필수 컬럼 체크
# -------------------------
need2 = {"ENCODED_MCT","TA_YM","RC_M1_SAA","DLV_SAA_RAT"}
need3 = {"ENCODED_MCT","TA_YM","MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT"}
miss2 = need2 - set(df2.columns)
miss3 = need3 - set(df3.columns)
if miss2: raise ValueError(f"big_data_set2_f.csv 컬럼 누락: {sorted(miss2)}")
if miss3: raise ValueError(f"big_data_set3_f.csv 컬럼 누락: {sorted(miss3)}")

# -------------------------
# 월 데이터 병합
# -------------------------
dfm = pd.merge(
    df2[list(need2)],
    df3[list(need3)],
    on=["ENCODED_MCT","TA_YM"],
    how="inner"
)

# -------------------------
# RC_M1_SAA_RANK 생성 (업종 유무에 따라 그룹 기준 다르게)
# -------------------------
def _to_bins_series(s: pd.Series, q=6):
    try:
        return pd.qcut(s, q=q, labels=range(1, q+1), duplicates="drop").astype(int)
    except Exception:
        n=len(s)
        if n==0: return s
        edges=[(i*n)//q for i in range(q+1)]
        idx=s.rank(method="first", ascending=False).astype(int)  # 큰 값=상위
        out=[min(max(sum(v >= e for e in edges[1:]) + 1, 1), q) for v in idx]
        return pd.Series(out, index=s.index).astype(int)

if "HPSN_MCT_ZCD_NM" in df2.columns:
    base = df2[["ENCODED_MCT","TA_YM","HPSN_MCT_ZCD_NM","RC_M1_SAA"]].copy()
    base = base.sort_values(["TA_YM","HPSN_MCT_ZCD_NM","RC_M1_SAA"], ascending=[True,True,False])
    base["__rank_id"] = base.groupby(["TA_YM","HPSN_MCT_ZCD_NM"]).cumcount() + 1
    base["RC_M1_SAA_RANK"] = base.groupby(["TA_YM","HPSN_MCT_ZCD_NM"])["__rank_id"].transform(_to_bins_series)
    dfm = pd.merge(dfm, base[["ENCODED_MCT","TA_YM","RC_M1_SAA_RANK"]], on=["ENCODED_MCT","TA_YM"], how="left")
else:
    base = df2[["ENCODED_MCT","TA_YM","RC_M1_SAA"]].copy()
    base = base.sort_values(["TA_YM","RC_M1_SAA"], ascending=[True,False])
    base["__rank_id"] = base.groupby(["TA_YM"]).cumcount() + 1
    base["RC_M1_SAA_RANK"] = base.groupby(["TA_YM"])["__rank_id"].transform(_to_bins_series)
    dfm = pd.merge(dfm, base[["ENCODED_MCT","TA_YM","RC_M1_SAA_RANK"]], on=["ENCODED_MCT","TA_YM"], how="left")

# -------------------------
# TREND_RATIO 주입(없거나 매핑 실패 시 50)
# -------------------------
trend_map = {}
if trend_path.exists():
    try:
        trend_raw = json.loads(trend_path.read_text(encoding="utf-8"))
        # 예상: 업종명 -> score
        trend_map = {k: float(v) for k, v in trend_raw.items() if isinstance(v, (int, float, str))}
    except Exception:
        pass

if "HPSN_MCT_ZCD_NM" in df2.columns and trend_map:
    df_tr = df2[["ENCODED_MCT","TA_YM","HPSN_MCT_ZCD_NM"]].drop_duplicates()
    df_tr["TREND_RATIO"] = df_tr["HPSN_MCT_ZCD_NM"].map(trend_map).fillna(50.0)
    dfm = pd.merge(dfm, df_tr[["ENCODED_MCT","TA_YM","TREND_RATIO"]], on=["ENCODED_MCT","TA_YM"], how="left")
else:
    dfm["TREND_RATIO"] = 50.0

# -------------------------
# 임베딩 병합
# -------------------------
df = pd.merge(
    dfm,
    df_emb[["ENCODED_MCT"] + latent_cols],
    on="ENCODED_MCT",
    how="inner"
)

# -------------------------
# RL 라벨 생성: reward, terminal, action
# -------------------------
df = df.sort_values(["ENCODED_MCT","TA_YM"])

# 다음달 매출
df["RC_next"] = df.groupby("ENCODED_MCT")["RC_M1_SAA"].shift(-1)

# 보상: 다음달 매출 증분 비율(안정화)
denom = df["RC_M1_SAA"].replace(0, np.nan)
df["reward_raw"] = (df["RC_next"] - df["RC_M1_SAA"]) / denom
df["reward_raw"] = df["reward_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)

if df["reward_raw"].std(ddof=0) > 0:
    df["reward"] = (df["reward_raw"] - df["reward_raw"].mean()) / (df["reward_raw"].std(ddof=0) + 1e-6)
else:
    df["reward"] = df["reward_raw"]

# 에피소드 종료 플래그: 다음달 없음 → 1
df["terminal"] = df["RC_next"].isna().astype(int)

# 액션(규칙 기반 더미): 0~3
def rule_action(row):
    new_ = row.get("MCT_UE_CLN_NEW_RAT", 0.0)
    reu_ = row.get("MCT_UE_CLN_REU_RAT", 0.0)
    dlv_ = row.get("DLV_SAA_RAT", 0.0)
    cnt = (new_ > 20) + (reu_ > 35) + (dlv_ > 30)  # 임계는 데이터 분포에 맞게 조정 가능
    if cnt >= 2: return 3
    if new_ > 20: return 1
    if (reu_ > 35) or (dlv_ > 30): return 2
    return 0

df["action"] = df.apply(rule_action, axis=1).astype(int)

# -------------------------
# 저장
# -------------------------
need_feats = ["DLV_SAA_RAT","MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT","RC_M1_SAA_RANK","TREND_RATIO"]
final_cols = ["ENCODED_MCT","TA_YM"] + latent_cols + need_feats + ["action","reward","terminal"]

df_final = df[final_cols].dropna(subset=latent_cols)

out_path.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ rl_dataset.csv 생성 완료: {out_path} (rows={len(df_final)})")
