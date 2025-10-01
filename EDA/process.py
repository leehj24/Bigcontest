import pandas as pd
import json
from pathlib import Path

# --- 경로 설정 ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "BIgcontest_Data"
CROLLING_DIR = ROOT_DIR / "Crolling"
OUTPUT_DIR = ROOT_DIR / "EDA"

def load_and_preprocess_data():
    """
    4개의 데이터셋을 불러와 전처리하고 하나로 병합하는 함수
    """
    print("--- 데이터 로딩 시작 ---")
    try:
        # ✅ [수정] encoding='cp949' 추가
        df1 = pd.read_csv(DATA_DIR / "big_data_set1_f.csv", encoding='cp949')
        print("✅ df1 (가맹점 개요) 로딩 성공")

        # ✅ [수정] encoding='cp949' 추가
        df2 = pd.read_csv(DATA_DIR / "big_data_set2_f.csv", encoding='cp949')
        print("✅ df2 (월별 이용 정보) 로딩 성공")

        # ✅ [수정] encoding='cp949' 추가
        df3 = pd.read_csv(DATA_DIR / "big_data_set3_f.csv", encoding='cp949')
        print("✅ df3 (월별 이용 고객) 로딩 성공")

        # 4. 업종별 트렌드 정보 (df_trend) - JSON은 보통 utf-8이므로 수정 필요 없음
        with open(CROLLING_DIR / "trend_results.json", 'r', encoding='utf-8') as f:
            trend_data = json.load(f)

        trend_list = []
        for category, results in trend_data.items():
            if not results: continue
            for result in results:
                for item in result['data']:
                    trend_list.append({
                        'HPSN_MCT_ZCD_NM': category,
                        'TA_YM': int(item['period'].replace('-', '')),
                        'TREND_RATIO': item['ratio']
                    })
        df_trend = pd.DataFrame(trend_list)
        print("✅ df_trend (업종 트렌드) 로딩 및 변환 성공")

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. \n{e}")
        return None
    except Exception as e:
        print(f"데이터 로딩 중 다른 오류 발생: {e}")
        return None

    print("\n--- 데이터 병합 시작 ---")
    df_monthly = pd.merge(df2, df3, on=['ENCODED_MCT', 'TA_YM'], how='left')
    print("  - df2, df3 병합 완료...")

    df_merged = pd.merge(df_monthly, df1, on='ENCODED_MCT', how='left')
    print("  - 월별 데이터 + 가맹점 정보 병합 완료...")

    df_final = pd.merge(df_merged, df_trend, on=['HPSN_MCT_ZCD_NM', 'TA_YM'], how='left')
    print("  - 트렌드 데이터 병합 완료...")
    
    df_final['TREND_RATIO'] = df_final['TREND_RATIO'].fillna(0)
    
    print("\n✅ 모든 데이터 병합 성공!")
    return df_final


def main():
    """
    메인 실행 함수
    """
    merged_df = load_and_preprocess_data()

    if merged_df is not None:
        print("\n--- 최종 병합 데이터 샘플 (상위 5개) ---")
        print(merged_df.head())
        print("\n--- 최종 데이터 정보 ---")
        merged_df.info()

        output_path = OUTPUT_DIR / "merged_data.csv"
        # ✅ [수정] 엑셀에서 한글이 깨지지 않도록 encoding='utf-8-sig' 사용
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 병합된 데이터가 '{str(output_path)}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()