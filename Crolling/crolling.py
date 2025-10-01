# Crolling/crolling.py

import requests
import json
import os
import pandas as pd
import time
from pathlib import Path  # ✅ pathlib 라이브러리에서 Path를 가져옵니다.

# .env 로드 (있으면)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- 경로 설정 ---
# ✅ 이 crolling.py 파일의 절대 경로를 기준으로 프로젝트 루트 폴더(Bigcontest)를 찾습니다.
ROOT_DIR = Path(__file__).resolve().parents[1]

# ✅ 데이터 파일과 결과 파일의 절대 경로를 만듭니다.
DATA_DIR = ROOT_DIR / "BIgcontest_Data"
PATH_DATA_SET1 = DATA_DIR / "big_data_set1_f.csv"
PATH_OUTPUT_JSON = ROOT_DIR /"Crolling"/ "trend_results.json"  # 결과 파일은 프로젝트 최상단에 저장
# --- 경로 설정 끝 ---


def get_naver_trend_data(api_key, client_secret, keywords):
    """네이버 데이터랩 API 호출 함수 (수정 없음)"""
    url = "https://openapi.naver.com/v1/datalab/search"
    headers = {
        "X-Naver-Client-Id": api_key,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json"
    }
    body = {
        "startDate": "2023-01-01",
        "endDate": "2024-12-31",
        "timeUnit": "month",
        "keywordGroups": [
            {
                "groupName": keywords[0],
                "keywords": keywords
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

def main():
    """메인 실행 함수"""
    # 1. 데이터 파일 읽기 (✅ encoding='cp949' 추가하여 오류 해결)
    try:
        df_stores = pd.read_csv(str(PATH_DATA_SET1), encoding='cp949')
    except FileNotFoundError:
        print(f"오류: {str(PATH_DATA_SET1)} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return
    except UnicodeDecodeError:
        print(f"오류: {str(PATH_DATA_SET1)} 파일의 인코딩이 'cp949'가 아닐 수 있습니다. 'euc-kr' 등으로 시도해보세요.")
        return


    # 2. 수집할 키워드(업종) 목록 생성
    unique_categories = df_stores['HPSN_MCT_ZCD_NM'].unique()

    # 3. 기존에 수집된 데이터가 있으면 불러오기
    all_trend_data = {}
    if os.path.exists(str(PATH_OUTPUT_JSON)):
        print(f"'{str(PATH_OUTPUT_JSON)}' 파일을 발견했습니다. 중간부터 작업을 재개합니다.")
        with open(str(PATH_OUTPUT_JSON), 'r', encoding='utf-8') as f:
            all_trend_data = json.load(f)
    else:
        print("새로운 데이터 수집을 시작합니다.")

    # API 키 로드
    NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
    NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("오류: 네이버 API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    # 수집해야 할 업종과 이미 수집된 업종 분리
    collected_categories = list(all_trend_data.keys())
    total_count = len(unique_categories)
    collected_count = len(collected_categories)
    
    print(f"총 {total_count}개 업종 중 {collected_count}개 수집 완료. 남은 업종: {total_count - collected_count}개")
    print("-" * 30)

    # 반복문으로 남은 업종에 대해서만 API 호출
    for i, category in enumerate(unique_categories):
        if not isinstance(category, str) or category in collected_categories:
            continue

        print(f"[{i+1}/{total_count}] '{category}' 업종 트렌드 데이터 수집 중...")
        trend_data = get_naver_trend_data(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, [category])

        if 'error' not in trend_data:
            all_trend_data[category] = trend_data.get('results', [])
            
            # 데이터를 수집할 때마다 파일에 즉시 저장
            with open(str(PATH_OUTPUT_JSON), 'w', encoding='utf-8') as f:
                json.dump(all_trend_data, f, ensure_ascii=False, indent=4)
        else:
            print(f"'{category}' 데이터 수집 중 오류 발생: {trend_data['error']}")
        
        # 네이버 서버에 부담을 주지 않도록 잠시 대기
        time.sleep(0.1)

    print("-" * 30)
    print(f"모든 업종의 트렌드 데이터 수집 완료! '{str(PATH_OUTPUT_JSON)}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()