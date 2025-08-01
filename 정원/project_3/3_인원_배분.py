#%%
# 라이브러리 호출
import pandas as pd
import os

# 현재 작업 디렉토리 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%%
# 가중치에 따라 업무점수 계산하기
def calculate_weighted_score(row, weights):
    score = 0
    for key, weight in weights.items():
        score += row[key] * weight
    return score

# %%
# 1인당 업무점수 계산해서 '1인당_업무점수' 컬럼에 저장하는 함수
def calculate_per_person_score(row):
    if row['정원'] > 0:
        return row['업무점수'] / (row['정원']+row['추가_정원1']+row['추가_정원2']+row['추가_정원3'])
    else:
        return 0
    
#%%
# 지정한 컬럼의 값을 dict 비율에 따라 정수로 나누는 함수
def split_TO(df: pd.DataFrame, column: str, ratio_dict: dict) -> pd.DataFrame:
    
    df_cp = df.copy()

    # 비율 키 추출 (예: '산안', '노동')
    keys = list(ratio_dict.keys())
    if len(keys) != 2:
        raise ValueError("비율 딕셔너리는 정확히 두 개의 키를 가져야 합니다.")

    key1, key2 = keys[0], keys[1]

    # 새로운 컬럼명 설정
    col1 = key1
    col2 = key2

    # 컬럼 초기화
    df_cp[col1] = 0
    df_cp[col2] = 0

    # 각 행에 대해 정수 분배 수행
    for idx, row in df_cp.iterrows():
        total = row[column]
        val1 = int(total * ratio_dict[key1])
        val2 = int(total * ratio_dict[key2])
        diff = total - (val1 + val2)

        # 부족한 몫을 더 큰 비율 쪽에 보정
        if diff > 0:
            if ratio_dict[key1] >= ratio_dict[key2]:
                val1 += diff
            else:
                val2 += diff

        # 결과 저장
        df_cp.at[idx, col1] = val1
        df_cp.at[idx, col2] = val2

    return df_cp

#%%
# 데이터 불러오기
df_br = pd.read_excel('2_4_지청_신설_표준화.xlsx')

#%%
# 인원 산정에 필요한 정보만 선택
df_br = df_br[['연번', '청', '지청', '규모', '기관명', '관할', '총정원', '정원', '정원_산안', '정원_노동', '사업체수_23', '사업체수_23_표준', '종사자수_23', '종사자수_23_표준', '재해자수_24_표준', '중대재해자수_24_표준', '근로손실일수_24_표준', '신고사건_24_표준']]

#%%
# 인원 배분 기초 자료 엑셀 파일로 저장
df_br.to_excel('3_1_인원_배분_기초자료.xlsx', index=False)

# %%
# 가중치 딕셔너리를 변수로 정의
weights = {
    '사업체수_23_표준': 0.35,
    '종사자수_23_표준': 0.2,
    '재해자수_24_표준': 0.2,
    '중대재해자수_24_표준': 0.1,
    '근로손실일수_24_표준': 0.1,
    '신고사건_24_표준': 0.05
}

#%%
# 가중치에 따라 각 열을 곱한 후 합산하여 업무점수 계산
df_br['업무점수'] = sum(df_br[col] * weight for col, weight in weights.items()) * 100 

# %%
df_br[['기관명','업무점수']].sort_values(by='업무점수', ascending=False)

# %%
# df_br 에 ['추가_정원'] 컬럼을 생성하고 초기값은 0으로 설정
df_br['추가_정원1'] = 0
df_br['추가_정원2'] = 0
df_br['추가_정원3'] = 0

#%%
# 사업체수_23 값에 따라 추가정원1에 3~6명 추가
# 사업체수가 30,000개 이하이면 3명
df_br.loc[df_br['사업체수_23'] <= 30000, '추가_정원1'] += 3

# 사업체수가 30,000개 초과 60,000개 이하이면 4명
df_br.loc[(df_br['사업체수_23'] > 30000) & (df_br['사업체수_23'] <= 60000), '추가_정원1'] += 4

# 사업체수가 60,000개 초과이면 6명
df_br.loc[df_br['사업체수_23'] > 60000, '추가_정원1'] += 6

#%%
print(df_br[['기관명', '사업체수_23', '추가_정원1']])

#%%
# 종사자수_23 값에 따라 추가정원2에 2~4명 추가
# 종사자수가 300,000명 이하이면 2명
df_br.loc[df_br['종사자수_23'] <= 300000, '추가_정원2'] += 2

# 종사자수가 300,000명 초과 600,000명 이하이면 3명
df_br.loc[(df_br['종사자수_23'] > 300000) & (df_br['종사자수_23'] <= 600000), '추가_정원2'] += 3

# 종사자수가 600,000명 초과이면 4명
df_br.loc[df_br['종사자수_23'] > 600000, '추가_정원2'] += 4

#%%
print(df_br[['기관명', '종사자수_23', '추가_정원2']])

#%%
n1 = df_br['추가_정원1'].sum() + df_br['추가_정원2'].sum()
print(n1)

#%%
# '1인당_업무점수' 컬럼 생성
df_br['1인당_업무점수'] = df_br.apply(calculate_per_person_score, axis=1)    

# %%
# n명에 대해서 다음 작업을 반복하고 싶어
# 1인당_업무점수가 가장 높은 지청에 추가_정원을 +1
# 다시 1인당 업무점수 계산
# # 이 작업을 n회 반복
# 추가 인원을 부여할 횟수
n = 2000 - n1

for _ in range(n):
    # 1인당 업무점수가 가장 높은 지청 찾기
    max_idx = df_br['1인당_업무점수'].idxmax()

    # 해당 지청의 추가 정원 1 증가
    df_br.loc[max_idx, '추가_정원3'] += 1

    # 전체 '1인당_업무점수' 다시 계산
    df_br['1인당_업무점수'] = df_br.apply(calculate_per_person_score, axis=1)

#%% 
df_br['추가_정원'] = df_br['추가_정원1'] + df_br['추가_정원2'] + df_br['추가_정원3']
# 결과 출력
print(df_br[['기관명','정원', '추가_정원', '업무점수','1인당_업무점수']].sort_values(by='1인당_업무점수', ascending=False))

# %%
ratio = {'추가_정원_산안': 0.65, '추가_정원_노동': 0.35}
df_br = split_TO(df_br, '추가_정원', ratio)

# %%
# 추가 정원 확인
df_br[['기관명', '정원', '정원_산안', '정원_노동', '추가_정원', '추가_정원_산안', '추가_정원_노동']]

#%%
new_cols = ['연번', '청', '지청', '규모', '기관명', '관할', '총정원', '정원', '정원_산안', '정원_노동',
       '사업체수_23', '사업체수_23_표준', '종사자수_23', '종사자수_23_표준', '재해자수_24_표준',
       '중대재해자수_24_표준', '근로손실일수_24_표준', '신고사건_24_표준', '업무점수', '1인당_업무점수', '추가_정원1', '추가_정원2', '추가_정원3', '추가_정원', '추가_정원_산안', '추가_정원_노동']
df_br = df_br[new_cols]

# %%
df_br.to_excel('3_2_인원_배분_결과.xlsx')

# %%
print('산안 추가 정원:', df_br['추가_정원_산안'].sum())
print('노동 추가 정원:', df_br['추가_정원_노동'].sum())

# %%
