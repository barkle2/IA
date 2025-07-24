# 지방관서 신설 및 정원 배정(안) 작성 프로그램

#%%
import pandas as pd
pd.set_option('display.max_colwidth', None)

from collections import Counter

#%%
# 청별로 "[연번] 기관명" 출력하는 함수 정의
def print_branch_names(df, cheong=None):
    
    if cheong is not None:
        print(f"청을 입력했습니다: {cheong}")
        df_filtered = df[df['청'] == cheong]

        records = []
        for index, row in df_filtered.iterrows():
            records.append(f"{row['연번']}: {row['기관명']}")

        if records:
            print(f"{cheong}:")
            # records 리스트를 5개씩 나누어 출력
            for i in range(0, len(records), 5):
                # i부터 i+5까지의 슬라이스를 출력
                print(f"  {', '.join(records[i:i+5])}")
        else:
            print(f"'{cheong}'에 해당하는 기관명이 없습니다.")

    else:
        print("청을 입력하지 않았습니다. 모든 청의 기관명을 출력합니다.")
        for cheong in df['청'].unique():
            print(f"\n{cheong}:")
            df_cheong = df[df['청'] == cheong]
            records = []
            for index, row in df_cheong.iterrows():
                records.append(f"[{row['연번']}] {row['기관명']}")
            
            for i in range(0, len(records), 5):
                print(", ".join(records[i:i+5]))

#%%
# 관할 문자열을 입력받아 해당 지역 해당값 총합을 반환하는 함수
def get_total_count(sggs_str, sgg_dict):
    sggs = [sgg.strip() for sgg in sggs_str.split(',')]
    return sum(sgg_dict.get(sgg, 0) for sgg in sggs)

# 시군구 관할 데이터로 입력된 컬럼 값 채우기
def get_count_by_sgg(df_br, df_sgg, columns_to_calculate):

    # 입력된 컬럼 이름이 단일 문자열이고 콤마로 구분되어 있을 경우 분리
    if len(columns_to_calculate) == 1 and ',' in columns_to_calculate[0]:
        columns_to_calculate = [col.strip() for col in columns_to_calculate[0].split(',')]

    for col_name in columns_to_calculate:
        # 해당 컬럼이 df_sgg에 없으면 에러 방지를 위해 건너뛰고 경고 출력
        if col_name not in df_sgg.columns:
            print(f"경고: df_sgg에 '{col_name}' 컬럼이 없습니다. 이 컬럼은 건너뜝니다.")
            continue

        # 지역명 -> 해당 컬럼 데이터 딕셔너리로 변환
        sgg_data_dict = df_sgg.set_index('지역명')[col_name].to_dict()

        # 각 관할에 대해 총합 계산
        counts = []
        for sgg_str in df_br['관할']:
            counts.append(get_total_count(sgg_str, sgg_data_dict))

        # df_br에 새 컬럼 추가
        df_br[col_name] = counts

    return df_br

#%%
# 1. 지방관서 관할, 정원 파일을 읽는다.
# 1-1. 파일명 설정 후 read_excel로 읽어오기
# 입력파일명 설정
input_file = '1_1_지청_기초_자료(v3).xlsx'
# 연번은 문자열로 읽음(나중에 지청신설 시 00-1, 00-2 등으로 구분하기 위함)
df_br = pd.read_excel(input_file, dtype={'연번': str})

#%%
# 1-2. df_br 정보 확인
df_br.head(2)

#%%
# 1-3. df_br 데이터 검증하기
# 1-3-1. 연번이 순서대로 입력되어 있는지 확인
df_br['연번'].is_monotonic_increasing # True면 연번이 순서대로 입력되어 있음

#%%
# 1-3-2. 청 값에 이상한 값이 없는지 확인
print("청 고유개수:", len(df_br['청'].unique()))
print(df_br['청'].unique())

#%%
# 1-3-3. 지청 값에 이상한 값이 없는지 확인
print("지청 고유개수:", len(df_br['지청'].unique()))
print(df_br['지청'].unique())

#%%
# 1-3-4. 기관명 컬럼 생성
df_br['기관명'] = df_br['지청'] + df_br['규모']

#%%
# 컬럼 순서 변경
df_br = df_br[['연번', '청', '지청', '규모', '기관명', '관할', '총정원', '정원', '정원_산안', '정원_노동', '재해자수_24', '중대재해자수_24', '근로손실일수_24', '신고사건_24']]

#%%
# 1-3-5. 기관명을 청 별로 출력
print_branch_names(df_br)

#%%
# 1-3-6. 규모 분석: 규모 컬럼에 청, 지청, 출장소가 각각 몇 개씩 있는지 확인
df_br['규모'].value_counts()

#%%
# 1-3-7. 관할에 중복값이 있는지 확인
# 관할을 읽어서 쉼표로 값을 구분해서 리스트로 만들기
sgg_list = []
for entry in df_br['관할']:
    # 각 엔트리를 쉼표로 분리하고, 각 분리된 문자열의 양쪽 공백을 제거하여 리스트에 추가
    parts = [part.strip() for part in entry.split(',')]
    sgg_list.extend(parts) # extend를 사용하여 리스트에 개별 요소로 추가제거

print(len(sgg_list), "개의 관할이 있습니다.")

#%%
# 관할_list에 카운트가 많은 순서대로 5개 출력
counts = Counter(sgg_list)
for sgg, count in counts.most_common(5):
    print(f"{sgg}: {count}개")

#%%
# 1-3-8. 관할에 빠진 값이 있는지 확인
# 시군구 사업체수, 종사자수 엑셀 파일의 모든 컬럼을 문자열로 읽어오기
df_sgg = pd.read_excel('1_2_시군구_사업체수_종사자수_인구_면적.xlsx', dtype={'연번': str})

sgg_total_list = df_sgg['지역명'].to_list()

#%%
# sgg_total_list에는 있고 sgg_list에는 없는 값 찾기
missing_sgg = set(sgg_total_list) - set(sgg_list)
print("사업체노동실태조사에는 있고, 관할에는 빠진 시군구:")
if len(missing_sgg) == 0:
    print("없음")
else:
    for sgg in missing_sgg:
        print(sgg)

missing_sgg = set(sgg_list) - set(sgg_total_list)
print("관할에는 있고, 사업체노동실태조사에는 빠진 시군구:")
if len(missing_sgg) == 0:
    print("없음")
else:
    for sgg in missing_sgg:
        print(sgg)

#%%
# 1-3-9. 경북 군위군 데이터 수정
print(" 경북 군위군은 2023년부터 대구 군위군으로 변경되었으므로", "\n", "사업체노동실태조사(v3)_시군구.xlsx 파일에서 경북 군위군은 삭제하고, ", "\n", "2022년 이전 데이터는 대구 군위군에 입력함")

#%%
# 1-3-10. 정원, 정원_산안, 정원_노동 컬럼의 값 확인
# 각각의 합 구하기
print("총정원 합계:", df_br['총정원'].sum())
print("정원 합계:", df_br['정원'].sum())
print("산안 합계:", df_br['정원_산안'].sum())
print("노동 합계:", df_br['정원_노동'].sum())

# 정원 = 산안 + 노동 를 충족하지 않는 행 찾기
non_matching_rows = df_br[df_br['정원'] != (df_br['정원_산안'] + df_br['정원_노동'])]
if non_matching_rows.empty:
    print("모든 행이 정원 = 산안 + 노동 을 충족합니다.")
else:
    print("정원 = 산안 + 노동 를 충족하지 않는 행이 있습니다:")
    print(non_matching_rows[['연번', '청', '지청', '정원', '정원_산안', '정원_노동']])

#%%
# 2. 사업체수, 종사자수 값을 추가

# 2-1. df_br에 있는 '관할' 컬럼의 정보로 사업체수, 종사자수 구하기
# get_count_by_sgg 함수로 df_br에 사업체수, 종사자수 컬럼 추가
columns_to_calculate = ['사업체수_23, 종사자수_23']
df_br = get_count_by_sgg(df_br, df_sgg, columns_to_calculate)

#%%
# 추가된 컬럼 확인 
# 23년 사업체수: 2124670, 23년 종사자수: 19159335
print("23년 사업체수 합:", df_br['사업체수_23'].sum())
print("23년 종사자수 합:", df_br['종사자수_23'].sum())

      
#%%
df_br.head()

#%%
# 컬럼 순서 변경
df_br = df_br[['연번', '청', '지청', '규모', '기관명', '관할', '총정원', '정원', '정원_산안', '정원_노동', '사업체수_23', '종사자수_23', '재해자수_24', '중대재해자수_24', '근로손실일수_24', '신고사건_24']]

#%%
# "지청 기초 자료.xlsx" 파일로 출력한다.
df_br.to_excel("1_3_지청_기초_자료(by_program).xlsx", index=False)

# %%
