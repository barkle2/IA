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
# 관할 문자열 활용 함수 정의
# 관할 문자열을 입력받아 해당 지역 해당값 총합을 반환하는 함수
def get_total_biz_count(br_series, sgg_dict):
    sggs = [sgg.strip() for sgg in br_series.split(',')]
    return sum(sgg_dict.get(sgg, 0) for sgg in sggs)

# df_sgg의 시군구별 사업체수, 종사자수 값을 활용하여 df_br에 지청별 사업체수, 종사자수 입력
def get_count_by_sgg(df_br, df_sgg):    
    # 지역명 -> 사업체수 딕셔너리로 변환
    sgg_biz23_dict = df_sgg.set_index('지역명')['사업체수_23'].to_dict()
    sgg_biz22_dict = df_sgg.set_index('지역명')['사업체수_22'].to_dict()
    sgg_per23_dict = df_sgg.set_index('지역명')['종사자수_23'].to_dict()
    sgg_per22_dict = df_sgg.set_index('지역명')['종사자수_22'].to_dict()
    sgg_pop24_dict = df_sgg.set_index('지역명')['인구수_24'].to_dict()
    sgg_pop23_dict = df_sgg.set_index('지역명')['인구수_23'].to_dict()
    sgg_pop22_dict = df_sgg.set_index('지역명')['인구수_22'].to_dict()
    sgg_size23_dict = df_sgg.set_index('지역명')['면적_23'].to_dict()

    count_biz23 = []
    count_biz22 = []
    count_per23 = []
    count_per22 = []
    count_pop24 = []
    count_pop23 = []
    count_pop22 = []
    count_size23 = []

    # 관할 컬럼의 각 시군구에 대해 사업체수, 종사자수 총합 구하기
    for sgg in df_br['관할']:
        count_biz23.append(get_total_biz_count(sgg, sgg_biz23_dict))
        count_biz22.append(get_total_biz_count(sgg, sgg_biz22_dict))
        count_per23.append(get_total_biz_count(sgg, sgg_per23_dict))
        count_per22.append(get_total_biz_count(sgg, sgg_per22_dict))
        count_pop24.append(get_total_biz_count(sgg, sgg_pop24_dict))
        count_pop23.append(get_total_biz_count(sgg, sgg_pop23_dict))
        count_pop22.append(get_total_biz_count(sgg, sgg_pop22_dict))
        count_size23.append(get_total_biz_count(sgg, sgg_size23_dict))

    # df_br에 사업체수, 종사자수 컬럼 추가
    df_br['사업체수_23'] = count_biz23
    df_br['사업체수_22'] = count_biz22
    df_br['종사자수_23'] = count_per23
    df_br['종사자수_22'] = count_per22
    df_br['인구수_24'] = count_pop24
    df_br['인구수_23'] = count_pop23
    df_br['인구수_22'] = count_pop22
    df_br['면적_23'] = count_size23

    return df_br

#%%
# 1. 지방관서 관할, 정원 파일을 읽는다.
# 1-1. 파일명 설정 후 read_excel로 읽어오기
# 입력파일명 설정
input_file = '250721_지방관서_기초_자료.xlsx'
# 연번은 문자열로 읽음(나중에 지청신설 시 00-1, 00-2 등으로 구분하기 위함)
df_br = pd.read_excel(input_file, dtype={'연번': str})

#%%
# 1-2. df_br 정보 확인
df_br.head()

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
df_sgg = pd.read_excel('시군구_사업체수_종사자수_인구_면적(v4).xlsx', dtype={'연번': str})

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
df_br = get_count_by_sgg(df_br, df_sgg)

#%%
# 추가된 컬럼 확인 
# 23년 사업체수: 2124670, 22년 사업체수: 2099955, 23년 종사자수: 19159335, 22년 종사자수: 18835715
print("23년 사업체수 합:", df_br['사업체수_23'].sum())
print("22년 사업체수 합:", df_br['사업체수_22'].sum())
print("23년 종사자수 합:", df_br['종사자수_23'].sum())
print("22년 종사자수 합:", df_br['종사자수_22'].sum())
print("24년 인구수 합:", df_br['인구수_24'].sum())
print("23년 인구수 합:", df_br['인구수_22'].sum())
print("22년 인구수 합:", df_br['인구수_22'].sum())
print("22년 면적 합:", df_br['면적_23'].sum())
      
#%%
df_br.head()

#%%
# "지청 기초 자료.xlsx" 파일로 출력한다.
df_br.to_excel("1.지청_기초_자료.xlsx", index=False)

# %%
