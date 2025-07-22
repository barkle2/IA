#%%
import pandas as pd
pd.set_option('display.max_colwidth', None)

from sklearn.preprocessing import MinMaxScaler

# %%
# 주어진 시군구 연번으로 관할에서 입력된 시군구를 반환하는 함수
def get_sgg(df_br: pd.DataFrame, br_index: str):
    # 입력받은 '시군구 연번'에 해당하는 행 찾기
    df_select = df_br[df_br['연번'] == br_index]

    # 선택한 행이 비어있지 않으면
    if not df_select.empty:
        # '관할' 컬럼에서 시군구 문자열 추출
        sggs = df_select['관할'].iloc[0]
        # 시군구 문자열을 쉼표로 분리하여 리스트로 변환
        sgg_list = [sgg.strip() for sgg in sggs.split(',')]
        return sgg_list
    # 선택한 행이 비어있으면
    else:
        return []

#%%
# 주어진 시군구 연번으로 관할에서 입력된 시군구를 추가하는 함수
def add_sgg(df_br: pd.DataFrame, br_index: str, sggs_to_add: str) -> pd.DataFrame:

    # 원본 DataFrame을 직접 수정하지 않기 위해 복사본을 만듭니다.
    df_br_cp = df_br.copy()

    # 입력받은 '시군구 연번'에 해당하는 행 찾기
    df_select = df_br_cp[df_br_cp['연번'] == br_index]

    # 선택한 행이 비어있지 않으면
    if not df_select.empty:
        
        # 입력할 시군구 문자열을 쉼표로 분리하여 리스트로 변환
        sgg_list = [sgg.strip() for sgg in sggs_to_add.split(',')]

        # 선택된 행에서 '관할' 컬럼의 현재 값을 가져옵니다.
        current_sggs = df_select['관할'].iloc[0]

        # 현재 '관할'이 없으면 current_sggs를 빈 리스트로 초기화
        if pd.isna(current_sggs) or current_sggs.strip() == '':
            current_sggs_list = []
        # 현재 '관할'이 있으면 쉼표로 분리하여 리스트로 변환
        else:
            current_sggs_list = [sgg.strip() for sgg in current_sggs.split(',')]

        for sgg in sgg_list:
            # 현재 '관할'에 추가할 시군구가 이미 존재하지 않으면 추가
            if sgg not in current_sggs_list:
                current_sggs_list.append(sgg)
                print(f"'{br_index}' 연번의 '관할'에 '{sgg}'를 추가했습니다.")
            # 현재 '관할'에 추가할 시군구가 이미 존재하면 추가하지 않음
            else:
                print(f"'{br_index}' 연번의 '관할'에 '{sgg}'는 이미 존재합니다. 추가하지 않습니다.")

        # 리스트를 다시 쉼표로 연결하여 문자열로 변환
        new_sgg_string = ', '.join(current_sggs_list)

        # DataFrame의 해당 행의 '관할' 컬럼을 업데이트
        df_br_cp.loc[df_br_cp['연번'] == br_index, '관할'] = new_sgg_string

        print(f"'{br_index}' 연번의 '관할'이 업데이트되었습니다: {new_sgg_string}")

        # 업데이트된 DataFrame 반환
        return df_br_cp

    # 선택한 행이 비어있으면 원본 df_br을 반환
    else:        
        print(f"경고: '연번' '{br_index}'에 해당하는 행을 찾을 수 없습니다. 원본 DataFrame을 반환합니다.")
        return df_br
    
#%%
def del_sgg(df_br: pd.DataFrame, index: str, sgg_to_delete: str) -> pd.DataFrame:

    df_br_updated = df_br.copy()

    # '연번'이 유니크하므로, 해당 연번에 대한 행이 있는지 확인하고 인덱스 추출
    row_idx = df_br_updated[df_br_updated['연번'] == index].index
    print(row_idx)

    # 셀 값을 안전하게 추출하여 문자열로 변환 (이전 수정사항 유지)
    cell_value = df_br_updated.loc[row_idx, '관할']
    if isinstance(cell_value, pd.Series):
        if cell_value.empty or pd.isna(cell_value.iloc[0]):
            current_sgg_string = ''
        else:
            current_sgg_string = str(cell_value.iloc[0])
    else:
        current_sgg_string = str(cell_value)

    # --- 수정된 부분: sgg_to_delete를 리스트로 분리 ---
    # sgg_to_delete를 쉼표로 분리하여 삭제할 시군구 리스트 생성
    sgg_to_delete_splitted = [sgg.strip() for sgg in sgg_to_delete.split(',') if sgg.strip()]
    
    # 현재 '관할' 문자열을 리스트로 변환
    if current_sgg_string.strip() == '':
        current_sggs = []
    else:
        current_sggs = [sgg.strip() for sgg in current_sgg_string.split(',')]
    
    # 변경이 실제로 발생했는지 추적
    is_changed = False
    removed_sggs = []

    # sgg_to_delete_splitted 리스트의 각 시군구를 제거 시도
    for sgg_item in sgg_to_delete_splitted:
        if sgg_item in current_sggs:
            current_sggs.remove(sgg_item)
            is_changed = True
            removed_sggs.append(sgg_item)
        else:
            print(f"'{index}' 연번의 '관할'에 '{sgg_item}'은(는) 존재하지 않습니다.")
    
    if is_changed:
        current_sggs.sort() # 일관된 순서를 위해 정렬
        new_sgg_string = ', '.join(current_sggs)
        df_br_updated.loc[row_idx, '관할'] = new_sgg_string
        print(f"'{index}' 연번의 '관할'에서 다음을 제거했습니다: {', '.join(removed_sggs)}")
    else:
        print(f"'{index}' 연번의 '관할'에 요청된 시군구 중 제거된 것이 없습니다.")
    
    return df_br_updated

#%%
# 지청을 신설하는 함수
def add_br(df_br: pd.DataFrame, index: str, cheong: str, branch: str, size: str, sgg: str) -> pd.DataFrame:
    # 입력된 데이터로 새로운 지청 행을 생성
    new_row = pd.DataFrame([{
        '연번': index,
        '청': cheong,
        '지청': branch,
        '규모': size,
        '관할': sgg,
        '기관명': branch+size
    }])

    # 시군구 연번이 중복이면 지청을 신설하지 않고 종료
    if index in df_br['연번'].values:
        print(f"경고: '연번' '{index}'는 이미 존재합니다. 데이터가 추가되지 않았습니다.")
        return df_br
    
    # 기존 데이터에 새로운 지청 행을 추가
    df_br_updated = pd.concat([df_br, new_row], ignore_index=True)
    print(f"새로운 지청 '{branch}' (연번: {index})이(가) 추가되었습니다.")
    
    return df_br_updated    

#%%
# 관할 문자열 활용 함수 정의
# 관할 문자열을 입력받아 해당 지역 해당값 총합을 반환하는 함수
def get_total_biz_count(br_series, sgg_dict):
    sggs = [sgg.strip() for sgg in br_series.split(',')]
    return sum(sgg_dict.get(sgg, 0) for sgg in sggs)

#%%
# df_sgg의 시군구별 사업체수, 종사자수 값을 활용하여 df_br에 지청별 사업체수, 종사자수 입력
def get_count_by_sgg(df_br: pd.DataFrame, df_sgg: pd.DataFrame) -> pd.DataFrame:
    # 원본 DataFrame을 직접 수정하지 않기 위해 복사본을 만듭니다.
    df_br_updated = df_br.copy()

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
    df_br_updated['사업체수_23'] = count_biz23
    df_br_updated['사업체수_22'] = count_biz22
    df_br_updated['종사자수_23'] = count_per23
    df_br_updated['종사자수_22'] = count_per22
    df_br_updated['인구수_24'] = count_pop24
    df_br_updated['인구수_23'] = count_pop23
    df_br_updated['인구수_22'] = count_pop22
    df_br_updated['면적_23'] = count_size23

    return df_br_updated

#%%
# 주어진 컬럼의 값을 0~1 MinMaxScaler 로 표준화 하는 함수
def standardize_columns(df: pd.DataFrame, columns_to_standardize: list) -> pd.DataFrame:

    # 원본 DataFrame을 직접 수정하지 않기 위해 복사본을 만듭니다.
    df_standardized = df.copy()

    # StandardScaler 인스턴스 생성
    scaler = MinMaxScaler()

    # 표준화할 컬럼들이 DataFrame에 있는지 확인
    missing_cols = [col for col in columns_to_standardize if col not in df_standardized.columns]
    if missing_cols:
        print(f"경고: 다음 컬럼들이 DataFrame에 없어 표준화에서 제외됩니다: {', '.join(missing_cols)}")
        # 존재하지 않는 컬럼은 리스트에서 제거하여 다음 과정에 영향을 주지 않도록 함
        columns_to_standardize = [col for col in columns_to_standardize if col in df_standardized.columns]
        if not columns_to_standardize: # 모든 컬럼이 없으면 그대로 반환
            print("표준화할 유효한 컬럼이 없습니다. 원본 DataFrame을 반환합니다.")
            return df_standardized

    try:
        # 지정된 컬럼들만 선택하여 표준화 적용
        # .loc를 사용하여 복사본 DataFrame에 안전하게 할당
        df_standardized.loc[:, columns_to_standardize] = scaler.fit_transform(
            df_standardized[columns_to_standardize]
        )
        print(f"다음 컬럼들이 0-1 범위로 표준화되었습니다: {', '.join(columns_to_standardize)}")
    except Exception as e:
        print(f"오류: 표준화 중 문제가 발생했습니다. 데이터 타입을 확인해주세요. ({e})")
        # 오류 발생 시 표준화되지 않은 복사본 반환
        return df.copy() # 원본과 동일한 복사본 반환

    return df_standardized

#%%
# 여러 컬럼에 가중치를 적용하여 합산한 후 비율을 계산하는 함수
def get_branch_ratio(df_br: pd.DataFrame, branches_indices: str, weighted_columns: dict) -> dict:

    # 1. 입력된 '연번' 문자열 파싱 및 유효성 검사
    branch_list = [idx.strip() for idx in branches_indices.split(',') if idx.strip()]
    
    # 2. 필수 컬럼 및 가중치 컬럼 존재 여부 확인
    required_data_columns = list(weighted_columns.keys())
        
    all_required_columns = ['연번'] + required_data_columns

    if not all(col in df_br.columns for col in all_required_columns):
        missing_cols = [col for col in all_required_columns if col not in df_br.columns]
        print(f"오류: 필수 컬럼이 DataFrame에 없습니다: {', '.join(missing_cols)}")
        return {}

    # 3. 해당 연번에 맞는 지청 데이터 필터링
    df_selected_branches = df_br[df_br['연번'].isin(branch_list)].copy() # .copy()를 사용하여 SettingWithCopyWarning 방지

    if df_selected_branches.empty:
        print(f"경고: 입력된 '연번' ({branches_indices})에 해당하는 지청을 찾을 수 없습니다.")
        return {}
    
    # 4. 가중치 적용하여 새로운 '계산된_총_값' 컬럼 생성
    df_selected_branches['계산된_총_값'] = 0.0 # 초기화

    for col_name, weight in weighted_columns.items():
        try:
            # 컬럼 데이터를 숫자로 변환하고 NaN은 0으로 처리
            df_selected_branches.loc[:, col_name] = pd.to_numeric(df_selected_branches[col_name], errors='coerce').fillna(0)
            df_selected_branches['계산된_총_값'] += df_selected_branches[col_name] * weight
        except Exception as e:
            print(f"오류: 컬럼 '{col_name}'의 데이터를 처리하는 중 문제가 발생했습니다. ({e})")
            return {}

    # 5. 계산된 총 값의 전체 합계 계산
    total_calculated_value = df_selected_branches['계산된_총_값'].sum()

    if total_calculated_value == 0:
        print(f"경고: 선택된 지청들의 가중치 적용 합계가 0입니다. 비율을 계산할 수 없습니다.")
        return {}

    # 6. 각 지청의 비율 계산 (연번: 비율 형태로 반환)
    branch_ratios = {}
    for index, row in df_selected_branches.iterrows():
        branch_serial_num = row['연번']
        branch_calculated_value = row['계산된_총_값']
        ratio = branch_calculated_value / total_calculated_value
        branch_ratios[branch_serial_num] = ratio
    
    return branch_ratios


#%%
def update_values(df_br: pd.DataFrame, branchs: str, weights: dict, columns: list) -> pd.DataFrame:
    df_br_updated = df_br.copy(deep=True)

    # 1. 입력값 확인
    print("[입력 확인]")
    print("branchs:", branchs)
    print("weights:", weights)
    print("columns:", columns)

    # 연번을 문자열로 변환해두면 안전함
    df_br_updated['연번'] = df_br_updated['연번'].astype(str)

    # 비율 계산 함수 호출
    ratios = get_branch_ratio(df_br_updated, branchs, weights)
    print("[계산된 비율 ratios]:", ratios)

    # 2. 유효한 컬럼 확인
    valid_cols = [col for col in columns if col in df_br_updated.columns]
    missing_cols = [col for col in columns if col not in df_br_updated.columns]

    if missing_cols:
        print(f"[경고] 다음 컬럼이 누락되어 무시됩니다: {', '.join(missing_cols)}")
    if not valid_cols:
        print("[오류] 분배할 유효한 컬럼이 없습니다. 원본 반환.")
        return df_br_updated

    # 3. 연번 리스트 파싱
    idx_list = [idx.strip() for idx in branchs.split(',') if idx.strip()]
    print("[대상 연번]:", idx_list)

    results = []  # 각 컬럼별 총합 결과

    # 4. 각 컬럼별 총합 계산
    for col_name in valid_cols:
        total_sum = 0.0
        for idx in idx_list:
            values = df_br_updated.loc[df_br_updated['연번'] == idx, col_name].values
            if len(values) > 0 and pd.notna(values[0]):
                total_sum += float(values[0])
        results.append((col_name, total_sum))

    print("[컬럼별 총합]", results)

    # 5. 비율에 따라 각 행에 분배
    for col, total in results:
        for idx, ratio in ratios.items():
            if not isinstance(total, (int, float)):
                print(f"[오류] 컬럼 {col}의 총합(total)이 숫자가 아님: {total}")
                continue

            target_idx = df_br_updated[df_br_updated['연번'] == idx].index
            if len(target_idx) == 0:
                print(f"[경고] 연번 {idx}에 해당하는 행이 없습니다. 건너뜀.")
                continue

            value = float(total) * float(ratio)
            df_br_updated = update_value(df_br_updated, idx, col, value)
            print(idx, col, value)

    return df_br_updated

#%%
# df_br의 '연번' 컬럼이 idx이고, column의 값을 value로 update 하는 함수
def update_value(df: pd.DataFrame, idx: str, column: str, value) -> pd.DataFrame:
    df_updated = df.copy(deep=True)
    
    # '연번' 컬럼이 문자열일 수 있도록 변환
    df_updated['연번'] = df_updated['연번'].astype(str)
    
    # 조건에 맞는 인덱스 추출
    target_idx = df_updated[df_updated['연번'] == idx].index
    
    if len(target_idx) == 0:
        print(f"[경고] 연번 '{idx}'에 해당하는 행이 없습니다.")
        return df_updated
    
    # 값 업데이트
    df_updated.loc[target_idx, column] = value
    print(f"[업데이트] 연번: {idx}, 컬럼: {column}, 값: {value}")
    
    return df_updated

#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%%
# 1. 지청 기초자료 읽기
# 1-1. 파일명 설정 후 read_excel로 읽어오기
# 입력파일명 설정
input_file = '1.지청_기초_자료.xlsx'
# 연번은 문자열로 읽음(나중에 지청신설 시 00-1, 00-2 등으로 구분하기 위함)
df_br = pd.read_excel(input_file, dtype={'연번': str})
# %%

df_br.head()
# %%
df_br['연번']

#%%
# 시군구 사업체수, 종사자수 엑셀 파일의 모든 컬럼을 문자열로 읽어오기
df_sgg = pd.read_excel('시군구_사업체수_종사자수_인구_면적(v4).xlsx', dtype={'연번': str})

#%%
# df_br2에 있는 '관할' 컬럼의 정보로 사업체수, 종사자수, 인구, 면적 구하기
# get_count_by_sgg 함수로 df_br2에 사업체수, 종사자수, 인구, 면적 컬럼 추가
df_br = get_count_by_sgg(df_br, df_sgg)


#%%
# 지청별 기초 데이터만 '2.신설지청포함_기초_자료1.xlsx'로 출력
# '연번','청', '지청','규모', '관할', '기관명', '사업체수_23', '사업체수_22', '종사자수_23', '종사자수_22', '인구수_24', '인구수_23', '인구수_22', '면적_23' 컬럼만 출력
df_br_basic = df_br[['연번','청', '지청','규모', '관할', '기관명', '사업체수_23', '사업체수_22', '종사자수_23', '종사자수_22', '인구수_24', '인구수_23', '인구수_22', '면적_23']]
df_br_basic.to_excel('2.신설지청포함_기초_자료1.xlsx', index=False)

#%%
df_br = add_br(df_br, '01-2', '서울청', '서울서초', '지청', '서울 서초구, 서울 중구')

#%%
df_br = add_sgg(df_br, '01', '테스트')
#%%
df_br.head(1)

#%%
df_br = del_sgg(df_br, '01', '테스트')

# %%
print(get_sgg(df_br, '01'))
print(get_sgg(df_br, '01-2'))



# %%
# df_br2 변수의 사업체수_23, 사업체수_22, 종사자수_23, 종사자수_22, 인구수_24, 인구수_23, 인구수_22, 면적_23 을 0~1로 표준화
columns_to_norm = [
    '사업체수_23', '사업체수_22', '종사자수_23', '종사자수_22', '인구수_24', '인구수_23', '인구수_22', '면적_23'
]
df_br3 = standardize_columns(df_br3, columns_to_norm)

# %%
df_br3.tail()
# %%

#%%
weights = {'사업체수_23': 1}
ratios1 = get_branch_ratio(df_br3, "01, 02", weights)
print(ratios1)
# %%
df_br3.tail(2)
# %%
branchs = '01, 01-2'
weights = {'사업체수_23': 0.8, '종사자수_23': 0.2}
columns = ['총정원', '정원']

df_up = update_values(df_br3, branchs, weights, columns)
# %%
df_up[['정원', '총정원']]
# %%
df_up.head(2)
# %%
df_up.tail(2)
# %%
