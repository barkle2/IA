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
# 주어진 컬럼의 값을 최대값=1이 되도록 표준화 하는 함수
def standardize_columns(df_br: pd.DataFrame, columns_to_edit: list) -> pd.DataFrame:

    # 원본 DataFrame을 직접 수정하지 않기 위해 복사본을 만듭니다.
    df_br_cp = df_br.copy()

    # StandardScaler 인스턴스 생성
    scaler = MinMaxScaler()

    # 표준화할 컬럼들이 DataFrame에 있는지 확인
    missing_cols = [col for col in columns_to_edit if col not in df_br_cp.columns]

    # missing_cols이 비어있으면
    if missing_cols:
        print(f"경고: DataFrame에 없는 컬럼이 있어 표준화를 하지 않습니다: {', '.join(missing_cols)}")
        return df_br
    else:
        # 주어진 값을 '값/최대값'으로 바꾼 표준 컬럼을 추가
        for col in columns_to_edit:
            max_val = df_br_cp[col].max()

            if max_val == 0:
                print(f"주의: '{col}' 컬럼의 최대값이 0이므로 나눗셈을 수행할 수 없습니다. 해당 컬럼은 처리하지 않습니다.")
                continue

            new_col = col + '_표준'
            df_br_cp[new_col] = df_br_cp[col] / max_val

            # 기존 컬럼 바로 뒤에 새 컬럼을 위치시키기 위한 작업
            col_idx = df_br_cp.columns.get_loc(col)
            cols = list(df_br_cp.columns)
            cols.remove(new_col)
            cols.insert(col_idx + 1, new_col)
            df_br_cp = df_br_cp[cols]

    return df_br_cp

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
def update_values(df_br: pd.DataFrame, branchs: str, ratios: dict, columns: list) -> pd.DataFrame:
    df_updated = df_br.copy()

    # 연번 리스트 파싱
    branch_list = [b.strip() for b in branchs.split(',') if b.strip()]
    
    # 비율이 정의된 연번만 필터링
    valid_branches = [b for b in branch_list if b in ratios]
    if not valid_branches:
        print("경고: 주어진 연번이 비율 딕셔너리에 없습니다.")
        return df_updated

    # 컬럼마다 작업 수행
    for col in columns:
        if col not in df_updated.columns:
            print(f"경고: 컬럼 '{col}'이(가) DataFrame에 없습니다. 건너뜁니다.")
            continue

        # 전체 합을 먼저 계산
        total = pd.to_numeric(df_updated.loc[df_updated['연번'].isin(valid_branches), col], errors='coerce').sum()
        
        # 비율 기반 값 계산 및 정수 변환
        values = [round(ratios[b] * total) for b in valid_branches]

        # 정수 변환 후 합계가 달라질 수 있으므로 마지막 항목 보정
        diff = int(total - sum(values))
        if values and diff != 0:
            values[-1] += diff  # 마지막 값에 보정치 적용

        # 업데이트
        for b, v in zip(valid_branches, values):
            df_updated.loc[df_updated['연번'] == b, col] = v

    return df_updated


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
input_file = '1_3_지청_기초_자료(by_program).xlsx'
# 연번은 문자열로 읽음(나중에 지청신설 시 00-1, 00-2 등으로 구분하기 위함)
df_br = pd.read_excel(input_file, dtype={'연번': str})
# %%
df_br.head(2)

#%%
# 시군구 사업체수, 종사자수 엑셀 파일의 모든 컬럼을 문자열로 읽어오기
df_sgg = pd.read_excel('1_2_시군구_사업체수_종사자수_인구_면적_행정통계(v2).xlsx', dtype={'연번': str})

#%%
# 화성지청 신설
df_br = add_br(df_br, '13-1', '중부', '화성', '지청', '경기 화성시, 경기 오산시')
# 경기지청에서 화성시 삭제
df_br = del_sgg(df_br, '13', '경기 화성시')
# 평택지청에서 오산시 삭제
df_br = del_sgg(df_br, '17', '경기 오산시')

# 관할구역 확인
print('경기지청:', get_sgg(df_br, '13'))
print('화성지청:', get_sgg(df_br, '13-1'))
print('평택지청:', get_sgg(df_br, '17'))

#%%
# 광주남부지청 신설
df_br = add_br(df_br, '37-1', '광주', '광주남부', '지청', '광주 동구, 광주 서구, 광주 남구, 전남 함평군, 전남 나주시, 전남 화순군')
# 광주청에서 광주 동구, 서구, 남구, 전남 함평군, 나주시, 화순군 삭제
df_br = del_sgg(df_br, '37', '광주 동구, 광주 서구, 광주 남구, 전남 함평군, 전남 나주시, 전남 화순군')

# 관할구역 확인
print('광주청:', get_sgg(df_br, '37'))
print('광주남부지청:', get_sgg(df_br, '37-1'))

#%%
# 서울서초지청 신설
df_br = add_br(df_br, '01-1', '서울', '서울서초', '지청', '서울 서초구')
# 서울청에서 서울 서초구 삭제
df_br = del_sgg(df_br, '01', '서울 서초구')

# 관할구역 확인
print('서울청:', get_sgg(df_br, '01'))
print('서울서초지청:', get_sgg(df_br, '01-1'))

#%%
# 세종지청 신설
df_br = add_br(df_br, '43-1', '대전', '세종', '지청', '세종 세종시, 충남 공주시, 충남 계룡시')
# 대전청에서 세종 세종시, 충남 공주시, 충남 계룡시 삭제
df_br = del_sgg(df_br, '43', '세종 세종시, 충남 공주시, 충남 계룡시')

# 관할구역 확인
print('대전청:', get_sgg(df_br, '43'))
print('세종지청:', get_sgg(df_br, '43-1'))

#%%
# 남양주지청 신설
df_br = add_br(df_br, '11-1', '중부', '남양주', '지청', '경기 구리시, 경기 남양주시')
# 의정부지청에서 경기 구리시, 경기 남양주시 삭제
df_br = del_sgg(df_br, '11', '경기 구리시, 경기 남양주시')

# 관할구역 확인
print('의정부지청:', get_sgg(df_br, '11'))
print('남양주지청:', get_sgg(df_br, '11-1'))

#%%
# 대구북부지청 신설
df_br = add_br(df_br, '31-1', '대구', '대구북부', '지청', '대구 중구, 대구 북구, 대구 군위군')
# 대구청에서 대구 중구, 대구 북구, 대구 군위군 삭제
df_br = del_sgg(df_br, '31', '대구 중구, 대구 북구, 대구 군위군')

# 관할구역 확인
print('대구청:', get_sgg(df_br, '31'))
print('대구북부지청:', get_sgg(df_br, '31-1'))

#%% 
# 울산동부지청 신설
df_br = add_br(df_br, '27-1', '부산', '울산동부', '지청', '울산 중구, 울산 동구, 울산 북구')
# 울산지청에서 울산 중구, 울산 동구, 울산 북구 삭제
df_br = del_sgg(df_br, '27', '울산 중구, 울산 동구, 울산 북구')

# 관할구역 확인
print('울산지청:', get_sgg(df_br, '27'))
print('울산동부지청:', get_sgg(df_br, '27-1'))

#%%
# 신설된 지청 포함한 파일 출력
df_br.to_excel('2_1_지청_신설.xlsx')

#%%
# get_count_by_sgg 함수로 df_br에 사업체수, 종사자수 컬럼 추가
columns_to_calculate = ['사업체수_23, 종사자수_23, 재해자수_24, 중대재해자수_24, 신고사건_24']
df_br = get_count_by_sgg(df_br, df_sgg, columns_to_calculate)

#%%
# 신설된 지청 포함한 파일 출력
df_br.to_excel('2_2_지청_신설_사업체수_종사자수_재계산.xlsx')

#%%
# 사업체수, 종사자수 합 확인
# 23년 사업체수: 2124670, 23년 종사자수: 19159335
print("23년 사업체수 합:", df_br['사업체수_23'].sum())
print("23년 종사자수 합:", df_br['종사자수_23'].sum())
print("24년 재해자수 합:", df_br['재해자수_24'].sum())
print("24년 중대재해자수 합:", df_br['중대재해자수_24'].sum())
print("24년 신고사건 합:", df_br['신고사건_24'].sum())

#%%
# 사업체수, 종사자수 컬럼을 최대값이 1이 되도록 표준화
columns_to_norm = ['사업체수_23', '종사자수_23', '재해자수_24', '중대재해자수_24', '신고사건_24']
df_br = standardize_columns(df_br, columns_to_norm)

#%%
# 표준화된 컬럼 확인
print(df_br.head(2))

#%%
# 가중치 딕셔너리를 변수로 정의
weights = {
    '사업체수_23_표준': 0.35,
    '종사자수_23_표준': 0.2
}

# 업데이트 할 컬럼: '총정원, 정원, 정원_산안, 정원_노동, 재해자수_24, 중대재해자수_24, 근로손실일수_24, 신고사건_24' 
columns_to_update = ['총정원', '정원', '정원_산안', '정원_노동', '근로손실일수_24']

#%%
# 경기지청과 화성지청, 평택지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '13, 13-1, 17', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '13, 13-1, 17', ratios, columns_to_update)

#%%
# 광주청과 광주남부지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '37, 37-1', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '37, 37-1', ratios, columns_to_update)

#%%
# 서울청과 서울서초지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '01, 01-1', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '01, 01-1', ratios, columns_to_update)

#%%
# 대전청과 세종지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '43, 43-1', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '43, 43-1', ratios, columns_to_update)

#%%
# 의정부지청과 남양주지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '11, 11-1', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '11, 11-1', ratios, columns_to_update)

#%%
# 대구청과 대구북부지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '31, 31-1', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '31, 31-1', ratios, columns_to_update)

#%%
# 울산지청과 울산동부지청 사이의 비율 계산
ratios = get_branch_ratio(df_br, '27, 27-1', weights)

# 결과 출력
print(ratios)

# 비율에 따라 컬럼값 업데이트
df_br = update_values(df_br, '27, 27-1', ratios, columns_to_update)

#%%
# 업데이트된 DataFrame 확인
df_br.to_excel('2_3_지청_신설_컬럼_업데이트.xlsx', index=False)


#%%
# 근로손실일수 컬럼을 0~1로 표준화
columns_to_norm = ['근로손실일수_24']
df_br = standardize_columns(df_br, columns_to_norm)

#%% 
# 표준화된 컬럼까지 파일로 저장
df_br.to_excel('2_4_지청_신설_표준화.xlsx', index=False)


#%%
# 데이터 확인
print(df_br['재해자수_24'].describe())
print(df_br['재해자수_24'].sum())

print(df_br['중대재해자수_24'].describe())
print(df_br['중대재해자수_24'].sum())

print(df_br['근로손실일수_24'].describe())
print(df_br['근로손실일수_24'].sum())

print(df_br['신고사건_24'].describe())
print(df_br['신고사건_24'].sum())

# %%
print(df_br[df_br['연번']=='11-1'])
# %%
