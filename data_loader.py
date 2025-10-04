import pandas as pd
import matplotlib.pyplot as plt

# 1) 파일 불러오기
df = pd.read_csv("Occupancy_Estimation.csv")

# 2) Datetime 구성
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)

# 3) 파생변수
df['Temp_avg']  = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)
df['Temp_diff'] = df['Temp_avg'].diff()
df['CO2_Slope'] = df['S5_CO2'].diff()

# 4) 슬림 뷰 (오탈자 수정: S5_CO2_Slope -> CO2_Slope)
df_slim = df[['Temp_avg', 'Temp_diff', 'S5_CO2', 'CO2_Slope', 'Room_Occupancy_Count']]
print(df_slim.info())

# 5) 모델 입력용 클린 데이터 (명시적 복사로 SettingWithCopy 방지)
df_clean = df[['Temp_avg', 'Temp_diff', 'S5_CO2', 'CO2_Slope', 'Room_Occupancy_Count']].dropna().copy()

# 6) 상태 레벨 지정 함수
def level_state(row):
    if row['Temp_avg'] >= 27 or row['S5_CO2'] >= 1500:
        return 2  # 위험
    elif row['Temp_avg'] >= 26 or row['S5_CO2'] >= 1000:
        return 1  # 불쾌
    else:
        return 0  # 정상

df_clean['State'] = df_clean.apply(level_state, axis=1)
print(df_clean.head(10))

# 7) 분포 그래프
ax = df_clean['State'].value_counts().sort_index().plot(
    kind='bar', color=['green', 'orange', 'red'], figsize=(6,4), title='상태 레벨 분포'
)
ax.set_xlabel('State (0=정상,1=불쾌,2=위험)')
ax.set_ylabel('Count')
plt.show()

# 8) 기준선 확인 그래프
plt.figure(figsize=(14,3))
plt.plot(df_clean.index, df_clean['Temp_avg'], label='Temp_avg')
plt.axhline(26, linestyle='--', label='Temp=26°C')
plt.axhline(27, linestyle='--', label='Temp=27°C')
plt.title('온도와 불쾌/위험 기준선')
plt.legend(); plt.grid(True, axis='y'); plt.show()

plt.figure(figsize=(14,3))
plt.plot(df_clean.index, df_clean['S5_CO2'], label='CO₂')
plt.axhline(1000, linestyle='--', label='CO₂=1000ppm')
plt.axhline(1500, linestyle='--', label='CO₂=1500ppm')
plt.title('CO₂와 불쾌/위험 기준선')
plt.legend(); plt.grid(True, axis='y'); plt.show()

# 9) CSV 저장: Datetime이 index로 날아가지 않도록 reset_index 후 컬럼 맨 앞에 배치
out = df_clean.reset_index()  # ← Datetime 컬럼으로 복원
# (선택) 컬럼 순서 정리
cols = ['Datetime', 'Temp_avg', 'Temp_diff', 'S5_CO2', 'CO2_Slope', 'Room_Occupancy_Count', 'State']
out = out[cols]

out.to_csv("preprocessed_env_dataset.csv", index=False)
print("'preprocessed_env_dataset.csv' 저장 완료")