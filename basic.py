import pandas as pd


df = pd.read_csv("Occupancy_Estimation.csv")


print(df.head())

df['Temp_avg'] = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)  # 평균 온도
# 예: 24.0~24.5, 24.5~25.0, ...
bins = pd.interval_range(start=22, end=30, freq=0.5, closed='left')  # 범위 조정 가능
df['Temp_bin_05'] = pd.cut(df['Temp_avg'], bins=bins)
bin_counts = df['Temp_bin_05'].value_counts().sort_index()
print(bin_counts)
stats = df.groupby('Temp_bin_05')['Temp_avg'].agg(['count', 'mean', 'min', 'max'])
print(stats)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
df['Temp_bin_05'].value_counts().sort_index().plot(kind='bar', color='orange')
plt.title('0.5°C 온도 구간별 데이터 수')
plt.xlabel('온도 구간 (0.5°C 단위)')
plt.ylabel('데이터 수')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import pandas as pd

#
# Datetime 구성
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)

# 평균 온도
df['Temp_avg'] = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)

# 온도 변화량
df['Temp_diff'] = df['Temp_avg'].diff()

# 선택한 컬럼만 추출
df_slim = df[['Temp_avg', 'Temp_diff', 'S5_CO2', 'S5_CO2_Slope', 'Room_Occupancy_Count']]

print(df_slim.head())


# 1000ppm 이상 비율 확인
high_co2_ratio = (df['S5_CO2'] > 1000).mean()
high_co2_count = (df['S5_CO2'] > 1000).sum()

print(f"1000ppm 초과 비율: {high_co2_ratio * 100:.2f}%")
print(f"1000ppm 초과 횟수: {high_co2_count}")

df_clean = df[['Temp_avg', 'Temp_diff', 'S5_CO2', 'CO2_Slope', 'Room_Occupancy_Count']].dropna()

df['Temp_avg'] = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)  # 평균 온도
df['Temp_diff'] = df['Temp_avg'].diff() # 온도 변화량
df['CO2_Slope'] = df['S5_CO2'].diff()   # CO2 변화량

df1 = df[['Temp_avg', 'Temp_diff', 'S5_CO2', 'S5_CO2_Slope', 'Room_Occupancy_Count']] # 선택한 컬럼만 추출
df_clean = df1[['Temp_avg', 'Temp_diff', 'S5_CO2', 'CO2_Slope', 'Room_Occupancy_Count']].dropna() # 결측치 제거