import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("preprocessed_env_dataset.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])

# 기준선
TEMP_THRESHOLDS = [26, 27]
CO2_THRESHOLDS = [1000, 1500]

# 시각화
fig, ax1 = plt.subplots(figsize=(14, 6))

# 온도 축 (왼쪽 y축)
ax1.set_title("→ 온도 및 CO₂ 변화")
ax1.plot(df['Datetime'], df['Temp_avg'], color='tab:blue', label='Temp_avg')
ax1.set_ylabel("온도 (°C)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 온도 기준선
for thresh in TEMP_THRESHOLDS:
    ax1.axhline(thresh, color='tab:blue', linestyle='--', alpha=0.5, label=f'Temp={thresh}°C')

# CO2 축 (오른쪽 y축)
ax2 = ax1.twinx()
ax2.plot(df['Datetime'], df['S5_CO2'], color='tab:red', label='CO₂ (ppm)')
ax2.set_ylabel("CO₂ (ppm)", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# CO₂ 기준선
for thresh in CO2_THRESHOLDS:
    ax2.axhline(thresh, color='tab:red', linestyle='--', alpha=0.5, label=f'CO₂={thresh}ppm')

# 범례 수동 추가
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()


import pandas as pd

# 데이터 불러오기
df = pd.read_csv("preprocessed_env_dataset.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])

# 기준 초과 조건
temp_exceed = df['Temp_avg'] > 26.0
co2_exceed = df['S5_CO2'] > 1000.0

# 각각 초과한 개수
num_temp_exceed = temp_exceed.sum()
num_co2_exceed = co2_exceed.sum()

# 둘 중 하나라도 초과한 개수 (OR 조건)
num_either_exceed = (temp_exceed | co2_exceed).sum()

# 둘 다 동시에 초과한 개수 (AND 조건)
num_both_exceed = (temp_exceed & co2_exceed).sum()

print(f"✅ 기준선 초과 개수 요약:")
print(f"- 온도 > 26.0°C: {num_temp_exceed}개")
print(f"- CO₂ > 1000ppm: {num_co2_exceed}개")
print(f"- 둘 중 하나라도 초과: {num_either_exceed}개")
print(f"- 둘 다 동시에 초과: {num_both_exceed}개")