import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기
df = pd.read_csv("env_dataset_augmented_evenly.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])

# 🔎 온도 & CO₂ 동시 초과 구간 필터링
mask = (df['Temp_avg'] > 26.0) & (df['S5_CO2'] > 1000.0)
df_common = df[mask].copy()

# ✅ 시각화 (동시 초과 구간만)
fig, ax1 = plt.subplots(figsize=(14, 6))

# 온도 (왼쪽)
ax1.plot(df_common['Datetime'], df_common['Temp_avg'], color='tab:blue', label='Temp_avg')
ax1.set_ylabel("온도 (°C)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.axhline(26.0, color='tab:blue', linestyle='--', alpha=0.3, label='Temp=26°C')

# CO₂ (오른쪽)
ax2 = ax1.twinx()
ax2.plot(df_common['Datetime'], df_common['S5_CO2'], color='tab:red', label='CO₂ (ppm)')
ax2.set_ylabel("CO₂ (ppm)", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.axhline(1000.0, color='tab:red', linestyle='--', alpha=0.3, label='CO₂=1000ppm')

# 범례 정리
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("→ Temp > 26°C & CO₂ > 1000ppm 구간만 시각화")
plt.tight_layout()
plt.show()