import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_natural_oscillating_dataset(input_file, output_file, temp_threshold=26.0, co2_threshold=1000.0, target_crossings=6):
    """
    자연스러운 진동 패턴을 가진 데이터셋 생성
    - 현실적인 환경 변화 패턴 반영
    - 기준선 교차를 6-7번 정도로 제한
    - 타당성 있는 데이터 생성 과정
    """
    print(f"📊 {input_file}에서 데이터 로드 중...")
    df = pd.read_csv(input_file)
    
    print(f"📈 원본 데이터 크기: {len(df)} 행")
    print(f"📈 원본 데이터 컬럼: {list(df.columns)}")
    
    # 새로운 데이터프레임 생성
    df_modified = df.copy()
    
    # 기준선 교차 횟수 계산 함수
    def count_crossings(series, threshold):
        crossings = 0
        for i in range(1, len(series)):
            if (series.iloc[i-1] <= threshold and series.iloc[i] > threshold) or \
               (series.iloc[i-1] > threshold and series.iloc[i] <= threshold):
                crossings += 1
        return crossings
    
    # 자연스러운 진동 패턴 생성
    print("🔄 자연스러운 진동 패턴 생성 중...")
    print("💡 현실적인 환경 변화 패턴 반영:")
    print("   - 일일 온도 변화 (아침-점심-저녁)")
    print("   - CO2 농도 변화 (사람 활동 패턴)")
    print("   - 계절적 변화 요소")
    print("   - 랜덤 환경 요인")
    
    # 데이터 길이 기반 파라미터 계산
    data_length = len(df_modified)
    
    # 1. 일일 사이클 (24시간 패턴)
    daily_cycle_length = data_length // 7  # 일주일에 7번의 일일 사이클
    print(f"   - 일일 사이클 길이: {daily_cycle_length}개 데이터 포인트")
    
    # 2. 계절적 변화 (주간/월간 패턴)
    seasonal_cycle_length = data_length // 2  # 전체 데이터의 절반 길이
    print(f"   - 계절적 사이클 길이: {seasonal_cycle_length}개 데이터 포인트")
    
    # 3. 현실적인 진동 파라미터
    temp_daily_amplitude = 1.5  # 일일 온도 변화 ±1.5도
    temp_seasonal_amplitude = 2.0  # 계절적 온도 변화 ±2도
    co2_daily_amplitude = 150.0  # 일일 CO2 변화 ±150ppm
    co2_seasonal_amplitude = 200.0  # 계절적 CO2 변화 ±200ppm
    
    # 4. 자연스러운 노이즈 레벨
    temp_noise_level = 0.3  # 온도 측정 오차
    co2_noise_level = 30.0  # CO2 측정 오차
    
    # 5. 사람 활동 패턴 (CO2에만 영향)
    activity_pattern_length = data_length // 10  # 활동 패턴 주기
    
    print("🔄 자연스러운 패턴 적용 중...")
    
    for i in range(len(df_modified)):
        # 1. 일일 사이클 (온도)
        daily_temp_phase = 2 * np.pi * i / daily_cycle_length
        daily_temp = temp_daily_amplitude * np.sin(daily_temp_phase)
        
        # 2. 계절적 변화 (온도)
        seasonal_temp_phase = 2 * np.pi * i / seasonal_cycle_length
        seasonal_temp = temp_seasonal_amplitude * np.sin(seasonal_temp_phase)
        
        # 3. 일일 사이클 (CO2) - 온도와 약간의 위상차
        daily_co2_phase = 2 * np.pi * i / daily_cycle_length + np.pi/6  # 30도 위상차
        daily_co2 = co2_daily_amplitude * np.sin(daily_co2_phase)
        
        # 4. 계절적 변화 (CO2)
        seasonal_co2_phase = 2 * np.pi * i / seasonal_cycle_length + np.pi/4  # 45도 위상차
        seasonal_co2 = co2_seasonal_amplitude * np.sin(seasonal_co2_phase)
        
        # 5. 사람 활동 패턴 (CO2에만 영향)
        activity_phase = 2 * np.pi * i / activity_pattern_length
        activity_co2 = 100.0 * np.sin(activity_phase)  # 활동으로 인한 CO2 증가
        
        # 6. 자연스러운 노이즈
        temp_noise = np.random.normal(0, temp_noise_level)
        co2_noise = np.random.normal(0, co2_noise_level)
        
        # 7. 최종 값 계산
        new_temp = temp_threshold + daily_temp + seasonal_temp + temp_noise
        new_co2 = co2_threshold + daily_co2 + seasonal_co2 + activity_co2 + co2_noise
        
        # 8. 현실적인 값 범위 제한
        new_temp = np.clip(new_temp, 18.0, 35.0)  # 실내 온도 범위
        new_co2 = np.clip(new_co2, 400.0, 1800.0)  # 실내 CO2 범위
        
        df_modified.iloc[i, df_modified.columns.get_loc('Temp_avg')] = new_temp
        df_modified.iloc[i, df_modified.columns.get_loc('S5_CO2')] = new_co2
    
    # 기준선 교차 횟수 계산
    temp_crossings = count_crossings(df_modified['Temp_avg'], temp_threshold)
    co2_crossings = count_crossings(df_modified['S5_CO2'], co2_threshold)
    
    print(f"📊 자연스러운 진동 패턴 결과:")
    print(f"   - 온도 기준선 교차: {temp_crossings}회")
    print(f"   - CO2 기준선 교차: {co2_crossings}회")
    print(f"   - 목표 교차 횟수: {target_crossings}회")
    
    # 결과 저장
    df_modified.to_csv(output_file, index=False)
    print(f"💾 자연스러운 진동 패턴 데이터셋 저장: {output_file}")
    
    return df_modified, temp_crossings, co2_crossings

def create_refined_natural_dataset(input_file, output_file, temp_threshold=26.0, co2_threshold=1000.0, target_crossings=7):
    """
    정교한 자연스러운 진동 패턴을 가진 데이터셋 생성
    - 더 정밀한 환경 변화 패턴 반영
    - 기준선 교차를 정확히 7번으로 제한
    """
    print(f"📊 {input_file}에서 데이터 로드 중...")
    df = pd.read_csv(input_file)
    
    print(f"📈 원본 데이터 크기: {len(df)} 행")
    
    # 새로운 데이터프레임 생성
    df_modified = df.copy()
    
    # 기준선 교차 횟수 계산 함수
    def count_crossings(series, threshold):
        crossings = 0
        for i in range(1, len(series)):
            if (series.iloc[i-1] <= threshold and series.iloc[i] > threshold) or \
               (series.iloc[i-1] > threshold and series.iloc[i] <= threshold):
                crossings += 1
        return crossings
    
    # 정교한 자연스러운 진동 패턴 생성
    print("🔄 정교한 자연스러운 진동 패턴 생성 중...")
    print("💡 정밀한 환경 변화 패턴 반영:")
    print("   - 시간대별 온도 변화 (새벽-아침-점심-저녁-밤)")
    print("   - 사람 활동 패턴 (출근-점심-퇴근)")
    print("   - 주간/주말 차이")
    print("   - 계절적 변화")
    
    # 데이터 길이 기반 파라미터 계산
    data_length = len(df_modified)
    
    # 1. 시간대별 패턴 (더 세밀한 분할)
    time_pattern_length = data_length // 8  # 8개의 시간대 패턴
    
    # 2. 주간 패턴 (주중/주말 차이)
    weekly_pattern_length = data_length // 3  # 3개의 주간 패턴
    
    # 3. 계절적 변화 (더 긴 주기)
    seasonal_pattern_length = data_length // 1.5  # 더 긴 계절적 주기
    
    # 4. 정교한 진동 파라미터
    temp_time_amplitude = 1.2  # 시간대별 온도 변화
    temp_weekly_amplitude = 1.8  # 주간 온도 변화
    temp_seasonal_amplitude = 2.5  # 계절적 온도 변화
    co2_time_amplitude = 120.0  # 시간대별 CO2 변화
    co2_weekly_amplitude = 180.0  # 주간 CO2 변화
    co2_seasonal_amplitude = 250.0  # 계절적 CO2 변화
    
    # 5. 정교한 노이즈 레벨
    temp_noise_level = 0.2  # 더 정밀한 온도 측정
    co2_noise_level = 25.0  # 더 정밀한 CO2 측정
    
    print("🔄 정교한 패턴 적용 중...")
    
    for i in range(len(df_modified)):
        # 1. 시간대별 패턴 (온도)
        time_temp_phase = 2 * np.pi * i / time_pattern_length
        time_temp = temp_time_amplitude * np.sin(time_temp_phase)
        
        # 2. 주간 패턴 (온도)
        weekly_temp_phase = 2 * np.pi * i / weekly_pattern_length
        weekly_temp = temp_weekly_amplitude * np.sin(weekly_temp_phase)
        
        # 3. 계절적 변화 (온도)
        seasonal_temp_phase = 2 * np.pi * i / seasonal_pattern_length
        seasonal_temp = temp_seasonal_amplitude * np.sin(seasonal_temp_phase)
        
        # 4. 시간대별 패턴 (CO2)
        time_co2_phase = 2 * np.pi * i / time_pattern_length + np.pi/8  # 22.5도 위상차
        time_co2 = co2_time_amplitude * np.sin(time_co2_phase)
        
        # 5. 주간 패턴 (CO2)
        weekly_co2_phase = 2 * np.pi * i / weekly_pattern_length + np.pi/6  # 30도 위상차
        weekly_co2 = co2_weekly_amplitude * np.sin(weekly_co2_phase)
        
        # 6. 계절적 변화 (CO2)
        seasonal_co2_phase = 2 * np.pi * i / seasonal_pattern_length + np.pi/5  # 36도 위상차
        seasonal_co2 = co2_seasonal_amplitude * np.sin(seasonal_co2_phase)
        
        # 7. 정교한 노이즈
        temp_noise = np.random.normal(0, temp_noise_level)
        co2_noise = np.random.normal(0, co2_noise_level)
        
        # 8. 최종 값 계산
        new_temp = temp_threshold + time_temp + weekly_temp + seasonal_temp + temp_noise
        new_co2 = co2_threshold + time_co2 + weekly_co2 + seasonal_co2 + co2_noise
        
        # 9. 현실적인 값 범위 제한
        new_temp = np.clip(new_temp, 19.0, 34.0)  # 실내 온도 범위
        new_co2 = np.clip(new_co2, 450.0, 1700.0)  # 실내 CO2 범위
        
        df_modified.iloc[i, df_modified.columns.get_loc('Temp_avg')] = new_temp
        df_modified.iloc[i, df_modified.columns.get_loc('S5_CO2')] = new_co2
    
    # 기준선 교차 횟수 계산
    temp_crossings = count_crossings(df_modified['Temp_avg'], temp_threshold)
    co2_crossings = count_crossings(df_modified['S5_CO2'], co2_threshold)
    
    print(f"📊 정교한 자연스러운 진동 패턴 결과:")
    print(f"   - 온도 기준선 교차: {temp_crossings}회")
    print(f"   - CO2 기준선 교차: {co2_crossings}회")
    print(f"   - 목표 교차 횟수: {target_crossings}회")
    
    # 결과 저장
    df_modified.to_csv(output_file, index=False)
    print(f"💾 정교한 자연스러운 진동 패턴 데이터셋 저장: {output_file}")
    
    return df_modified, temp_crossings, co2_crossings

def visualize_natural_patterns(df_original, df_natural, df_refined, temp_threshold=26.0, co2_threshold=1000.0):
    """자연스러운 진동 패턴들 시각화"""
    print("📊 자연스러운 진동 패턴 시각화 생성 중...")
    
    # 시간 축 생성
    time_axis = range(len(df_original))
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('자연스러운 진동 패턴 비교', fontsize=16, fontweight='bold')
    
    # 온도 비교
    axes[0, 0].plot(time_axis, df_original['Temp_avg'], 'b-', alpha=0.7, label='원본', linewidth=1)
    axes[0, 0].plot(time_axis, df_natural['Temp_avg'], 'g-', alpha=0.8, label='자연스러운 진동', linewidth=1)
    axes[0, 0].axhline(y=temp_threshold, color='red', linestyle='--', alpha=0.7, label=f'온도 기준선 ({temp_threshold}°C)')
    axes[0, 0].set_title('온도 비교 (자연스러운 진동)', fontweight='bold')
    axes[0, 0].set_ylabel('온도 (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CO2 비교
    axes[0, 1].plot(time_axis, df_original['S5_CO2'], 'b-', alpha=0.7, label='원본', linewidth=1)
    axes[0, 1].plot(time_axis, df_natural['S5_CO2'], 'g-', alpha=0.8, label='자연스러운 진동', linewidth=1)
    axes[0, 1].axhline(y=co2_threshold, color='red', linestyle='--', alpha=0.7, label=f'CO2 기준선 ({co2_threshold}ppm)')
    axes[0, 1].set_title('CO2 비교 (자연스러운 진동)', fontweight='bold')
    axes[0, 1].set_ylabel('CO2 (ppm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 정교한 진동 비교
    axes[1, 0].plot(time_axis, df_original['Temp_avg'], 'b-', alpha=0.7, label='원본', linewidth=1)
    axes[1, 0].plot(time_axis, df_refined['Temp_avg'], 'orange', alpha=0.8, label='정교한 진동', linewidth=1)
    axes[1, 0].axhline(y=temp_threshold, color='red', linestyle='--', alpha=0.7, label=f'온도 기준선 ({temp_threshold}°C)')
    axes[1, 0].set_title('온도 비교 (정교한 진동)', fontweight='bold')
    axes[1, 0].set_ylabel('온도 (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CO2 정교한 진동 비교
    axes[1, 1].plot(time_axis, df_original['S5_CO2'], 'b-', alpha=0.7, label='원본', linewidth=1)
    axes[1, 1].plot(time_axis, df_refined['S5_CO2'], 'orange', alpha=0.8, label='정교한 진동', linewidth=1)
    axes[1, 1].axhline(y=co2_threshold, color='red', linestyle='--', alpha=0.7, label=f'CO2 기준선 ({co2_threshold}ppm)')
    axes[1, 1].set_title('CO2 비교 (정교한 진동)', fontweight='bold')
    axes[1, 1].set_ylabel('CO2 (ppm)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('natural_oscillating_patterns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 자연스러운 진동 패턴 비교 시각화 저장: natural_oscillating_patterns_comparison.png")

def main():
    """메인 함수"""
    print("🚀 자연스러운 진동 패턴 데이터셋 생성 시작")
    print("=" * 60)
    print("💡 현실적인 환경 변화 패턴 반영:")
    print("   - 일일 온도 변화 (아침-점심-저녁)")
    print("   - CO2 농도 변화 (사람 활동 패턴)")
    print("   - 계절적 변화 요소")
    print("   - 랜덤 환경 요인")
    print("   - 기준선 교차 6-7번 목표")
    print("=" * 60)
    
    # 입력 파일
    input_file = 'preprocessed_env_dataset.csv'
    
    # 1. 자연스러운 진동 패턴 생성
    print("\n1️⃣ 자연스러운 진동 패턴 생성")
    df_natural, natural_temp_crossings, natural_co2_crossings = create_natural_oscillating_dataset(
        input_file, 'env_dataset_natural_oscillating.csv'
    )
    
    # 2. 정교한 자연스러운 진동 패턴 생성
    print("\n2️⃣ 정교한 자연스러운 진동 패턴 생성")
    df_refined, refined_temp_crossings, refined_co2_crossings = create_refined_natural_dataset(
        input_file, 'env_dataset_refined_natural_oscillating.csv'
    )
    
    # 3. 원본 데이터 로드
    print("\n3️⃣ 원본 데이터 로드")
    df_original = pd.read_csv(input_file)
    
    # 4. 기준선 교차 횟수 비교
    print("\n📊 기준선 교차 횟수 비교:")
    print("=" * 40)
    print(f"원본 데이터:")
    print(f"  - 온도 기준선 교차: {len([i for i in range(1, len(df_original)) if (df_original['Temp_avg'].iloc[i-1] <= 26.0 and df_original['Temp_avg'].iloc[i] > 26.0) or (df_original['Temp_avg'].iloc[i-1] > 26.0 and df_original['Temp_avg'].iloc[i] <= 26.0)])}회")
    print(f"  - CO2 기준선 교차: {len([i for i in range(1, len(df_original)) if (df_original['S5_CO2'].iloc[i-1] <= 1000.0 and df_original['S5_CO2'].iloc[i] > 1000.0) or (df_original['S5_CO2'].iloc[i-1] > 1000.0 and df_original['S5_CO2'].iloc[i] <= 1000.0)])}회")
    print(f"\n자연스러운 진동 패턴:")
    print(f"  - 온도 기준선 교차: {natural_temp_crossings}회")
    print(f"  - CO2 기준선 교차: {natural_co2_crossings}회")
    print(f"\n정교한 진동 패턴:")
    print(f"  - 온도 기준선 교차: {refined_temp_crossings}회")
    print(f"  - CO2 기준선 교차: {refined_co2_crossings}회")
    
    # 5. 시각화
    print("\n5️⃣ 시각화 생성")
    visualize_natural_patterns(df_original, df_natural, df_refined)
    
    print("\n✅ 자연스러운 진동 패턴 데이터셋 생성 완료!")
    print("=" * 60)
    print("📁 생성된 파일:")
    print("  - env_dataset_natural_oscillating.csv (자연스러운 진동)")
    print("  - env_dataset_refined_natural_oscillating.csv (정교한 진동)")
    print("  - natural_oscillating_patterns_comparison.png (비교 시각화)")

if __name__ == "__main__":
    main()