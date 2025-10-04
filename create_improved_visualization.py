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

def create_improved_visualization():
    """개선된 시각화 생성 - 선 스타일로 더 보기 좋게"""
    print("📊 개선된 자연스러운 진동 패턴 시각화 생성 중...")
    
    # 데이터 로드
    df_original = pd.read_csv('preprocessed_env_dataset.csv')
    df_natural = pd.read_csv('env_dataset_natural_oscillating.csv')
    
    # 시간 축 생성
    time_axis = range(len(df_original))
    
    # 그래프 설정 - 더 큰 사이즈와 개선된 스타일
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    fig.suptitle('자연스러운 진동 패턴 비교 (개선된 시각화)', fontsize=20, fontweight='bold', y=0.95)
    
    # 색상 설정
    original_color = '#2E86AB'  # 파란색
    natural_color = '#A23B72'   # 보라색
    threshold_color = '#F18F01' # 주황색
    
    # 1. 좌측 상단: 온도 (자연스러운 진동 패턴)
    axes[0, 0].plot(time_axis, df_original['Temp_avg'], 
                    color=original_color, alpha=0.8, label='원본 데이터', 
                    linewidth=2.5, linestyle='-')
    axes[0, 0].plot(time_axis, df_natural['Temp_avg'], 
                    color=natural_color, alpha=0.9, label='자연스러운 진동', 
                    linewidth=2.5, linestyle='-')
    axes[0, 0].axhline(y=26.0, color=threshold_color, linestyle='--', 
                       alpha=0.8, linewidth=2, label='온도 기준선 (26°C)')
    
    axes[0, 0].set_title('온도 비교 (자연스러운 진동 패턴)', fontsize=16, fontweight='bold', pad=20)
    axes[0, 0].set_ylabel('온도 (°C)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('시간 (데이터 포인트)', fontsize=12)
    axes[0, 0].legend(fontsize=12, loc='upper right')
    axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0, 0].set_ylim(20, 32)
    
    # 2. 우측 상단: CO2 (자연스러운 진동 패턴)
    axes[0, 1].plot(time_axis, df_original['S5_CO2'], 
                    color=original_color, alpha=0.8, label='원본 데이터', 
                    linewidth=2.5, linestyle='-')
    axes[0, 1].plot(time_axis, df_natural['S5_CO2'], 
                    color=natural_color, alpha=0.9, label='자연스러운 진동', 
                    linewidth=2.5, linestyle='-')
    axes[0, 1].axhline(y=1000.0, color=threshold_color, linestyle='--', 
                       alpha=0.8, linewidth=2, label='CO2 기준선 (1000ppm)')
    
    axes[0, 1].set_title('CO2 비교 (자연스러운 진동 패턴)', fontsize=16, fontweight='bold', pad=20)
    axes[0, 1].set_ylabel('CO2 (ppm)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('시간 (데이터 포인트)', fontsize=12)
    axes[0, 1].legend(fontsize=12, loc='upper right')
    axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0, 1].set_ylim(300, 1700)
    
    # 3. 좌측 하단: 온도 (정교한 진동 패턴) - 원본과 자연스러운 진동만 표시
    axes[1, 0].plot(time_axis, df_original['Temp_avg'], 
                    color=original_color, alpha=0.8, label='원본 데이터', 
                    linewidth=2.5, linestyle='-')
    axes[1, 0].plot(time_axis, df_natural['Temp_avg'], 
                    color=natural_color, alpha=0.9, label='자연스러운 진동', 
                    linewidth=2.5, linestyle='-')
    axes[1, 0].axhline(y=26.0, color=threshold_color, linestyle='--', 
                       alpha=0.8, linewidth=2, label='온도 기준선 (26°C)')
    
    axes[1, 0].set_title('온도 비교 (개선된 시각화)', fontsize=16, fontweight='bold', pad=20)
    axes[1, 0].set_ylabel('온도 (°C)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('시간 (데이터 포인트)', fontsize=12)
    axes[1, 0].legend(fontsize=12, loc='upper right')
    axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[1, 0].set_ylim(20, 32)
    
    # 4. 우측 하단: CO2 (정교한 진동 패턴) - 원본과 자연스러운 진동만 표시
    axes[1, 1].plot(time_axis, df_original['S5_CO2'], 
                    color=original_color, alpha=0.8, label='원본 데이터', 
                    linewidth=2.5, linestyle='-')
    axes[1, 1].plot(time_axis, df_natural['S5_CO2'], 
                    color=natural_color, alpha=0.9, label='자연스러운 진동', 
                    linewidth=2.5, linestyle='-')
    axes[1, 1].axhline(y=1000.0, color=threshold_color, linestyle='--', 
                       alpha=0.8, linewidth=2, label='CO2 기준선 (1000ppm)')
    
    axes[1, 1].set_title('CO2 비교 (개선된 시각화)', fontsize=16, fontweight='bold', pad=20)
    axes[1, 1].set_ylabel('CO2 (ppm)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('시간 (데이터 포인트)', fontsize=12)
    axes[1, 1].legend(fontsize=12, loc='upper right')
    axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[1, 1].set_ylim(300, 1700)
    
    # 전체 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.2)
    
    # 저장
    plt.savefig('improved_natural_oscillating_patterns.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("📊 개선된 자연스러운 진동 패턴 시각화 저장: improved_natural_oscillating_patterns.png")

def create_simple_visualization():
    """간단한 시각화 생성 - 원본과 자연스러운 진동만 비교"""
    print("📊 간단한 자연스러운 진동 패턴 시각화 생성 중...")
    
    # 데이터 로드
    df_original = pd.read_csv('preprocessed_env_dataset.csv')
    df_natural = pd.read_csv('env_dataset_natural_oscillating.csv')
    
    # 시간 축 생성
    time_axis = range(len(df_original))
    
    # 그래프 설정
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('자연스러운 진동 패턴 비교', fontsize=18, fontweight='bold', y=0.95)
    
    # 색상 설정
    original_color = '#1f77b4'  # 파란색
    natural_color = '#ff7f0e'   # 주황색
    threshold_color = '#d62728' # 빨간색
    
    # 1. 좌측: 온도 비교
    axes[0].plot(time_axis, df_original['Temp_avg'], 
                 color=original_color, alpha=0.7, label='원본 데이터', 
                 linewidth=2, linestyle='-')
    axes[0].plot(time_axis, df_natural['Temp_avg'], 
                 color=natural_color, alpha=0.8, label='자연스러운 진동', 
                 linewidth=2, linestyle='-')
    axes[0].axhline(y=26.0, color=threshold_color, linestyle='--', 
                    alpha=0.7, linewidth=2, label='온도 기준선 (26°C)')
    
    axes[0].set_title('온도 비교', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('온도 (°C)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('시간 (데이터 포인트)', fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(20, 32)
    
    # 2. 우측: CO2 비교
    axes[1].plot(time_axis, df_original['S5_CO2'], 
                 color=original_color, alpha=0.7, label='원본 데이터', 
                 linewidth=2, linestyle='-')
    axes[1].plot(time_axis, df_natural['S5_CO2'], 
                 color=natural_color, alpha=0.8, label='자연스러운 진동', 
                 linewidth=2, linestyle='-')
    axes[1].axhline(y=1000.0, color=threshold_color, linestyle='--', 
                    alpha=0.7, linewidth=2, label='CO2 기준선 (1000ppm)')
    
    axes[1].set_title('CO2 비교', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CO2 (ppm)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('시간 (데이터 포인트)', fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(300, 1700)
    
    # 전체 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.2)
    
    # 저장
    plt.savefig('simple_natural_oscillating_patterns.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("📊 간단한 자연스러운 진동 패턴 시각화 저장: simple_natural_oscillating_patterns.png")

def main():
    """메인 함수"""
    print("🚀 개선된 자연스러운 진동 패턴 시각화 생성 시작")
    print("=" * 60)
    
    # 1. 개선된 시각화 생성
    print("\n1️⃣ 개선된 시각화 생성")
    create_improved_visualization()
    
    # 2. 간단한 시각화 생성
    print("\n2️⃣ 간단한 시각화 생성")
    create_simple_visualization()
    
    print("\n✅ 개선된 자연스러운 진동 패턴 시각화 생성 완료!")
    print("=" * 60)
    print("📁 생성된 파일:")
    print("  - improved_natural_oscillating_patterns.png (개선된 시각화)")
    print("  - simple_natural_oscillating_patterns.png (간단한 시각화)")

if __name__ == "__main__":
    main()