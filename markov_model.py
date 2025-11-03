import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MarkovChainModel:
    """
    마르코프 체인 모델 클래스
    환경 데이터의 상태 전이를 학습하고 예측하는 모델
    """
    
    def __init__(self, temp_threshold=26.0, co2_threshold=1000.0):
        """
        마르코프 모델 초기화
        
        Parameters:
        -----------
        temp_threshold : float
            온도 기준선 (기본값: 26.0°C)
        co2_threshold : float
            CO2 기준선 (기본값: 1000.0ppm)
        """
        self.temp_threshold = temp_threshold
        self.co2_threshold = co2_threshold
        
        # 상태 정의: (온도 상태, CO2 상태)
        # 0: (Temp <= 26, CO2 <= 1000) - 둘 다 기준선 이하
        # 1: (Temp > 26, CO2 <= 1000) - 온도만 초과
        # 2: (Temp <= 26, CO2 > 1000) - CO2만 초과
        # 3: (Temp > 26, CO2 > 1000) - 둘 다 초과
        self.state_names = ['Normal', 'High_Temp', 'High_CO2', 'High_Both']
        self.n_states = 4
        
        # 상태 전이 행렬
        self.transition_matrix = None
        
        # 초기 상태 분포
        self.initial_state_dist = None
        
    def _get_state(self, temp, co2):
        """
        온도와 CO2 값으로부터 상태 결정
        
        Parameters:
        -----------
        temp : float
            온도 값
        co2 : float
            CO2 값
            
        Returns:
        --------
        int : 상태 인덱스 (0-3)
        """
        temp_exceed = temp > self.temp_threshold
        co2_exceed = co2 > self.co2_threshold
        
        if not temp_exceed and not co2_exceed:
            return 0  # Normal
        elif temp_exceed and not co2_exceed:
            return 1  # High_Temp
        elif not temp_exceed and co2_exceed:
            return 2  # High_CO2
        else:  # temp_exceed and co2_exceed
            return 3  # High_Both
    
    def fit(self, df):
        """
        마르코프 모델 학습
        
        Parameters:
        -----------
        df : pandas.DataFrame
            학습 데이터 (Temp_avg, S5_CO2 컬럼 포함)
        """
        print("🔄 마르코프 모델 학습 중...")
        
        # 상태 시퀀스 생성
        states = []
        for idx, row in df.iterrows():
            state = self._get_state(row['Temp_avg'], row['S5_CO2'])
            states.append(state)
        
        states = np.array(states)
        
        # 초기 상태 분포 계산
        initial_state = states[0]
        self.initial_state_dist = np.zeros(self.n_states)
        self.initial_state_dist[initial_state] = 1.0
        
        # 상태 전이 행렬 계산
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            self.transition_matrix[current_state, next_state] += 1
        
        # 정규화 (각 행의 합이 1이 되도록)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        # 0으로 나누기 방지
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums
        
        # 상태 분포 출력
        print(f"\n📊 상태 분포:")
        for i, state_name in enumerate(self.state_names):
            count = np.sum(states == i)
            percentage = count / len(states) * 100
            print(f"  {state_name}: {count}회 ({percentage:.2f}%)")
        
        print(f"\n📈 상태 전이 행렬:")
        print(self.transition_matrix.round(3))
        
    def predict_next_state(self, current_state):
        """
        현재 상태에서 다음 상태 예측
        
        Parameters:
        -----------
        current_state : int
            현재 상태 인덱스
            
        Returns:
        --------
        int : 예측된 다음 상태 인덱스
        """
        if self.transition_matrix is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 현재 상태에서 다음 상태로의 전이 확률
        transition_probs = self.transition_matrix[current_state, :]
        
        # 가장 확률이 높은 상태 선택
        next_state = np.argmax(transition_probs)
        
        return next_state
    
    def predict_sequence(self, df, start_idx=0, n_steps=None):
        """
        시퀀스 예측
        
        Parameters:
        -----------
        df : pandas.DataFrame
            예측할 데이터
        start_idx : int
            시작 인덱스
        n_steps : int or None
            예측할 스텝 수 (None이면 전체)
            
        Returns:
        --------
        list : 예측된 상태 시퀀스
        """
        if n_steps is None:
            n_steps = len(df) - start_idx
        
        predictions = []
        current_state = self._get_state(df.iloc[start_idx]['Temp_avg'], 
                                       df.iloc[start_idx]['S5_CO2'])
        
        for i in range(n_steps):
            next_state = self.predict_next_state(current_state)
            predictions.append(next_state)
            current_state = next_state
        
        return predictions
    
    def evaluate(self, df):
        """
        모델 평가
        
        Parameters:
        -----------
        df : pandas.DataFrame
            평가 데이터
            
        Returns:
        --------
        dict : 평가 결과 (accuracy, confusion_matrix, report)
        """
        print("\n📊 모델 평가 중...")
        
        # 실제 상태 시퀀스
        actual_states = []
        for idx, row in df.iterrows():
            state = self._get_state(row['Temp_avg'], row['S5_CO2'])
            actual_states.append(state)
        
        # 예측 상태 시퀀스
        predicted_states = []
        for i in range(len(df) - 1):
            current_state = actual_states[i]
            next_state_pred = self.predict_next_state(current_state)
            predicted_states.append(next_state_pred)
        
        # 실제 다음 상태
        actual_next_states = actual_states[1:]
        
        # 정확도 계산
        accuracy = accuracy_score(actual_next_states, predicted_states)
        
        # 혼동 행렬
        cm = confusion_matrix(actual_next_states, predicted_states)
        
        # 분류 리포트
        report = classification_report(actual_next_states, predicted_states, 
                                      target_names=self.state_names, 
                                      output_dict=True)
        
        print(f"\n✅ 평가 결과:")
        print(f"  정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📋 상태별 정확도:")
        for state_name in self.state_names:
            state_idx = self.state_names.index(state_name)
            if state_idx in report:
                precision = report[str(state_idx)]['precision']
                recall = report[str(state_idx)]['recall']
                f1 = report[str(state_idx)]['f1-score']
                support = report[str(state_idx)]['support']
                print(f"  {state_name}:")
                print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Support: {support}")
        
        print(f"\n📊 혼동 행렬:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predicted_states': predicted_states,
            'actual_states': actual_next_states
        }
    
    def visualize_transition_matrix(self, save_path='markov_transition_matrix.png'):
        """
        상태 전이 행렬 시각화
        
        Parameters:
        -----------
        save_path : str
            저장 경로
        """
        if self.transition_matrix is None:
            print("모델이 학습되지 않았습니다.")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.transition_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='Blues',
                   xticklabels=self.state_names,
                   yticklabels=self.state_names,
                   cbar_kws={'label': 'Transition Probability'})
        plt.title('Markov Chain Transition Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Next State', fontsize=12)
        plt.ylabel('Current State', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 상태 전이 행렬 시각화 저장: {save_path}")
    
    def visualize_predictions(self, df, predictions, actual_states, save_path='markov_predictions.png'):
        """
        예측 결과 시각화
        
        Parameters:
        -----------
        df : pandas.DataFrame
            데이터프레임
        predictions : list
            예측된 상태
        actual_states : list
            실제 상태
        save_path : str
            저장 경로
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 시간 축
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        time_axis = df['Datetime'].iloc[1:len(predictions)+1]
        
        # 첫 번째 서브플롯: 실제 상태
        axes[0].plot(time_axis, actual_states, 'o-', label='Actual State', alpha=0.7, markersize=3)
        axes[0].set_ylabel('State', fontsize=12)
        axes[0].set_title('Actual States Over Time', fontsize=14, fontweight='bold')
        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].set_yticklabels(self.state_names)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 두 번째 서브플롯: 예측 상태
        axes[1].plot(time_axis, predictions, 's-', label='Predicted State', alpha=0.7, markersize=3, color='orange')
        axes[1].set_ylabel('State', fontsize=12)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_title('Predicted States Over Time', fontsize=14, fontweight='bold')
        axes[1].set_yticks([0, 1, 2, 3])
        axes[1].set_yticklabels(self.state_names)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 예측 결과 시각화 저장: {save_path}")


def main():
    """메인 함수"""
    print("🚀 마르코프 모델 실행 시작")
    print("=" * 60)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv('env_dataset_natural_oscillating.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"데이터 크기: {len(df)} 행")
    
    # 학습/테스트 분할 (80:20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"\n📊 데이터 분할:")
    print(f"  학습 데이터: {len(df_train)} 행")
    print(f"  테스트 데이터: {len(df_test)} 행")
    
    # 마르코프 모델 생성 및 학습
    print("\n" + "=" * 60)
    model = MarkovChainModel(temp_threshold=26.0, co2_threshold=1000.0)
    model.fit(df_train)
    
    # 상태 전이 행렬 시각화
    model.visualize_transition_matrix()
    
    # 모델 평가
    print("\n" + "=" * 60)
    results = model.evaluate(df_test)
    
    # 예측 결과 시각화
    print("\n" + "=" * 60)
    print("📊 예측 결과 시각화 생성 중...")
    model.visualize_predictions(df_test, results['predicted_states'], results['actual_states'])
    
    # 최종 결과 요약
    print("\n" + "=" * 60)
    print("✅ 최종 결과 요약")
    print("=" * 60)
    print(f"정확도: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    if results['accuracy'] >= 0.5:
        print("✅ 목표 정확도 50% 이상 달성!")
    else:
        print("⚠️  목표 정확도 50% 미달")
    
    return model, results


if __name__ == '__main__':
    model, results = main()
