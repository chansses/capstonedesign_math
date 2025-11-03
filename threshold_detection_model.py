import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ThresholdDetectionModel:
    """
    기준선 초과 탐지 모델
    - 목적: 다음 시점에 기준선을 넘을지 미리 예측
    - 특징: 온도, CO2, 변화율, 추세 등 활용
    """
    
    def __init__(self, temp_threshold=26.0, co2_threshold=1000.0):
        """
        모델 초기화
        
        Parameters:
        -----------
        temp_threshold : float
            온도 기준선
        co2_threshold : float
            CO2 기준선
        
        Note:
        -----
        High 상태 정의: 온도 > temp_threshold AND CO2 > co2_threshold (둘 다 넘어야 High)
        """
        self.temp_threshold = temp_threshold
        self.co2_threshold = co2_threshold
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df, window_size=3):
        """
        특징 준비
        
        Parameters:
        -----------
        df : pandas.DataFrame
            데이터프레임
        window_size : int
            윈도우 크기 (최근 N개 시점 고려)
            
        Returns:
        --------
        X : np.array
            특징 행렬
        y : np.array
            타겟 (다음 시점에 High 상태인지: 1=High, 0=Normal)
        """
        features_list = []
        targets = []
        
        for i in range(window_size, len(df) - 1):
            # 현재 시점의 특징
            current_temp = df.iloc[i]['Temp_avg']
            current_co2 = df.iloc[i]['S5_CO2']
            current_temp_diff = df.iloc[i]['Temp_diff']
            current_co2_slope = df.iloc[i]['CO2_Slope']
            current_occupancy = df.iloc[i]['Room_Occupancy_Count']
            
            # 최근 윈도우의 평균과 추세
            window_temp = df.iloc[i-window_size:i+1]['Temp_avg'].values
            window_co2 = df.iloc[i-window_size:i+1]['S5_CO2'].values
            
            # 기준선까지의 거리
            distance_to_temp_threshold = current_temp - self.temp_threshold
            distance_to_co2_threshold = current_co2 - self.co2_threshold
            
            # 추세 (최근 N개 시점의 변화율)
            temp_trend = np.mean(np.diff(window_temp)) if len(window_temp) > 1 else 0
            co2_trend = np.mean(np.diff(window_co2)) if len(window_co2) > 1 else 0
            
            # 최근 평균
            temp_avg_recent = np.mean(window_temp)
            co2_avg_recent = np.mean(window_co2)
            
            # 특징 벡터
            features = [
                current_temp,
                current_co2,
                current_temp_diff,
                current_co2_slope,
                current_occupancy,
                distance_to_temp_threshold,
                distance_to_co2_threshold,
                temp_trend,
                co2_trend,
                temp_avg_recent,
                co2_avg_recent,
                # 기준선 초과 여부
                1 if current_temp > self.temp_threshold else 0,
                1 if current_co2 > self.co2_threshold else 0,
            ]
            
            features_list.append(features)
            
            # 타겟: 다음 시점이 High 상태인지 (AND 조건: 둘 다 넘어야 High)
            next_temp = df.iloc[i+1]['Temp_avg']
            next_co2 = df.iloc[i+1]['S5_CO2']
            is_high = 1 if (next_temp > self.temp_threshold and next_co2 > self.co2_threshold) else 0
            targets.append(is_high)
        
        return np.array(features_list), np.array(targets)
    
    def fit(self, df, window_size=3, test_size=0.2, random_state=42):
        """
        모델 학습
        
        Parameters:
        -----------
        df : pandas.DataFrame
            학습 데이터
        window_size : int
            윈도우 크기
        test_size : float
            테스트 데이터 비율
        random_state : int
            랜덤 시드
        """
        print("🔄 특징 준비 중...")
        X, y = self.prepare_features(df, window_size)
        print(f"✅ 특징 준비 완료: {X.shape[0]}개 샘플, {X.shape[1]}개 특징")
        print(f"📊 타겟 분포: High={np.sum(y)}개 ({np.sum(y)/len(y)*100:.2f}%), Normal={len(y)-np.sum(y)}개")
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest 모델 학습
        print("\n🌲 Random Forest 모델 학습 중...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # 클래스 불균형 해결
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 학습 성능 평가
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred)
        
        print(f"✅ 학습 완료!")
        print(f"   학습 정확도: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   학습 F1-Score: {train_f1:.4f} ({train_f1*100:.2f}%)")
        
        # 테스트 성능 평가
        test_pred = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        
        print(f"\n📊 테스트 성능:")
        print(f"   정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"   Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
        print(f"   F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, test_pred)
        print(f"\n📊 혼동 행렬:")
        print(f"   실제\\예측  Normal  High")
        print(f"   Normal      {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"   High        {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # 특징 중요도
        feature_names = [
            'Temp_avg', 'S5_CO2', 'Temp_diff', 'CO2_Slope', 'Occupancy',
            'Dist_to_Temp_Thresh', 'Dist_to_CO2_Thresh',
            'Temp_Trend', 'CO2_Trend',
            'Temp_Avg_Recent', 'CO2_Avg_Recent',
            'Temp_Exceed', 'CO2_Exceed'
        ]
        
        importances = self.model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 상위 5개 중요 특징:")
        for i, (name, importance) in enumerate(feature_importance[:5], 1):
            print(f"   {i}. {name}: {importance:.4f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def evaluate_on_data(self, df, window_size=3):
        """
        전체 데이터에 대한 평가
        
        Parameters:
        -----------
        df : pandas.DataFrame
            평가 데이터
        window_size : int
            윈도우 크기
        """
        print("\n" + "="*70)
        print("📊 전체 데이터 평가")
        print("="*70)
        
        X, y = self.prepare_features(df, window_size)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        
        print(f"\n✅ 평가 결과:")
        print(f"   정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        
        # Normal 상태에서 High 상태로 전이 예측 성능
        print(f"\n📊 Normal → High 전이 예측 성능:")
        normal_indices = []
        for i in range(window_size, len(df) - 1):
            current_temp = df.iloc[i]['Temp_avg']
            current_co2 = df.iloc[i]['S5_CO2']
            if current_temp <= self.temp_threshold and current_co2 <= self.co2_threshold:
                normal_indices.append(i - window_size)
        
        if len(normal_indices) > 0:
            normal_y = y[normal_indices]
            normal_pred = predictions[normal_indices]
            normal_prob = probabilities[normal_indices]
            
            n2h_precision = precision_score(normal_y, normal_pred, zero_division=0)
            n2h_recall = recall_score(normal_y, normal_pred, zero_division=0)
            n2h_f1 = f1_score(normal_y, normal_pred, zero_division=0)
            
            print(f"   Normal 상태 사례 수: {len(normal_indices)}개")
            print(f"   실제 High로 전이: {np.sum(normal_y)}개")
            print(f"   Precision: {n2h_precision:.4f} ({n2h_precision*100:.2f}%)")
            print(f"   Recall: {n2h_recall:.4f} ({n2h_recall*100:.2f}%)")
            print(f"   F1-Score: {n2h_f1:.4f} ({n2h_f1*100:.2f}%) ← 핵심 지표")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'normal_to_high_f1': n2h_f1 if len(normal_indices) > 0 else 0,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def visualize_feature_importance(self, save_path='feature_importance.png', top_n=10):
        """특징 중요도 시각화"""
        if self.model is None:
            print("모델이 학습되지 않았습니다.")
            return
        
        feature_names = [
            'Temp_avg', 'S5_CO2', 'Temp_diff', 'CO2_Slope', 'Occupancy',
            'Dist_to_Temp_Thresh', 'Dist_to_CO2_Thresh',
            'Temp_Trend', 'CO2_Trend',
            'Temp_Avg_Recent', 'CO2_Avg_Recent',
            'Temp_Exceed', 'CO2_Exceed'
        ]
        
        importances = self.model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        top_features = feature_importance[:top_n]
        names, values = zip(*top_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(names)), values)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 특징 중요도 시각화 저장: {save_path}")


def main():
    """메인 함수"""
    print("🚀 기준선 초과 탐지 모델 실행 시작")
    print("=" * 70)
    print("💡 목적: 다음 시점에 기준선을 넘을지 미리 예측")
    print("💡 High 상태 정의: 온도 > 26°C AND CO2 > 1000ppm (둘 다 넘어야 High)")
    print("=" * 70)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv('env_dataset_natural_oscillating.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"데이터 크기: {len(df)} 행")
    
    # 모델 생성 및 학습
    print("\n" + "=" * 70)
    model = ThresholdDetectionModel(temp_threshold=26.0, co2_threshold=1000.0)
    results = model.fit(df, window_size=3)
    
    # 특징 중요도 시각화
    model.visualize_feature_importance()
    
    # 전체 데이터 평가
    print("\n" + "=" * 70)
    eval_results = model.evaluate_on_data(df)
    
    # 최종 결과 요약
    print("\n" + "=" * 70)
    print("✅ 최종 결과 요약")
    print("=" * 70)
    print(f"전체 정확도: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    print(f"High 상태 탐지 F1-Score: {eval_results['f1']:.4f} ({eval_results['f1']*100:.2f}%)")
    print(f"Normal→High 전이 예측 F1-Score: {eval_results['normal_to_high_f1']:.4f} ({eval_results['normal_to_high_f1']*100:.2f}%) ← 핵심 지표")
    
    if eval_results['normal_to_high_f1'] >= 0.5:
        print("✅ 목표 F1-Score 50% 이상 달성!")
    else:
        print("⚠️  목표 F1-Score 50% 미달")
    
    return model, results, eval_results


if __name__ == '__main__':
    model, results, eval_results = main()
