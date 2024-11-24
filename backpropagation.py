import json
import numpy as np
import matplotlib.pyplot as plt


# 활성화 함수
def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    output = 1 / (1 + np.exp(-x))
    output = np.clip(output, 1e-7, 1 - 1e-7)  # 클리핑 추가
    if derivative:
        return output * (1 - output)
    return output

# 손실 함수
def mse_loss(y_true, y_pred, derivative=False):
    if derivative:
        return y_pred - y_true
    return np.mean((y_pred - y_true) ** 2)

# 데이터 로드 함수
def load_data(train_file, test_file):
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    X_train = np.array([item["features"] for item in train_data])  # shape: (n_samples, n_features)
    y_train = np.array([item["label"] for item in train_data]).reshape(-1, 1)  # shape: (n_samples, 1)
    X_test = np.array([item["features"] for item in test_data])
    y_test = np.array([item["label"] for item in test_data]).reshape(-1, 1)

    return X_train, y_train, X_test, y_test


# Adam Optimizer 클래스
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(param) for key, param in params.items()}
        if self.v is None:
            self.v = {key: np.zeros_like(param) for key, param in params.items()}

        self.t += 1

        for key in params.keys():
            # 편향된 1차 및 2차 모멘텀 계산
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 편향 보정된 모멘텀 계산
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 파라미터 업데이트
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


# 신경망 클래스
class NN:
    def __init__(self):
        # 초기화: 레이어 정의
        self.layer_1 = self.initialize_layer(2, 64)
        self.layer_2 = self.initialize_layer(64, 128)
        self.layer_3 = self.initialize_layer(128, 64)
        self.layer_4 = self.initialize_layer(64, 1)
        
        # Adam Optimizer 초기화
        self.optimizer = AdamOptimizer(learning_rate=0.003)

    def initialize_layer(self, input_features, output_features):
        std = np.sqrt(2.0 / input_features)  # He Initialization
        weights = np.random.normal(0, std, (input_features, output_features))
        biases = np.zeros((1, output_features))
        return {"weights": weights, "biases": biases}

    def forward(self, X):
        # 순전파 단계
        self.z1 = np.dot(X, self.layer_1["weights"]) + self.layer_1["biases"]
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.layer_2["weights"]) + self.layer_2["biases"]
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.layer_3["weights"]) + self.layer_3["biases"]
        self.a3 = relu(self.z3)

        self.z4 = np.dot(self.a3, self.layer_4["weights"]) + self.layer_4["biases"]
        self.a4 = sigmoid(self.z4)
        return sigmoid(self.z4)

    def backward(self, X, y):
        # 출력층 기울기 계산
        dz4 = mse_loss(self.a4, y, derivative=True) * sigmoid(self.z4, derivative=True)
        dw4 = np.dot(self.a3.T, dz4)
        db4 = np.sum(dz4, axis=0)

        dz3 = np.dot(dz4, self.layer_4["weights"].T) * relu(self.z3, derivative=True)
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0)

        dz2 = np.dot(dz3, self.layer_3["weights"].T) * relu(self.z2, derivative=True)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, self.layer_2["weights"].T) * relu(self.z1, derivative=True)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # 기울기와 파라미터를 딕셔너리로 정리
        params = {
            "layer_4_weights": self.layer_4["weights"],
            "layer_4_biases": self.layer_4["biases"],
            "layer_3_weights": self.layer_3["weights"],
            "layer_3_biases": self.layer_3["biases"],
            "layer_2_weights": self.layer_2["weights"],
            "layer_2_biases": self.layer_2["biases"],
            "layer_1_weights": self.layer_1["weights"],
            "layer_1_biases": self.layer_1["biases"],
        }
        
        grads = {
            "layer_4_weights": dw4,
            "layer_4_biases": db4,
            "layer_3_weights": dw3,
            "layer_3_biases": db3,
            "layer_2_weights": dw2,
            "layer_2_biases": db2,
            "layer_1_weights": dw1,
            "layer_1_biases": db1,
        }

        # Adam Optimizer로 파라미터 업데이트
        updated_params = self.optimizer.update(params=params, grads=grads)

        # 업데이트된 파라미터를 다시 저장
        for key in updated_params:
            layer_name, param_type = key.split("_")
            getattr(self, layer_name)[param_type] = updated_params[key]


# Smoothing 함수 정의
def smooth_curve(values, beta=0.9):
    smoothed_values = []
    avg_value = 0
    for value in values:
        avg_value = beta * avg_value + (1 - beta) * value
        smoothed_values.append(avg_value / (1 - beta ** (len(smoothed_values) + 1)))
    return smoothed_values


# 학습 함수
def train_model(model, X_train, y_train, X_test, y_test, epochs=1500):
    train_losses = []  # 훈련 손실 저장 리스트
    test_losses = []   # 테스트 손실 저장 리스트

    for epoch in range(epochs):
        # 순전파 및 훈련 손실 계산
        y_pred_train = model.forward(X_train)
        train_loss = mse_loss(y_train, y_pred_train)
        train_losses.append(train_loss)

        # 역전파 수행
        model.backward(X_train, y_train)

        # 테스트 데이터 손실 계산
        y_pred_test = model.forward(X_test)
        test_loss = mse_loss(y_test, y_pred_test)
        test_losses.append(test_loss)

        # Epoch별 결과 출력
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    # 손실 그래프 시각화 (Smoothing 적용)
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), smooth_curve(train_losses), label='Smoothed Train Loss', color='blue')
    plt.plot(range(epochs), smooth_curve(test_losses), label='Smoothed Test Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()


# 데이터 경로 설정 및 실행
train_file = "spiral_trainset.json"
test_file = "spiral_testset.json"

X_train, y_train, X_test, y_test = load_data(train_file=train_file,test_file=test_file )

model=NN()
train_model(model,X_train,y_train,X_test,y_test )