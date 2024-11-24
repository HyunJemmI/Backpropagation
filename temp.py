import json
import numpy as np
import matplotlib.pyplot as plt


# 활성화 함수
def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    if derivative:
        y = sigmoid(x)
        return y * (1 - y)
    return 1 / (1 + np.exp(-x))

# 손실 함수
def mse_loss(y_true, y_pred, derivative=False):
    if derivative:
        return (y_pred - y_true)
    return np.mean((y_pred - y_true) ** 2)/2

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


# 신경망 클래스
class NN:
    def __init__(self):
        # 초기화: 레이어 정의
        self.layer_1 = self.initialize_layer(2, 64)
        self.layer_2 = self.initialize_layer(64, 128)
        self.layer_3 = self.initialize_layer(128, 64)
        self.layer_4 = self.initialize_layer(64, 1)

    def initialize_layer(self, input_features, output_features):
        std = 1/ input_features
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
        
        return self.a4

    def backward(self, X, y, learning_rate=0.01):
        # 출력층 기울기 계산
        dz4 = mse_loss(y,self.a4, derivative=True) * sigmoid(self.z4, derivative=True)
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

        # 가중치 및 편향 업데이트 (SGD 방식)
        self.layer_4["weights"] -= learning_rate * dw4
        self.layer_4["biases"] -= learning_rate * db4

        self.layer_3["weights"] -= learning_rate * dw3
        self.layer_3["biases"] -= learning_rate * db3

        self.layer_2["weights"] -= learning_rate * dw2
        self.layer_2["biases"] -= learning_rate * db2

        self.layer_1["weights"] -= learning_rate * dw1
        self.layer_1["biases"] -= learning_rate * db1



def decision_boundary(model, X, y):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.clf()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig('decision_boundary.png')

# 학습 함수
def train_model(model, X_train, y_train, X_test, y_test, epochs=15000):
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
            if epoch > 0:
                plt.clf()
                plt.plot(range(epoch+1), train_losses, label= 'Train Loss', color='blue')
                plt.plot(range(epoch+1), test_losses, label='Test Loss', color='orange')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Test Loss over Epochs')
                plt.legend()
                plt.grid()
                plt.savefig('loss_plot.png')
                
            # Decision Boundary 그리기
        if epoch % 1000 == 0 or epoch == epochs - 1:
            decision_boundary(model, X_train, y_train)



# 데이터 경로 설정 및 실행
train_file = "spiral_trainset.json"
test_file = "spiral_testset.json"

X_train, y_train, X_test, y_test = load_data(train_file=train_file,test_file=test_file )

model=NN()
train_model(model,X_train,y_train,X_test,y_test )