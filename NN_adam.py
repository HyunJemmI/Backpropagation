import numpy as np
import matplotlib.pyplot as plt
import json

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
    return np.mean((y_pred - y_true) ** 2) / 2

# 데이터 로드 함수
def load_data(train_file, test_file):
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    X_train = np.array([item["features"] for item in train_data])
    y_train = np.array([item["label"] for item in train_data]).reshape(-1, 1)
    X_test = np.array([item["features"] for item in test_data])
    y_test = np.array([item["label"] for item in test_data]).reshape(-1, 1)

    return X_train, y_train, X_test, y_test

# 신경망 클래스
class NN:
    def __init__(self):
        self.layer_1 = self.initialize_layer(2, 64)
        self.layer_2 = self.initialize_layer(64, 128)
        self.layer_3 = self.initialize_layer(128, 64)
        self.layer_4 = self.initialize_layer(64, 1)
        self.initialize_adam()

    def initialize_layer(self, input_features, output_features):
        std = 1 / input_features
        weights = np.random.normal(0, std, (input_features, output_features))
        biases = np.zeros((1, output_features))
        return {"weights": weights, "biases": biases}

    def initialize_adam(self):
        self.adam_params = {}
        for i in range(1, 5):
            self.adam_params[f"layer_{i}_m_w"] = 0
            self.adam_params[f"layer_{i}_v_w"] = 0
            self.adam_params[f"layer_{i}_m_b"] = 0
            self.adam_params[f"layer_{i}_v_b"] = 0

    def forward(self, X):
        self.z1 = np.dot(X, self.layer_1["weights"]) + self.layer_1["biases"]
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.layer_2["weights"]) + self.layer_2["biases"]
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.layer_3["weights"]) + self.layer_3["biases"]
        self.a3 = relu(self.z3)

        self.z4 = np.dot(self.a3, self.layer_4["weights"]) + self.layer_4["biases"]
        self.a4 = sigmoid(self.z4)

        return self.a4

    def adam_update(self, layer, dw, db, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        m_w = self.adam_params[f"{layer}_m_w"]
        v_w = self.adam_params[f"{layer}_v_w"]
        m_b = self.adam_params[f"{layer}_m_b"]
        v_b = self.adam_params[f"{layer}_v_b"]

        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        self.adam_params[f"{layer}_m_w"] = m_w
        self.adam_params[f"{layer}_v_w"] = v_w
        self.adam_params[f"{layer}_m_b"] = m_b
        self.adam_params[f"{layer}_v_b"] = v_b

        return learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon), learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def backward(self, X, y, t, learning_rate=0.003):
        dz4 = mse_loss(y, self.a4, derivative=True) * sigmoid(self.z4, derivative=True)
        dw4 = np.dot(self.a3.T, dz4)
        db4 = np.sum(dz4, axis=0, keepdims=True)
        dw4, db4 = self.adam_update("layer_4", dw4, db4, t, learning_rate)

        dz3 = np.dot(dz4, self.layer_4["weights"].T) * relu(self.z3, derivative=True)
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        dw3, db3 = self.adam_update("layer_3", dw3, db3, t, learning_rate)

        dz2 = np.dot(dz3, self.layer_3["weights"].T) * relu(self.z2, derivative=True)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dw2, db2 = self.adam_update("layer_2", dw2, db2, t, learning_rate)

        dz1 = np.dot(dz2, self.layer_2["weights"].T) * relu(self.z1, derivative=True)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        dw1, db1 = self.adam_update("layer_1", dw1, db1, t, learning_rate)

        self.layer_4["weights"] -= dw4
        self.layer_4["biases"] -= db4

        self.layer_3["weights"] -= dw3
        self.layer_3["biases"] -= db3

        self.layer_2["weights"] -= dw2
        self.layer_2["biases"] -= db2

        self.layer_1["weights"] -= dw1
        self.layer_1["biases"] -= db1

def decision_boundary(model, X, y):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y,cmap="coolwarm", s=40)
    plt.contourf(xx, yy, Z, levels=30, cmap="coolwarm", alpha=0.5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend(["Neural Network Output: Gaussian"])
    plt.savefig('photos/decision_boundary.png')
    
# 학습 함수
def train_model(model, X_train, y_train, X_test, y_test, epochs=200):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        t = epoch + 1
        y_pred_train = model.forward(X_train)
        train_loss = mse_loss(y_train, y_pred_train)
        train_losses.append(train_loss)

        model.backward(X_train, y_train, t)

        y_pred_test = model.forward(X_test)
        test_loss = mse_loss(y_test, y_pred_test)
        test_losses.append(test_loss)

                # Epoch별 결과 출력 및 그래프 그리기
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
                plt.savefig('photos/loss_plot.png')
                
            # Decision Boundary 그리기
        if epoch % 100 == 0 or epoch == epochs - 1:
            decision_boundary(model, X_train, y_train)


# 데이터 경로 설정 및 실행
train_file = "data_set/spiral_testset.json"
test_file = "data_set/spiral_testset.json"

X_train, y_train, X_test, y_test = load_data(train_file, test_file)

# 모델 초기화 및 학습 시작
model = NN()
train_model(model, X_train, y_train, X_test, y_test)

