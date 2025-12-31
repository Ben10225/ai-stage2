import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# 確保 models 資料夾存在
save_dir = "models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 儲存路徑
save_path = os.path.join(save_dir, "classifier_88plus.pth")

# --- 1. 資料準備 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

df = pd.read_csv("csv/tokenized/shuffled_articles.csv")
d2v_model = Doc2Vec.load("models/doc2vec.model")

# 標籤編碼
le = LabelEncoder()
df["label"] = le.fit_transform(df["Board"])
num_classes = len(le.classes_)

# 準備 X (向量) 和 y (標籤)
X = np.array([d2v_model.dv[i] for i in range(len(df))])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- 2. 建立 PyTorch Dataset ---
class ArticleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(ArticleDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ArticleDataset(X_test, y_test), batch_size=32)


# --- 3. 定義 MLP 模型 ---
class MultiClassClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiClassClassification, self).__init__()
        self.net = nn.Sequential(
            # 輸入層 -> 隱藏層 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # 隱藏層 1 -> 隱藏層 2
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            # 隱藏層 2 -> 輸出層
            nn.Linear(32, output_dim),
            # 注意：訓練時這裡「不放」Softmax，直接輸出 Raw Scores (Logits)
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型
# input_dim = 70 (Doc2Vec), hidden_dim 建議 64~128, output_dim 為看板總數
model = MultiClassClassification(input_dim=150, hidden_dim=128, output_dim=num_classes)

# Categorical Cross Entropy 在 PyTorch 中就是 CrossEntropyLoss
# 它會自動幫你的輸出做 Softmax + Log + NLLLoss
criterion = nn.CrossEntropyLoss()

# 優化器常用 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 當 Acc 三個 Epoch 沒進步，就把 LR 除以 2
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "max", patience=3, factor=0.5
)

# --- 5. 訓練迴圈 ---

target_accuracy = 85
epochs = 30  # 把上限設高，讓它有時間衝刺

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 驗證階段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    current_acc = 100 * correct / total
    print(
        f"Epoch [{epoch+1}], Loss: {total_loss/len(train_loader):.4f}, Acc: {current_acc:.2f}%"
    )

    # Scheduler 會根據 current_acc 是否不再進步，來決定要不要調降 LR
    scheduler.step(current_acc)

    # 檢查是否達到目標
    if current_acc >= target_accuracy:
        print(f"🎉 達到目標準確率 {target_accuracy}% 停止訓練")
        torch.save(model.state_dict(), save_path)
        break
