import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.utils.data import DataLoader, TensorDataset
import random
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 设置为无显示的后端
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LSTMEmbedder(nn.Module):
    """LSTM model for generating embeddings."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMEmbedder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  
        embedding = self.fc(hn[-1])
        # 对嵌入向量进行归一化
        #normalized_embedding = F.normalize(embedding, p=2, dim=1)  # L2 归一化到单位范数
        #return normalized_embedding
        return embedding

def create_tensor(slices_list, outcome_list):
    features = ['chat_count', 'total_bot_moderator_messages', 'total_bot_otherengage_messages',
                'total_human_moderator_messages', 'total_broadcaster_messages', 'total_badges_messages', 'GCC']
    
    # 优化: 先将列表转换为 numpy 数组再转为 tensor
    X = torch.tensor(np.array([slice[features].values for slice in slices_list]), dtype=torch.float32)
    y = torch.tensor(outcome_list, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)


def get_slices(df, treat_indices):
    slices_list, outcome_list = [], []
    valid_treat_windows = []  
    n = len(treat_indices)  
    valid_indices = [
        idx for i, idx in enumerate(treat_indices)
        if (i == 0 or idx - treat_indices[i - 1] > 20) and  
        (i == n - 1 or treat_indices[i + 1] - idx > 20)    
        and idx >= 10 and idx + 10 < len(df)                
    ]
    for idx in valid_indices:
        treat_window_data = df.iloc[idx - 10:idx + 1]
        outcome = treat_window_data['chat_count'].mean()
        slices_list.append(treat_window_data)
        outcome_list.append(outcome)
        valid_treat_windows.append((idx - 10, idx + 10))

    length = 10
    for i in range(len(df) - 2 * length):
        if all(i + 2 * length < idx - 10 or i > idx + 10 for idx in treat_indices):
            if all(not ((i + 2 * length >= start and i + 2 * length <= end) or (i >= start and i <= end)) for start, end in valid_treat_windows):
                control_window_data = df.iloc[i:i + length + 1]
                outcome = control_window_data['chat_count'].mean()
                slices_list.append(control_window_data)
                outcome_list.append(outcome)

    return slices_list, outcome_list

def train_model(model, data_loader, optimizer, criterion, min_epochs=100, patience=3, smooth_factor=0.1):
    model.train()
    losses = []
    smooth_loss = None

    for epoch in range(200):  # 最大 200 次
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        # 计算平滑损失（指数移动平均）
        if smooth_loss is None:
            smooth_loss = avg_loss  # 初始化平滑损失
        else:
            smooth_loss = smooth_factor * avg_loss + (1 - smooth_factor) * smooth_loss

        # 检查平滑损失趋势
        if epoch >= min_epochs:
            if len(losses) > patience and smooth_loss > min(losses[-patience:]):
                print(f"Stopping early at epoch {epoch+1}")
                break

    return model, losses

def main():
    minute_videos_file_path = 'G:/Dataset/Twitch/processed_data/NewTables/StreamWithVideoWithChat/DifferentModTypes/ContentModeration/Matching-withoutDID/one_min_level_filled_treat/'
    check_files_list = [filename for filename in os.listdir(minute_videos_file_path) if filename.endswith('.csv')]

    #随机抽取50%的训练样本
    check_files_list = random.sample(check_files_list, k=len(check_files_list) // 2)
    
    all_slices_list, all_outcome_list = [], []
    for video_file in tqdm(check_files_list, desc='Combining Training Data'):
        file_path = os.path.join(minute_videos_file_path, video_file)
        try:
            df = pd.read_csv(file_path)
            treat_indices = df.index[df['current_moderation_count'] > 0].tolist()
            slices_list, outcome_list = get_slices(df, treat_indices)
            all_slices_list.extend(slices_list)
            all_outcome_list.extend(outcome_list)
        except Exception as e:
            print("Error occurred:", e)
            print("Skipping this data:", video_file)

    dataset = create_tensor(all_slices_list, all_outcome_list)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    print('Training Vector Embedding Model:')
    
    model = LSTMEmbedder(input_dim=7, hidden_dim=128, output_dim=1, num_layers=2) #目标变量只有1维
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trained_model, losses = train_model(model, data_loader, optimizer, criterion)

    torch.save(trained_model.state_dict(), 'saved_model.pth')
    print("Model saved as 'saved_model.pth'")

    # 绘制 epoch-loss 曲线并保存图像文件
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss_curve.png')
    print("Training loss curve saved as 'training_loss_curve.png'.")

if __name__ == "__main__":
    set_seed(666)
    main()
