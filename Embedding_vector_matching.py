import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
from sklearn.metrics.pairwise import euclidean_distances
import random
import os
from tqdm import tqdm

# 设置随机种子
def set_seed(seed):
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
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding

# 将数据转换为 PyTorch 张量
def create_tensor(data):
    embedding_features = ['chat_count', 'total_bot_moderator_messages', 'total_bot_otherengage_messages',
                          'total_human_moderator_messages', 'total_broadcaster_messages', 'total_badges_messages', 'GCC']
    return torch.tensor(data[embedding_features].values, dtype=torch.float32).unsqueeze(0)

# 获取实验组嵌入向量
def get_experimental_embeddings(df, model, treat_indices, video_id):
    experimental_embeddings = {}
    valid_windows = []
    n = len(treat_indices)

    valid_indices = [
        idx for i, idx in enumerate(treat_indices)
        if (i == 0 or idx - treat_indices[i - 1] > 20) and  
           (i == n - 1 or treat_indices[i + 1] - idx > 20) and
           idx >= 10 and idx + 10 < len(df)
    ]

    for idx in valid_indices:
        window_data = df.iloc[idx - 10:idx + 1]
        window_tensor = create_tensor(window_data)
        with torch.no_grad():
            embedding = model(window_tensor).numpy().flatten()
        experimental_embeddings[f"{idx-10}-{idx}"] = embedding
        valid_windows.append((idx - 10, idx + 10))

    return {video_id: experimental_embeddings}, valid_windows

# 获取对照组嵌入向量
def get_control_embeddings(df, model, treat_indices, valid_windows, video_id):
    control_embeddings = {}
    length = 10
    for i in range(len(df) - 2 * length):
        if all(i + 2 * length < idx - 10 or i > idx + 10 for idx in treat_indices):
            if all(not ((i + 2 * length >= start and i + 2 * length <= end) or (i >= start and i <= end)) for start, end in valid_windows):
                window_data = df.iloc[i:i + length + 1]
                window_tensor = create_tensor(window_data)
                with torch.no_grad():
                    embedding = model(window_tensor).numpy().flatten()
                control_embeddings[f"{i}-{i+length}"] = embedding
    return {video_id: control_embeddings}

# 匹配嵌入向量
def match_embeddings(experimental_embeddings, control_embeddings, distance_threshold):
    matched_results = []
    for video_id, exp_embeddings in experimental_embeddings.items():
        for exp_range, exp_embed in exp_embeddings.items():
            for ctrl_video_id, ctrl_embeddings in control_embeddings.items():
                for ctrl_range, ctrl_embed in ctrl_embeddings.items():
                    distance = euclidean_distances([exp_embed], [ctrl_embed])[0][0]
                    if distance < distance_threshold:
                        matched_results.append({
                            'video_id': video_id,
                            'exp_range': exp_range,
                            'ctrl_video_id': ctrl_video_id,
                            'ctrl_range': ctrl_range,
                            'distance': distance
                        })
    return matched_results

# 主程序
def main():
    set_seed(666)
    lstm_embedder = LSTMEmbedder(7, 128, 1, 2)
    lstm_embedder.load_state_dict(torch.load('saved_model.pth'))
    lstm_embedder.eval()  # 设置为评估模式

    # 设置路径和阈值
    minute_videos_file_path = 'G:/Dataset/Twitch/processed_data/NewTables/StreamWithVideoWithChat/DifferentModTypes/ContentModeration/Matching-withoutDID/one_min_level_filled_treat/'
    check_files_list = [filename for filename in os.listdir(minute_videos_file_path) if filename.endswith('.csv')]
    distance_threshold = 0.05  # 设置距离阈值

    results = []
    for video_file in tqdm(check_files_list, desc='Getting Treat and Control Slices in a same video chatroom:'):
        file_path = os.path.join(minute_videos_file_path, video_file.replace("'", ""))
        try:
            df = pd.read_csv(file_path)
            video_ids = df['video_id'].unique()
            all_experimental_embeddings, all_control_embeddings = {}, {}

            for video_id in video_ids:
                video_df = df[df['video_id'] == video_id].reset_index(drop=True)
                treat_indices = video_df.index[video_df['current_moderation_count'] > 0].tolist()
                experimental_embeddings, valid_windows = get_experimental_embeddings(video_df, lstm_embedder, treat_indices, video_id)
                all_experimental_embeddings.update(experimental_embeddings)

                control_embeddings = get_control_embeddings(video_df, lstm_embedder, treat_indices, valid_windows, video_id)
                all_control_embeddings.update(control_embeddings)

            # 使用距离阈值进行匹配
            matched_results = match_embeddings(all_experimental_embeddings, all_control_embeddings, distance_threshold)
            results.extend(matched_results)

        except Exception as e:
            print("Error occurred:", e)
            print("Skipping this data:", video_file)

    # 将结果保存到 CSV 文件
    results_df = pd.DataFrame(results, columns=['video_id', 'exp_range', 'ctrl_video_id', 'ctrl_range', 'distance'])
    results_df.to_csv('Results/FeaturesEmbedding_SlicesMatching_SameChatroom_10min_10min_dim32_normalized_threshold.csv', index=False)
    print("Matched results saved to 'Results/FeaturesEmbedding_SlicesMatching_SameChatroom_10min_10min_dim32_normalized_threshold.csv'.")

if __name__ == "__main__":
    main()
