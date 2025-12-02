import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = './tmp/kuairec'
processed_dataset_path = './tmp/processed/kuairec'

# 定义文件路径
df = pd.read_csv(f'{dataset_path}/big_matrix.csv')

expected_max_user_id = df['user_id'].max()
expected_max_video_id = df['video_id'].max()
unique_viedos = df['video_id'].nunique()

print(f'expected_max_user_id = {expected_max_user_id}')
print(f'expected_max_video_id = {expected_max_video_id}')
print(f'unique_viedos = {unique_viedos}')

df1 = df.copy()

n_videos = df['video_id'].nunique()
n_users = df['user_id'].nunique()
print(f'Current user number: {n_users}')
print(f'Current video number: {n_videos}')

video_counts_threshold = 5
user_seq_len_threshold = 5

while True :
  # 统计每个 video_id 出现的次数
  video_id_counts = df1['video_id'].value_counts()

  # 找到出现次数大于等于 video_counts_threshold 的 video_id
  popular_video_ids = video_id_counts[video_id_counts >= video_counts_threshold].index

  # 筛选 df1，只保留 video_id 在 popular_video_ids 中的记录
  df1 = df1[df1['video_id'].isin(popular_video_ids)]
  
    # 统计每个 user_id 出现的次数
  user_id_counts = df1['user_id'].value_counts()

  # 找到出现次数大于等于 user_seq_len_threshold 的 user_id
  active_user_ids = user_id_counts[user_id_counts >= user_seq_len_threshold].index

  # 筛选 df1，只保留 user_id 在 active_user_ids 中的记录
  df1 = df1[df1['user_id'].isin(active_user_ids)]
  
  current_n_videos = df1['video_id'].nunique()
  current_n_users = df1['user_id'].nunique()
  
  if n_videos == current_n_videos and n_users == current_n_users :
    break
  else :
    print(f'Current user number: {current_n_users}')
    print(f'Current video number: {current_n_videos}')
    n_videos = current_n_videos
    n_users = current_n_users
    
# 假设数据集已经加载到 DataFrame 中，名为 df
# 计算用户数量
num_users = df1['user_id'].nunique()

# 计算视频数量
num_videos = df1['video_id'].nunique()

# 计算总的交互数量
total_interactions = len(df1)

# 计算每个用户的交互次数
user_interactions_counts = df1.groupby('user_id').size()

# 计算平均序列长度
average_sequence_length = user_interactions_counts.mean()

# 打印统计信息
print(f"Number of users: {num_users}")
print(f"Number of videos: {num_videos}")
print(f"Total interactions: {total_interactions}")
print(f"Average sequence length: {average_sequence_length:.2f}")

# 输出预期的最大值
expected_max_user_id = df1['user_id'].max()
expected_max_video_id = df1['video_id'].max()
print(f"Expected maximum user_id: {expected_max_user_id}")
print(f"Expected maximum video_id: {expected_max_video_id}")

df1['timestamp'] = (df1['timestamp'] * 1000).astype('int64')

import os
import pandas as pd
import csv

# 创建video_id到新ID的映射，并按原始video_id排序
unique_video_ids = df1['video_id'].unique()
sorted_video_ids = sorted(unique_video_ids)  # 按原始ID排序
video_id_to_new = {vid: idx for idx, vid in enumerate(sorted_video_ids)}

if not os.path.exists(processed_dataset_path) :
  os.makedirs(processed_dataset_path)

# 保存映射信息到CSV文件
mapping_output_path = f'{processed_dataset_path}/video_id_mapping.csv'
mapping_df = pd.DataFrame(list(video_id_to_new.items()), columns=['original_video_id', 'new_id'])
mapping_df.to_csv(mapping_output_path, index=False)

# 初始化输出文件内容
output_data = []

# 逐个用户处理数据
grouped = df1.groupby('user_id')
total_users = len(grouped)

for user_id, group in grouped:
    index = len(output_data)
    
    # 按时间戳升序排列当前用户的数据
    group = group.sort_values(by='timestamp')
    
    # 将video_id映射为新ID并生成序列
    new_video_ids = group['video_id'].map(video_id_to_new).astype(str)
    sequence_item_ids = ",".join(new_video_ids.tolist())
    sequence_ratings = ",".join(["4.0"] * len(group))
    sequence_timestamps = ",".join(group['timestamp'].astype(str).tolist())
    
    output_data.append([index, user_id, sequence_item_ids, sequence_ratings, sequence_timestamps])

    # 检查时间戳是否升序
    time_ms = group['timestamp'].values
    if not all(time_ms[i] <= time_ms[i+1] for i in range(len(time_ms)-1)):
        raise Exception('The timestamp is not monotonically increasing.')

# 写入转换后的数据到CSV文件
output_file = f'{processed_dataset_path}/sasrec_format.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["index", "user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"])
    writer.writerows(output_data)