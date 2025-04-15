import os
import csv
import random

npz_root = 'shapenetcoreV2/npz'
category_id = '03001627'
model_dir = os.path.join(npz_root, category_id)

# 获取所有子文件夹（即模型 ID）
all_models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
all_models.sort()

random.seed(42)
random.shuffle(all_models)

num_total = len(all_models)
num_train = int(num_total * 0.9)
train_models = all_models[:num_train]
test_models = all_models[num_train:]

save_dir = 'shapenetcoreV2'
os.makedirs(save_dir, exist_ok=True)

def save_csv(models, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for model_id in models:
            writer.writerow([category_id, model_id])

save_csv(train_models, os.path.join(save_dir, 'shapenet_train.csv'))
save_csv(test_models, os.path.join(save_dir, 'shapenet_test.csv'))

print(f"✅ CSV 文件已保存：训练集 {len(train_models)} 个，测试集 {len(test_models)} 个")
