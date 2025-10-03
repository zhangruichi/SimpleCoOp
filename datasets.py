import os
import json
from collections import defaultdict
import random
from torch.utils.data import Dataset
from PIL import Image


class StanfordCars(Dataset):
    def __init__(self, root, num_shots=0, seed=0, split="train", transform=None):
        random.seed(seed)
        self.root = root
        self.split_path = os.path.join(self.root, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.root, "split_fewshot")
        with open(self.split_path, "r") as f:
            all_data = json.load(f)
            data = all_data[split]
        if split == "test":
            self.data = data
        else:
            self.data = self.generate_fewshot_dataset(data, num_shots)
        self.transform = transform
        
        
        self.classes = self.get_class_names()

    def generate_fewshot_dataset(self, datas, num_shots):
        tracker = defaultdict(list)
        for item in datas:
            _, label, _ = item
            tracker[label].append(item)
        dataset = []
        for label, item in tracker.items():
            if len(item) >= num_shots:
                sampled_items = random.sample(item, num_shots)
            dataset.extend(sampled_items)
        return dataset
    
    def get_class_names(self):
        class_names = set()
        for _, label, class_name in self.data:
            class_names.add((label, class_name))
        class_names = sorted(list(class_names), key=lambda x: x[0])
        return [name for _, name in class_names]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, _ = self.data[idx]
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":
    datasets = StanfordCars("Path/To/stanford_cars", 4, 0, "train")
    print(datasets)
