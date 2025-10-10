import os
import re
from datasets import Dataset

def load_multiple_datasets(base_path="/workspaces/summary/Dataset"):
    datasets = {}
    
    # Detect all DUC folders (e.g., DUC2006, DUC2007)
    duc_folders = [f for f in os.listdir(base_path) if f.startswith('DUC') and os.path.isdir(os.path.join(base_path, f))]
    
    for duc_folder in duc_folders:
        duc_path = os.path.join(base_path, duc_folder)
        raw_data_path = os.path.join(duc_path, "raw_data")
        gold_summaries_path = os.path.join(duc_path, "gold_summaries")
        
        if not os.path.exists(raw_data_path) or not os.path.exists(gold_summaries_path):
            print(f"Skipping {duc_folder}: Missing raw_data or gold_summaries")
            continue
        
        data = []
        
        # Load gold summaries
        gold_files = [f for f in os.listdir(gold_summaries_path) if f.endswith('.txt')]
        summaries = {}
        for file in gold_files:
            topic = file.split('_')[0]  # e.g., T1 from T1_1.txt
            with open(os.path.join(gold_summaries_path, file), 'r', encoding='utf-8') as f:
                summary = f.read().strip()
            summaries.setdefault(topic, []).append(summary)

        if not summaries:
            print(f"Skipping {duc_folder}: No gold summaries found")
            continue

        topic_prefix = re.match(r'^[A-Za-z]+', next(iter(summaries.keys()))).group()

        # Prepare mapping topic -> raw folder using numeric alignment (e.g., T1 -> D0601A)
        raw_subfolders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
        folder_map = {}
        for folder in raw_subfolders:
            match = re.search(r'(\d+)', folder)
            if not match:
                continue
            folder_number = int(match.group())
            base = (folder_number // 100) * 100
            topic_index = folder_number - base
            topic_name = f"{topic_prefix}{topic_index}"
            folder_map[topic_name] = folder

        missing_topics = [topic for topic in summaries.keys() if topic not in folder_map]
        if missing_topics:
            print(f"Warning: {duc_folder} missing raw folders for topics: {missing_topics}")

        # Combine summaries per topic and align with correct raw data folder
        for topic, sum_list in summaries.items():
            topic_folder = folder_map.get(topic)
            if not topic_folder:
                continue

            raw_path = os.path.join(raw_data_path, topic_folder)
            articles = []
            for file in sorted(os.listdir(raw_path)):
                file_path = os.path.join(raw_path, file)
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    articles.append(f.read().strip())
            if not articles:
                continue

            combined_article = ' '.join(articles)

            data.append({
                'dataset': duc_folder,
                'topic': topic,
                'article': combined_article,
                'references': sum_list
            })
        
        datasets[duc_folder] = Dataset.from_list(data)
        print(f"Loaded {len(data)} samples for {duc_folder}")
    
    return datasets

def load_duc2006_data(base_path="/workspaces/summary"):
    # Backward compatibility
    return load_multiple_datasets(os.path.dirname(base_path))[os.path.basename(base_path)]

if __name__ == "__main__":
    all_datasets = load_multiple_datasets()
    for name, dataset in all_datasets.items():
        print(f"{name}: {len(dataset)} samples")
        if len(dataset) > 0:
            print(dataset[0])