import os
from datasets import Dataset

def load_multiple_datasets(base_path="/workspaces/summary"):
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
        gold_files = sorted([f for f in os.listdir(gold_summaries_path) if f.endswith('.txt')])
        summaries = {}
        for file in gold_files:
            topic = file.split('_')[0]  # e.g., T1 from T1_1.txt
            with open(os.path.join(gold_summaries_path, file), 'r', encoding='utf-8') as f:
                summary = f.read().strip()
            if topic not in summaries:
                summaries[topic] = []
            summaries[topic].append(summary)
        
        # Combine summaries per topic
        for topic, sum_list in summaries.items():
            combined_summary = ' '.join(sum_list)  # Simple concatenation
            
            # Map topic to raw data folder (e.g., T1 -> D0601A, but check existing folders)
            raw_subfolders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
            # For simplicity, assume first subfolder or map based on topic
            # In DUC2006, T1 uses D0601A, T2 might use another
            topic_folder = None
            if topic == 'T1' and 'D0601A' in raw_subfolders:
                topic_folder = 'D0601A'
            elif topic == 'T2' and 'D0601A' in raw_subfolders:  # Adjust as needed
                topic_folder = 'D0601A'
            else:
                # If no specific mapping, skip or use first available
                if raw_subfolders:
                    topic_folder = raw_subfolders[0]  # Fallback
            
            if topic_folder:
                raw_path = os.path.join(raw_data_path, topic_folder)
                articles = []
                for file in sorted(os.listdir(raw_path)):
                    with open(os.path.join(raw_path, file), 'r', encoding='utf-8') as f:
                        articles.append(f.read().strip())
                combined_article = ' '.join(articles)  # Use all articles, not just 5
                
                data.append({
                    'dataset': duc_folder,
                    'topic': topic,
                    'article': combined_article,
                    'summary': combined_summary
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