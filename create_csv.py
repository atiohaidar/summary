import os
import csv
import re

def extract_text_from_doc(file_path):
    """Membaca file, mengekstrak teks dari dalam tag <TEXT>, dan membersihkan tag HTML."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

    # Ekstrak konten di antara <TEXT> dan </TEXT>
    match = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL)
    if not match:
        return ""
    text = match.group(1)
    # Hapus semua tag HTML/XML (seperti <P>) dan ganti dengan spasi
    cleaned_text = re.sub(r'<[^>]+>', ' ', text)
    # Ganti beberapa spasi/newline menjadi satu spasi dan rapikan
    return ' '.join(cleaned_text.split())

def get_folder_name_duc2006(topic_num):
    letter = chr(64 + ((topic_num - 1) % 9) + 1)
    folder = "D06{:02d}{}".format(topic_num, letter)
    return folder

def get_folder_name_duc2007(topic_num):
    huruf_list = ['A']*5 + ['B']*4 + ['C']*4 + ['D']*5 + ['E']*4 + ['F']*4 + ['G']*5 + ['H']*5 + ['I']*5 + ['J']*4
    letter = huruf_list[topic_num - 1]
    folder = "D07{:02d}{}".format(topic_num, letter)
    return folder

def get_content_for_topic(dataset_path, topic_num, dataset='DUC2006'):
    if dataset == 'DUC2006':
        folder = get_folder_name_duc2006(topic_num)
    elif dataset == 'DUC2007':
        folder = get_folder_name_duc2007(topic_num)
    folder_path = os.path.join(dataset_path, 'raw_data', folder)
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        texts.append(extract_text_from_doc(file_path))
    return ' '.join(texts)

def create_csv_for_dataset(dataset_path, output_file, dataset='DUC2006'):
    num_topics = 50 if dataset == 'DUC2006' else 45
    topic_prefix = 'T' if dataset == 'DUC2006' else 'S'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'article', 'highlights'])
        for topic_num in range(1, num_topics + 1):
            content = get_content_for_topic(dataset_path, topic_num, dataset)
            summary_dir = os.path.join(dataset_path, 'gold_summaries')
            summaries = []
            for file_name in os.listdir(summary_dir):
                if file_name.startswith('{}{}_'.format(topic_prefix, topic_num)) and file_name.endswith('.txt'):
                    summary_path = os.path.join(summary_dir, file_name)
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summaries.append(f.read().strip())
            combined_summary = ' '.join(summaries)
            writer.writerow(['{}{}'.format(topic_prefix, topic_num), content, combined_summary])

def main():
    # For DUC2006
    duc2006_path = './Dataset/DUC2006'
    create_csv_for_dataset(duc2006_path, 'duc2006.csv', 'DUC2006')
    
    # For DUC2007
    duc2007_path = './Dataset/DUC2007'
    create_csv_for_dataset(duc2007_path, 'duc2007.csv', 'DUC2007')

if __name__ == '__main__':
    main()