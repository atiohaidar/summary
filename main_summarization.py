import os
import datetime
from load_data import load_multiple_datasets
from bertsum_summarization import bertsum_summarize, evaluate_rouge as evaluate_rouge_bertsum
from pegasus_summarization import pegasus_summarize, evaluate_rouge as evaluate_rouge_pegasus

def save_results(dataset_name, model_name, summaries, scores, avg_scores, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{dataset_name}_{model_name}_summaries.txt'), 'w') as f:
        for i, summary in enumerate(summaries):
            f.write(f"Sample {i+1}:\n{summary}\n\n")
    with open(os.path.join(result_dir, f'{dataset_name}_{model_name}_scores.txt'), 'w') as f:
        f.write(f"Average ROUGE-1 F1: {avg_scores['rouge-1']:.4f}\n")
        f.write(f"Average ROUGE-2 F1: {avg_scores['rouge-2']:.4f}\n")
        f.write(f"Average ROUGE-SU4 F1: {avg_scores['rouge-su4']:.4f}\n\n")
        for i, score in enumerate(scores):
            f.write(f"Sample {i+1}:\n")
            f.write(f"ROUGE-1 F1: {score['rouge-1']['f']:.4f}\n")
            f.write(f"ROUGE-2 F1: {score['rouge-2']['f']:.4f}\n")
            f.write(f"ROUGE-SU4 F1: {score['rouge-su4']['f']:.4f}\n\n")

if __name__ == "__main__":
    all_datasets = load_multiple_datasets()
    print("Loaded datasets:", list(all_datasets.keys()))

    # Buat folder hasil dengan timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('results', timestamp)
    os.makedirs(result_dir, exist_ok=True)

    print("\nMenu Test Data:")
    print("1. Jalankan SEMUA data")
    print("2. Jalankan BEBERAPA data saja (misal 1 file)")
    menu = input("Pilih menu (1/2): ").strip()

    if menu == "2":
        try:
            n_samples = int(input("Masukkan jumlah data yang ingin diuji (misal 1): ").strip())
        except ValueError:
            print("Input tidak valid, default ke 1 data.")
            n_samples = 1
    else:
        n_samples = None  # None berarti semua data

    overall_bertsum_scores = []
    overall_pegasus_scores = []

    for dataset_name, dataset in all_datasets.items():
        print(f"\n=== Processing {dataset_name} ===")
        print(f"Dataset has {len(dataset)} samples")

        # Pilih subset jika user memilih hanya beberapa data
        if n_samples is not None:
            dataset = list(dataset)[:n_samples]
        else:
            dataset = list(dataset)

        if not dataset:
            print(f"Skipping {dataset_name}: dataset kosong")
            continue

        references = [sample['references'] for sample in dataset]

        # Bertsum
        print(f"\n--- Bertsum (Extractive) for {dataset_name} ---")
        bertsum_summaries = bertsum_summarize(dataset)
        bertsum_scores = evaluate_rouge_bertsum(bertsum_summaries, references)

        bertsum_avg = {
            'rouge-1': sum(s['rouge-1']['f'] for s in bertsum_scores) / len(bertsum_scores),
            'rouge-2': sum(s['rouge-2']['f'] for s in bertsum_scores) / len(bertsum_scores),
            'rouge-su4': sum(s['rouge-su4']['f'] for s in bertsum_scores) / len(bertsum_scores)
        }

        print(f"Average ROUGE-1 F1: {bertsum_avg['rouge-1']:.4f}")
        print(f"Average ROUGE-2 F1: {bertsum_avg['rouge-2']:.4f}")
        print(f"Average ROUGE-SU4 F1: {bertsum_avg['rouge-su4']:.4f}")

        save_results(dataset_name, 'bertsum', bertsum_summaries, bertsum_scores, bertsum_avg, result_dir)
        overall_bertsum_scores.extend(bertsum_scores)

        # Pegasus
        print(f"\n--- Pegasus (Abstractive) for {dataset_name} ---")
        pegasus_summaries = pegasus_summarize(dataset)
        pegasus_scores = evaluate_rouge_pegasus(pegasus_summaries, references)

        pegasus_avg = {
            'rouge-1': sum(s['rouge-1']['f'] for s in pegasus_scores) / len(pegasus_scores),
            'rouge-2': sum(s['rouge-2']['f'] for s in pegasus_scores) / len(pegasus_scores),
            'rouge-su4': sum(s['rouge-su4']['f'] for s in pegasus_scores) / len(pegasus_scores)
        }

        print(f"Average ROUGE-1 F1: {pegasus_avg['rouge-1']:.4f}")
        print(f"Average ROUGE-2 F1: {pegasus_avg['rouge-2']:.4f}")
        print(f"Average ROUGE-SU4 F1: {pegasus_avg['rouge-su4']:.4f}")

        save_results(dataset_name, 'pegasus', pegasus_summaries, pegasus_scores, pegasus_avg, result_dir)
        overall_pegasus_scores.extend(pegasus_scores)
    
    # Overall averages
    print("\n=== Overall Summary ===")
    if len(overall_bertsum_scores) > 0:
        overall_bertsum_avg = {
            'rouge-1': sum(s['rouge-1']['f'] for s in overall_bertsum_scores) / len(overall_bertsum_scores),
            'rouge-2': sum(s['rouge-2']['f'] for s in overall_bertsum_scores) / len(overall_bertsum_scores),
            'rouge-su4': sum(s['rouge-su4']['f'] for s in overall_bertsum_scores) / len(overall_bertsum_scores)
        }
        print("Overall Bertsum average ROUGE-1 F1:", overall_bertsum_avg['rouge-1'])
        print("Overall Bertsum average ROUGE-2 F1:", overall_bertsum_avg['rouge-2'])
        print("Overall Bertsum average ROUGE-SU4 F1:", overall_bertsum_avg['rouge-su4'])
    else:
        overall_bertsum_avg = None
        print("Tidak ada skor Bertsum yang dihitung.")

    if len(overall_pegasus_scores) > 0:
        overall_pegasus_avg = {
            'rouge-1': sum(s['rouge-1']['f'] for s in overall_pegasus_scores) / len(overall_pegasus_scores),
            'rouge-2': sum(s['rouge-2']['f'] for s in overall_pegasus_scores) / len(overall_pegasus_scores),
            'rouge-su4': sum(s['rouge-su4']['f'] for s in overall_pegasus_scores) / len(overall_pegasus_scores)
        }
        print("Overall Pegasus average ROUGE-1 F1:", overall_pegasus_avg['rouge-1'])
        print("Overall Pegasus average ROUGE-2 F1:", overall_pegasus_avg['rouge-2'])
        print("Overall Pegasus average ROUGE-SU4 F1:", overall_pegasus_avg['rouge-su4'])
    else:
        overall_pegasus_avg = None
        print("Tidak ada skor Pegasus yang dihitung.")

    # Save overall results
    with open(os.path.join(result_dir, 'overall_summary.txt'), 'w') as f:
        if overall_bertsum_avg:
            f.write("Overall Bertsum average ROUGE-1 F1: {:.4f}\n".format(overall_bertsum_avg['rouge-1']))
            f.write("Overall Bertsum average ROUGE-2 F1: {:.4f}\n".format(overall_bertsum_avg['rouge-2']))
            f.write("Overall Bertsum average ROUGE-SU4 F1: {:.4f}\n".format(overall_bertsum_avg['rouge-su4']))
        else:
            f.write("Tidak ada skor Bertsum yang dihitung.\n")
        if overall_pegasus_avg:
            f.write("Overall Pegasus average ROUGE-1 F1: {:.4f}\n".format(overall_pegasus_avg['rouge-1']))
            f.write("Overall Pegasus average ROUGE-2 F1: {:.4f}\n".format(overall_pegasus_avg['rouge-2']))
            f.write("Overall Pegasus average ROUGE-SU4 F1: {:.4f}\n".format(overall_pegasus_avg['rouge-su4']))
        else:
            f.write("Tidak ada skor Pegasus yang dihitung.\n")