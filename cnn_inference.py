import os
import datetime
import pandas as pd
from bertsum_summarization import bertsum_summarize, evaluate_rouge as evaluate_rouge_bertsum
from pegasus_summarization import pegasus_summarize, evaluate_rouge as evaluate_rouge_pegasus
from datasets import Dataset

def load_cnn_dailymail(file_path, n_samples=None):
    """
    Load CNN/DailyMail dataset from CSV file
    
    Args:
        file_path: Path to CSV file (test.csv, train.csv, or validation.csv)
        n_samples: Number of samples to load (None = all)
    
    Returns:
        Dataset object compatible with summarization functions
    """
    print(f"\n📂 Loading CNN/DailyMail dataset from: {file_path}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    print(f"✅ Total samples in file: {len(df)}")
    
    # Limit samples if specified
    if n_samples is not None and n_samples > 0:
        df = df.head(n_samples)
        print(f"🎯 Selected samples: {len(df)}")
    
    # Convert to dataset format
    data = []
    for idx, row in df.iterrows():
        data.append({
            'id': row['id'],
            'article': row['article'],
            'references': [row['highlights']]  # Wrap in list for compatibility
        })
    
    dataset = Dataset.from_list(data)
    return dataset

def print_sample_info(dataset, sample_idx=0):
    """Print information about a sample"""
    sample = dataset[sample_idx]
    print(f"\n{'='*80}")
    print(f"📄 SAMPLE #{sample_idx + 1} INFO")
    print(f"{'='*80}")
    print(f"ID: {sample['id']}")
    print(f"\n📰 ARTICLE (first 500 chars):")
    print(f"{sample['article'][:500]}...")
    print(f"\n📊 Article length: {len(sample['article'])} characters")
    print(f"\n✨ REFERENCE SUMMARY:")
    print(f"{sample['references'][0]}")
    print(f"{'='*80}\n")

def save_results_cnn(dataset, model_name, summaries, scores, avg_scores, result_dir):
    """Save CNN/DailyMail inference results"""
    os.makedirs(result_dir, exist_ok=True)
    
    # Save summaries with article info
    summary_file = os.path.join(result_dir, f'cnn_{model_name}_summaries.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"CNN/DailyMail Inference Results - {model_name.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Samples: {len(summaries)}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, summary in enumerate(summaries):
            sample = dataset[i]
            f.write(f"{'='*80}\n")
            f.write(f"SAMPLE #{i+1}\n")
            f.write(f"{'='*80}\n")
            f.write(f"ID: {sample['id']}\n\n")
            f.write(f"ORIGINAL ARTICLE:\n")
            f.write(f"{sample['article']}\n\n")
            f.write(f"{'-'*80}\n")
            f.write(f"REFERENCE SUMMARY (Ground Truth):\n")
            f.write(f"{sample['references'][0]}\n\n")
            f.write(f"{'-'*80}\n")
            f.write(f"GENERATED SUMMARY ({model_name.upper()}):\n")
            f.write(f"{summary}\n\n")
            if i < len(scores):
                f.write(f"ROUGE SCORES:\n")
                f.write(f"  ROUGE-1: {scores[i]['rouge1'].fmeasure:.4f}\n")
                f.write(f"  ROUGE-2: {scores[i]['rouge2'].fmeasure:.4f}\n")
                f.write(f"  ROUGE-L: {scores[i]['rougeLsum'].fmeasure:.4f}\n")
            f.write(f"\n\n")
    
    # Save scores summary
    scores_file = os.path.join(result_dir, f'cnn_{model_name}_scores.txt')
    with open(scores_file, 'w', encoding='utf-8') as f:
        f.write(f"CNN/DailyMail ROUGE Scores - {model_name.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Samples: {len(scores)}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"AVERAGE SCORES:\n")
        f.write(f"  ROUGE-1: {avg_scores['rouge1']:.4f}\n")
        f.write(f"  ROUGE-2: {avg_scores['rouge2']:.4f}\n")
        f.write(f"  ROUGE-L: {avg_scores['rougeLsum']:.4f}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write(f"INDIVIDUAL SCORES:\n\n")
        for i, score in enumerate(scores):
            f.write(f"Sample #{i+1}:\n")
            f.write(f"  ROUGE-1: {score['rouge1'].fmeasure:.4f}\n")
            f.write(f"  ROUGE-2: {score['rouge2'].fmeasure:.4f}\n")
            f.write(f"  ROUGE-L: {score['rougeLsum'].fmeasure:.4f}\n")
            f.write(f"\n")
    
    print(f"✅ Results saved to: {result_dir}")
    print(f"   - Summaries: {summary_file}")
    print(f"   - Scores: {scores_file}")

def main():
    print("\n" + "="*80)
    print("🚀 CNN/DAILYMAIL TEXT SUMMARIZATION INFERENCE")
    print("="*80)
    
    # Select dataset file
    print("\n📋 SELECT DATASET FILE:")
    print("1. test.csv (11,490 samples) - Recommended for evaluation")
    print("2. validation.csv (13,368 samples)")
    print("3. train.csv (287,113 samples) - Very large, not recommended")
    
    dataset_choice = input("\nPilih dataset (1/2/3) [default: 1]: ").strip()
    
    dataset_files = {
        '1': 'test.csv',
        '2': 'validation.csv',
        '3': 'train.csv',
        '': 'test.csv'  # default
    }
    
    dataset_file = dataset_files.get(dataset_choice, 'test.csv')
    file_path = f'/workspaces/summary/cnn_dailymail/{dataset_file}'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return
    
    # Select number of samples
    print(f"\n🎯 PILIH JUMLAH SAMPEL:")
    print("Berapa artikel yang ingin di-inference?")
    print("  - 1 sample: ~1-2 menit (testing cepat)")
    print("  - 5 samples: ~5-10 menit")
    print("  - 10 samples: ~10-20 menit")
    print("  - 50+ samples: lebih lama (tergantung resource)")
    
    try:
        n_samples = int(input("\nMasukkan jumlah sampel [default: 1]: ").strip() or "1")
        if n_samples < 1:
            n_samples = 1
    except ValueError:
        print("⚠️  Input tidak valid, menggunakan default: 1 sample")
        n_samples = 1
    
    # Load dataset
    dataset = load_cnn_dailymail(file_path, n_samples)
    
    # Show sample info
    if len(dataset) > 0:
        print_sample_info(dataset, 0)
    
    # Select model
    print(f"\n🤖 PILIH MODEL SUMMARIZATION:")
    print("1. BertSum (Extractive) - Cepat, pilih kalimat dari artikel")
    print("2. Pegasus (Abstractive) - Lambat, generate kalimat baru")
    print("3. Keduanya (Perbandingan) - Paling lambat tapi lengkap")
    
    model_choice = input("\nPilih model (1/2/3) [default: 3]: ").strip()
    
    # Create result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('results_cnn', timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n💾 Results akan disimpan di: {result_dir}\n")
    
    # Prepare references
    references = [sample['references'] for sample in dataset]
    
    # Run inference based on choice
    if model_choice in ['1', '3', '']:
        print(f"\n{'='*80}")
        print(f"🔵 RUNNING BERTSUM (EXTRACTIVE) SUMMARIZATION")
        print(f"{'='*80}")
        
        bertsum_summaries = bertsum_summarize(dataset)
        bertsum_scores = evaluate_rouge_bertsum(bertsum_summaries, references)
        
        bertsum_avg = {
            'rouge1': sum(s['rouge1'].fmeasure for s in bertsum_scores) / len(bertsum_scores),
            'rouge2': sum(s['rouge2'].fmeasure for s in bertsum_scores) / len(bertsum_scores),
            'rougeLsum': sum(s['rougeLsum'].fmeasure for s in bertsum_scores) / len(bertsum_scores)
        }
        
        print(f"\n📊 BERTSUM AVERAGE SCORES:")
        print(f"   ROUGE-1: {bertsum_avg['rouge1']:.4f}")
        print(f"   ROUGE-2: {bertsum_avg['rouge2']:.4f}")
        print(f"   ROUGE-L: {bertsum_avg['rougeLsum']:.4f}")
        
        save_results_cnn(dataset, 'bertsum', bertsum_summaries, bertsum_scores, bertsum_avg, result_dir)
    
    if model_choice in ['2', '3', '']:
        print(f"\n{'='*80}")
        print(f"🟢 RUNNING PEGASUS (ABSTRACTIVE) SUMMARIZATION")
        print(f"{'='*80}")
        
        pegasus_summaries = pegasus_summarize(dataset)
        pegasus_scores = evaluate_rouge_pegasus(pegasus_summaries, references)
        
        pegasus_avg = {
            'rouge1': sum(s['rouge1'].fmeasure for s in pegasus_scores) / len(pegasus_scores),
            'rouge2': sum(s['rouge2'].fmeasure for s in pegasus_scores) / len(pegasus_scores),
            'rougeLsum': sum(s['rougeLsum'].fmeasure for s in pegasus_scores) / len(pegasus_scores)
        }
        
        print(f"\n📊 PEGASUS AVERAGE SCORES:")
        print(f"   ROUGE-1: {pegasus_avg['rouge1']:.4f}")
        print(f"   ROUGE-2: {pegasus_avg['rouge2']:.4f}")
        print(f"   ROUGE-L: {pegasus_avg['rougeLsum']:.4f}")
        
        save_results_cnn(dataset, 'pegasus', pegasus_summaries, pegasus_scores, pegasus_avg, result_dir)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ INFERENCE SELESAI!")
    print(f"{'='*80}")
    print(f"📁 Hasil tersimpan di: {result_dir}")
    print(f"📝 Total samples diproses: {len(dataset)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
