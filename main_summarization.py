import os
from load_data import load_multiple_datasets
from bertsum_summarization import bertsum_summarize, evaluate_rouge as evaluate_rouge_bertsum
from pegasus_summarization import pegasus_summarize, evaluate_rouge as evaluate_rouge_pegasus

def save_results(dataset_name, model_name, summaries, scores, avg_scores):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{dataset_name}_{model_name}_summaries.txt', 'w') as f:
        for i, summary in enumerate(summaries):
            f.write(f"Sample {i+1}:\n{summary}\n\n")
    
    with open(f'results/{dataset_name}_{model_name}_scores.txt', 'w') as f:
        f.write(f"Average ROUGE-1: {avg_scores['rouge1']:.4f}\n")
        f.write(f"Average ROUGE-2: {avg_scores['rouge2']:.4f}\n")
        f.write(f"Average ROUGE-L: {avg_scores['rougeLsum']:.4f}\n\n")
        for i, score in enumerate(scores):
            f.write(f"Sample {i+1}:\n")
            f.write(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}\n")
            f.write(f"ROUGE-2: {score['rouge2'].fmeasure:.4f}\n")
            f.write(f"ROUGE-L: {score['rougeLsum'].fmeasure:.4f}\n\n")

if __name__ == "__main__":
    all_datasets = load_multiple_datasets()
    print("Loaded datasets:", list(all_datasets.keys()))
    
    overall_bertsum_scores = []
    overall_pegasus_scores = []
    
    for dataset_name, dataset in all_datasets.items():
        print(f"\n=== Processing {dataset_name} ===")
        print(f"Dataset has {len(dataset)} samples")
        
        references = [sample['summary'] for sample in dataset]
        
        # Bertsum
        print(f"\n--- Bertsum (Extractive) for {dataset_name} ---")
        bertsum_summaries = bertsum_summarize(dataset)
        bertsum_scores = evaluate_rouge_bertsum(bertsum_summaries, references)
        
        bertsum_avg = {
            'rouge1': sum(s['rouge1'].fmeasure for s in bertsum_scores) / len(bertsum_scores),
            'rouge2': sum(s['rouge2'].fmeasure for s in bertsum_scores) / len(bertsum_scores),
            'rougeLsum': sum(s['rougeLsum'].fmeasure for s in bertsum_scores) / len(bertsum_scores)
        }
        
        print(f"Average ROUGE-1: {bertsum_avg['rouge1']:.4f}")
        print(f"Average ROUGE-2: {bertsum_avg['rouge2']:.4f}")
        print(f"Average ROUGE-L: {bertsum_avg['rougeLsum']:.4f}")
        
        save_results(dataset_name, 'bertsum', bertsum_summaries, bertsum_scores, bertsum_avg)
        overall_bertsum_scores.extend(bertsum_scores)
        
        # Pegasus
        print(f"\n--- Pegasus (Abstractive) for {dataset_name} ---")
        pegasus_summaries = pegasus_summarize(dataset)
        pegasus_scores = evaluate_rouge_pegasus(pegasus_summaries, references)
        
        pegasus_avg = {
            'rouge1': sum(s['rouge1'].fmeasure for s in pegasus_scores) / len(pegasus_scores),
            'rouge2': sum(s['rouge2'].fmeasure for s in pegasus_scores) / len(pegasus_scores),
            'rougeLsum': sum(s['rougeLsum'].fmeasure for s in pegasus_scores) / len(pegasus_scores)
        }
        
        print(f"Average ROUGE-1: {pegasus_avg['rouge1']:.4f}")
        print(f"Average ROUGE-2: {pegasus_avg['rouge2']:.4f}")
        print(f"Average ROUGE-L: {pegasus_avg['rougeLsum']:.4f}")
        
        save_results(dataset_name, 'pegasus', pegasus_summaries, pegasus_scores, pegasus_avg)
        overall_pegasus_scores.extend(pegasus_scores)
    
    # Overall averages
    print("\n=== Overall Summary ===")
    overall_bertsum_avg = {
        'rouge1': sum(s['rouge1'].fmeasure for s in overall_bertsum_scores) / len(overall_bertsum_scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in overall_bertsum_scores) / len(overall_bertsum_scores),
        'rougeLsum': sum(s['rougeLsum'].fmeasure for s in overall_bertsum_scores) / len(overall_bertsum_scores)
    }
    
    overall_pegasus_avg = {
        'rouge1': sum(s['rouge1'].fmeasure for s in overall_pegasus_scores) / len(overall_pegasus_scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in overall_pegasus_scores) / len(overall_pegasus_scores),
        'rougeLsum': sum(s['rougeLsum'].fmeasure for s in overall_pegasus_scores) / len(overall_pegasus_scores)
    }
    
    print("Overall Bertsum average ROUGE-1:", overall_bertsum_avg['rouge1'])
    print("Overall Bertsum average ROUGE-2:", overall_bertsum_avg['rouge2'])
    print("Overall Bertsum average ROUGE-L:", overall_bertsum_avg['rougeLsum'])
    print("Overall Pegasus average ROUGE-1:", overall_pegasus_avg['rouge1'])
    print("Overall Pegasus average ROUGE-2:", overall_pegasus_avg['rouge2'])
    print("Overall Pegasus average ROUGE-L:", overall_pegasus_avg['rougeLsum'])
    
    # Save overall results
    with open('results/overall_summary.txt', 'w') as f:
        f.write("Overall Bertsum average ROUGE-1: {:.4f}\n".format(overall_bertsum_avg['rouge1']))
        f.write("Overall Bertsum average ROUGE-2: {:.4f}\n".format(overall_bertsum_avg['rouge2']))
        f.write("Overall Bertsum average ROUGE-L: {:.4f}\n".format(overall_bertsum_avg['rougeLsum']))
        f.write("Overall Pegasus average ROUGE-1: {:.4f}\n".format(overall_pegasus_avg['rouge1']))
        f.write("Overall Pegasus average ROUGE-2: {:.4f}\n".format(overall_pegasus_avg['rouge2']))
        f.write("Overall Pegasus average ROUGE-L: {:.4f}\n".format(overall_pegasus_avg['rougeLsum']))