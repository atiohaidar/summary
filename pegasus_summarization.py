from transformers import pipeline
from load_data import load_duc2006_data
from rouge_metric import PyRouge
import psutil

def pegasus_summarize(dataset):
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    summaries = []
    for idx, sample in enumerate(dataset):
        print(f"[Pegasus] Memproses sample ke-{idx+1} dari {len(dataset)}...")
        article = sample['article'][:1024]  # Limit length for Pegasus
        print(f"Article length (chars): {len(article)}")
        # Log CPU and RAM usage
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        print(f"[Resource] CPU Usage: {cpu_percent}% | RAM Usage: {ram.percent}%")
        summary = summarizer(article, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        print(f"Generated summary length (chars): {len(summary)}")
        summaries.append(summary)
        print(f"Generated Summary: {summary}\n")
    return summaries

def evaluate_rouge(predictions, reference_lists):
    # Initialize PyRouge for ROUGE1, ROUGE2, and ROUGE SU4
    rouge = PyRouge(rouge_n=(1, 2), rouge_su=True, skip_gap=4, multi_ref_mode='best', alpha=0.5)
    scores = []
    for pred, references in zip(predictions, reference_lists):
        if not references:
            continue
        # Evaluate with multiple references
        hypotheses = [pred]
        multi_references = [references]
        score = rouge.evaluate(hypotheses, multi_references)
        scores.append(score)
    return scores

if __name__ == "__main__":
    dataset = load_duc2006_data()
    pegasus_summaries = pegasus_summarize(dataset)
    references = [sample['references'] for sample in dataset]

    rouge_scores = evaluate_rouge(pegasus_summaries, references)

    for i, score in enumerate(rouge_scores):
        print(f"Sample {i+1}:")
        print(f"Pegasus Summary: {pegasus_summaries[i]}")
        print(f"ROUGE-1 F1: {score['rouge-1']['f']:.4f}")
        print(f"ROUGE-2 F1: {score['rouge-2']['f']:.4f}")
        print(f"ROUGE-SU4 F1: {score['rouge-su4']['f']:.4f}")
        print("-" * 50)