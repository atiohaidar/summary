from transformers import pipeline
from load_data import load_duc2006_data
from rouge_score import rouge_scorer
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
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = []
    for pred, references in zip(predictions, reference_lists):
        if not references:
            continue
        best_score = None
        for ref in references:
            score = scorer.score(ref, pred)
            if best_score is None:
                best_score = score
            else:
                for metric in score:
                    if score[metric].fmeasure > best_score[metric].fmeasure:
                        best_score[metric] = score[metric]
        if best_score is not None:
            scores.append(best_score)
    return scores

if __name__ == "__main__":
    dataset = load_duc2006_data()
    pegasus_summaries = pegasus_summarize(dataset)
    references = [sample['references'] for sample in dataset]

    rouge_scores = evaluate_rouge(pegasus_summaries, references)

    for i, score in enumerate(rouge_scores):
        print(f"Sample {i+1}:")
        print(f"Pegasus Summary: {pegasus_summaries[i]}")
        print(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {score['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: {score['rougeLsum'].fmeasure:.4f}")
        print("-" * 50)