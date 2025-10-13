from summarizer import Summarizer
from load_data import load_duc2006_data
from rouge_metric import PyRouge

def bertsum_summarize(dataset):
    model = Summarizer('distilbert-base-uncased')
    summaries = []
    for idx, sample in enumerate(dataset):
        print(f"[Bertsum] Memproses sample ke-{idx+1} dari {len(dataset)}...")
        article = sample['article']
        summary = model(article, ratio=0.3)  # Extract 30% of sentences
        summaries.append(summary)
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
    bertsum_summaries = bertsum_summarize(dataset)
    references = [sample['references'] for sample in dataset]

    rouge_scores = evaluate_rouge(bertsum_summaries, references)

    for i, score in enumerate(rouge_scores):
        print(f"Sample {i+1}:")
        print(f"Bertsum Summary: {bertsum_summaries[i]}")
        print(f"ROUGE-1 F1: {score['rouge-1']['f']:.4f}")
        print(f"ROUGE-2 F1: {score['rouge-2']['f']:.4f}")
        print(f"ROUGE-SU4 F1: {score['rouge-su4']['f']:.4f}")
        print("-" * 50)