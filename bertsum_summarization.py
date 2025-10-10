from summarizer import Summarizer
from load_data import load_duc2006_data
from rouge_score import rouge_scorer

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
    bertsum_summaries = bertsum_summarize(dataset)
    references = [sample['references'] for sample in dataset]

    rouge_scores = evaluate_rouge(bertsum_summaries, references)

    for i, score in enumerate(rouge_scores):
        print(f"Sample {i+1}:")
        print(f"Bertsum Summary: {bertsum_summaries[i]}")
        print(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {score['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: {score['rougeLsum'].fmeasure:.4f}")
        print("-" * 50)