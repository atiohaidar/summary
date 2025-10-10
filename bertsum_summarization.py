from summarizer import Summarizer
from load_data import load_duc2006_data
from rouge_score import rouge_scorer

def bertsum_summarize(dataset):
    model = Summarizer('distilbert-base-uncased')
    summaries = []
    for sample in dataset:
        article = sample['article']
        summary = model(article, ratio=0.3)  # Extract 30% of sentences
        summaries.append(summary)
    return summaries

def evaluate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score)
    return scores

if __name__ == "__main__":
    dataset = load_duc2006_data()
    bertsum_summaries = bertsum_summarize(dataset)
    references = [sample['summary'] for sample in dataset]

    rouge_scores = evaluate_rouge(bertsum_summaries, references)

    for i, score in enumerate(rouge_scores):
        print(f"Sample {i+1}:")
        print(f"Bertsum Summary: {bertsum_summaries[i]}")
        print(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {score['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: {score['rougeLsum'].fmeasure:.4f}")
        print("-" * 50)