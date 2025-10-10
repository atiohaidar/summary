from transformers import pipeline
from load_data import load_duc2006_data
from rouge_score import rouge_scorer

def pegasus_summarize(dataset):
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    summaries = []
    for sample in dataset:
        article = sample['article'][:1024]  # Limit length for Pegasus
        summary = summarizer(article, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
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
    pegasus_summaries = pegasus_summarize(dataset)
    references = [sample['summary'] for sample in dataset]

    rouge_scores = evaluate_rouge(pegasus_summaries, references)

    for i, score in enumerate(rouge_scores):
        print(f"Sample {i+1}:")
        print(f"Pegasus Summary: {pegasus_summaries[i]}")
        print(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {score['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: {score['rougeLsum'].fmeasure:.4f}")
        print("-" * 50)