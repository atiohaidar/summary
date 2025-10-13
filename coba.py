from rouge_metric import PyRouge

# Example reference and candidate summaries
reference = "The cat was found under the bed."
candidate = "The cat was under the bed."

# Initialize PyRouge with ROUGE SU4 metric
# ROUGE SU4: Skip-bigram with unigram co-occurrence, max skip gap of 4
rouge = PyRouge(rouge_su=True, skip_gap=4)

# Evaluate
hypotheses = [candidate]
references = [[reference]]

scores = rouge.evaluate(hypotheses, references)

print("ROUGE SU4 Scores:")
print(scores)
