"""
Generative AI Example 2: Evaluating an LLM with BLEU and ROUGE
Demonstrates an end-to-end pipeline for evaluating LLM-generated text
against reference answers using BLEU and ROUGE scores.

The `MockLLM` class simulates an LLM. Replace it with a real LLM client
(e.g., OpenAI, HuggingFace, Ollama) to evaluate actual model outputs.
"""

import sys
sys.path.insert(0, 'src')

from data_quality_checker import DataQualityChecker, QualityMetricsAggregator


# =============================================================================
# Mock LLM — replace this with a real LLM client
# =============================================================================

class MockLLM:
    """
    Simulates an LLM by returning pre-defined responses.

    To use a real LLM, replace the `generate` method. Examples:

    OpenAI:
        import openai
        client = openai.OpenAI(api_key="...")
        def generate(self, prompt):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

    HuggingFace Transformers:
        from transformers import pipeline
        pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
        def generate(self, prompt):
            return pipe(prompt, max_new_tokens=100)[0]["generated_text"]

    Ollama (local):
        import ollama
        def generate(self, prompt):
            return ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
                              ["message"]["content"]
    """

    # Simulated LLM responses for each prompt
    _responses = {
        "What is machine learning?":
            "Machine learning is a branch of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed.",

        "Explain neural networks in one sentence.":
            "Neural networks are computational models inspired by the human brain, "
            "consisting of interconnected layers of nodes that process and learn from data.",

        "What is natural language processing?":
            "Natural language processing is a field of AI that focuses on enabling computers "
            "to understand, interpret, and generate human language.",

        "Define supervised learning.":
            "Supervised learning is a machine learning approach where a model is trained on "
            "labeled data, learning to map inputs to outputs based on example input-output pairs.",

        "What is a transformer model?":
            "A transformer is a deep learning architecture that uses self-attention mechanisms "
            "to process sequential data, forming the basis of modern large language models.",
    }

    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        return self._responses.get(prompt, "I don't have a response for that prompt.")


# =============================================================================
# Evaluation dataset: prompts + reference (ground truth) answers
# =============================================================================

EVALUATION_DATASET = [
    {
        "prompt": "What is machine learning?",
        "reference": "Machine learning is a type of artificial intelligence that allows "
                     "computers to learn from data and improve their performance over time "
                     "without being explicitly programmed for each task."
    },
    {
        "prompt": "Explain neural networks in one sentence.",
        "reference": "Neural networks are a series of algorithms that mimic the operations "
                     "of a human brain to recognize relationships between vast amounts of data."
    },
    {
        "prompt": "What is natural language processing?",
        "reference": "Natural language processing is a subfield of linguistics and AI concerned "
                     "with the interactions between computers and human language."
    },
    {
        "prompt": "Define supervised learning.",
        "reference": "Supervised learning is a machine learning paradigm where the algorithm "
                     "learns from a training dataset that includes both input data and the "
                     "correct output labels."
    },
    {
        "prompt": "What is a transformer model?",
        "reference": "A transformer model is a neural network architecture introduced in the "
                     "paper 'Attention Is All You Need', which relies on self-attention to "
                     "model relationships in sequential data."
    },
]


# =============================================================================
# Evaluation pipeline
# =============================================================================

def run_llm_evaluation():
    llm = MockLLM()
    checker = DataQualityChecker()
    aggregator = QualityMetricsAggregator()

    print("=" * 65)
    print("LLM EVALUATION WITH BLEU AND ROUGE")
    print("=" * 65)
    print(f"\nEvaluating {len(EVALUATION_DATASET)} prompts...\n")

    all_rouge_scores = []
    all_bleu_scores = []
    results_table = []

    # Step 1: Generate LLM responses and score each one
    for i, item in enumerate(EVALUATION_DATASET, 1):
        prompt = item["prompt"]
        reference = item["reference"]

        # Generate response from LLM
        prediction = llm.generate(prompt)

        # Compute BLEU and ROUGE
        rouge = checker.compute_rouge(prediction, reference)
        bleu = checker.compute_bleu([prediction], [[reference]])

        all_rouge_scores.append(rouge)
        all_bleu_scores.append(bleu)

        results_table.append({
            "id": i,
            "prompt": prompt,
            "prediction": prediction,
            "reference": reference,
            "rouge1": rouge.get("rouge1", 0.0),
            "rouge2": rouge.get("rouge2", 0.0),
            "rougeL": rouge.get("rougeL", 0.0),
            "bleu": bleu.get("bleu", 0.0),
        })

    # Step 2: Print per-sample results
    print("-" * 65)
    print(f"{'#':<4} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'BLEU':<10} Prompt")
    print("-" * 65)
    for r in results_table:
        prompt_short = r["prompt"][:35] + "..." if len(r["prompt"]) > 35 else r["prompt"]
        print(f"{r['id']:<4} {r['rouge1']:<10.4f} {r['rouge2']:<10.4f} {r['rougeL']:<10.4f} {r['bleu']:<10.4f} {prompt_short}")

    # Step 3: Print aggregated scores
    print("\n" + "=" * 65)
    print("AGGREGATED SCORES ACROSS ALL PROMPTS")
    print("=" * 65)

    agg_rouge = aggregator.aggregate_rouge_scores(all_rouge_scores, aggregation_type='mean')
    agg_bleu = aggregator.aggregate_bleu_scores(all_bleu_scores, aggregation_type='mean')

    print(f"\n  {'Metric':<20} {'Mean Score'}")
    print(f"  {'-'*35}")
    print(f"  {'ROUGE-1':<20} {agg_rouge.get('rouge1', 0):.4f}")
    print(f"  {'ROUGE-2':<20} {agg_rouge.get('rouge2', 0):.4f}")
    print(f"  {'ROUGE-L':<20} {agg_rouge.get('rougeL', 0):.4f}")
    print(f"  {'BLEU':<20} {agg_bleu.get('bleu', 0):.4f}")

    # Step 4: Identify best and worst performing prompts
    print("\n" + "=" * 65)
    print("BEST AND WORST PERFORMING RESPONSES (by ROUGE-1)")
    print("=" * 65)

    sorted_by_rouge1 = sorted(results_table, key=lambda x: x["rouge1"], reverse=True)

    print(f"\n  Best:  [{sorted_by_rouge1[0]['rouge1']:.4f}] {sorted_by_rouge1[0]['prompt']}")
    print(f"  Worst: [{sorted_by_rouge1[-1]['rouge1']:.4f}] {sorted_by_rouge1[-1]['prompt']}")

    # Step 5: Detailed view of best response
    best = sorted_by_rouge1[0]
    print(f"\n  --- Best Response Detail ---")
    print(f"  Prompt:     {best['prompt']}")
    print(f"  Prediction: {best['prediction']}")
    print(f"  Reference:  {best['reference']}")
    print(f"  ROUGE-1: {best['rouge1']:.4f}  ROUGE-2: {best['rouge2']:.4f}  "
          f"ROUGE-L: {best['rougeL']:.4f}  BLEU: {best['bleu']:.4f}")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)


def main():
    run_llm_evaluation()


if __name__ == "__main__":
    main()

# Made with Bob
