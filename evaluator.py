from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

def run_evaluation(dataset):
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy]
    )

    print(result)