"""
RAG Evaluation Script
Evaluates the RAG pipeline using custom metrics.
Usage: python eval/evaluate.py
"""

import asyncio
import json
from pathlib import Path
from loguru import logger

# Add parent dir to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_pipeline import process_query
from app.models.schemas import ChatRequest, QueryType


# Sample test cases - add your own domain-specific ones
TEST_CASES = [
    {
        "query": "What is machine learning?",
        "expected_type": QueryType.WEB_SEARCH,
        "expected_keywords": ["algorithm", "data", "learn", "model", "pattern"],
    },
]


async def evaluate_single(test_case: dict) -> dict:
    """Evaluate a single test case."""
    request = ChatRequest(
        query=test_case["query"],
        query_type=test_case.get("expected_type"),
    )

    try:
        response = await process_query(request)

        # Check if expected keywords appear in the answer
        answer_lower = response.answer.lower()
        keywords_found = sum(1 for kw in test_case["expected_keywords"] if kw in answer_lower)
        keyword_coverage = keywords_found / len(test_case["expected_keywords"]) if test_case["expected_keywords"] else 1.0

        # Check if query was routed correctly
        type_correct = response.query_type == test_case["expected_type"] if test_case.get("expected_type") else True

        # Check if answer is non-empty and reasonable length
        has_answer = len(response.answer) > 20
        reasonable_length = 20 < len(response.answer) < 5000

        return {
            "query": test_case["query"],
            "answer": response.answer[:200] + "..." if len(response.answer) > 200 else response.answer,
            "query_type": response.query_type.value,
            "type_correct": type_correct,
            "keyword_coverage": keyword_coverage,
            "has_answer": has_answer,
            "reasonable_length": reasonable_length,
            "num_sources": len(response.sources),
            "num_web_sources": len(response.web_sources),
            "passed": type_correct and keyword_coverage >= 0.4 and has_answer,
        }

    except Exception as e:
        return {
            "query": test_case["query"],
            "error": str(e),
            "passed": False,
        }


async def run_evaluation():
    """Run all test cases and report results."""
    logger.info(f"Running evaluation with {len(TEST_CASES)} test cases")

    results = []
    for i, tc in enumerate(TEST_CASES):
        logger.info(f"[{i + 1}/{len(TEST_CASES)}] Evaluating: {tc['query'][:60]}...")
        result = await evaluate_single(tc)
        results.append(result)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_keyword_coverage = sum(r.get("keyword_coverage", 0) for r in results) / total if total else 0

    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed / total * 100:.1f}%" if total else "N/A",
        "avg_keyword_coverage": f"{avg_keyword_coverage * 100:.1f}%",
        "results": results,
    }

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total:     {total}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {total - passed}")
    print(f"Pass Rate: {summary['pass_rate']}")
    print(f"Avg Keyword Coverage: {summary['avg_keyword_coverage']}")
    print(f"\nDetailed results saved to: {output_path}")
    print("=" * 60)

    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"\n{status} {r['query'][:60]}")
        if "error" in r:
            print(f"   Error: {r['error']}")
        else:
            print(f"   Type: {r['query_type']} | Keywords: {r.get('keyword_coverage', 0) * 100:.0f}%")
            print(f"   Answer: {r.get('answer', 'N/A')[:100]}...")

    return summary


if __name__ == "__main__":
    asyncio.run(run_evaluation())
