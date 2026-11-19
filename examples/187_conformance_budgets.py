"""Lesson 294: assess measured GPT errors against explicit numerical budgets."""

from rebuild_gpt.conformance_budgets import check_conformance_budgets


if __name__ == "__main__":
    result = check_conformance_budgets(
        "out/release_review/extended_conformance.json",
    )
    print("Budget status:", result["status"])
    print("Measurements checked:", result["checked_measurements"])
