def build_test_cases() -> list[dict]:
    return [
        {
            "question": "What standard applies to fee-shifting under 28 U.S.C. § 1927?",
            "ground_truth": "Fee-shifting under Section 1927 requires a finding of bad faith, "
                             "assessed under a subjective standard, where an attorney's conduct "
                             "unreasonably and vexatiously multiplied proceedings.",
        },
        {
            "question": "What is the legal standard for arbitration clause disputes discussed in this case?",
            "ground_truth": "The court addressed whether the arbitration clause was sufficiently "
                             "explicit and whether the dispute could be compelled to arbitration.",
        },
    ]