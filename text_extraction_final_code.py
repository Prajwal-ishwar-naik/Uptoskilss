def evaluate_assignment(text: str) -> int:
    score = 0
    word_count = len(text.split())
    if word_count > 300:
        score += 3
    elif word_count > 150:
        score += 2

    sentence_count = len(text.split('.'))
    if sentence_count > 20:
        score += 2
    elif sentence_count > 10:
        score += 1

    keywords = ["introduction", "conclusion", "result", "analysis", "method"]
    keyword_hits = sum(1 for k in keywords if k in text.lower())
    score += min(keyword_hits, 3)

    if re.search(r"\b(fig|table|reference)\b", text.lower()):
        score += 1

    return min(score, 10)

# TEST CODE

if __name__ == "__main__":
    sample_text = """
    Introduction: This assignment explains the method and analysis of the results.
    The conclusion summarizes the findings. See table 1 and figure 2 for details.
    """
    marks = evaluate_assignment(sample_text)
    print("Marks:", marks)
