import pandas as pd
from sklearn.metrics import cohen_kappa_score

def pairwise_kappa_from_excel(
    file_path,
    rater_columns,
    sheet_name=0,
    clean=True
):
    """
    file_path: Excel file path
    rater_columns: list like ["LLM1", "LLM2", "LLM3"]
    """

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data = df[rater_columns].copy()

    # Optional cleaning
    if clean:
        data = data.apply(lambda col: col.astype(str).str.strip().str.lower())

    results = {}

    for i in range(len(rater_columns)):
        for j in range(i + 1, len(rater_columns)):
            col1 = rater_columns[i]
            col2 = rater_columns[j]

            kappa = cohen_kappa_score(data[col1], data[col2])

            results[f"{col1} vs {col2}"] = kappa

    return results


# ===== USAGE =====
result = pairwise_kappa_from_excel(
    file_path="3 moods with final mood.xlsx",
    rater_columns=["ChatGPT Mood", "Claude Mood", "deepseek/copilot Mood"]
)

for pair, score in result.items():
    print(f"{pair}: {score}")