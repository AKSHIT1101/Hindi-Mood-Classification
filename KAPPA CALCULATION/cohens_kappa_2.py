import pandas as pd
from sklearn.metrics import cohen_kappa_score

def pairwise_kappa_row_range(
    file_path,
    rater_columns,
    start_row=None,   # inclusive (0-based index)
    end_row=None,     # exclusive
    sheet_name=0
):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Select only required columns
    data = df[rater_columns].copy()

    # Clean text
    data = data.apply(lambda col: col.astype(str).str.strip().str.lower())

    # ---- ROW RANGE FILTER ----
    if start_row is not None or end_row is not None:
        data = data.iloc[start_row:end_row]

    # ---- KAPPA ----
    results = {}

    for i in range(len(rater_columns)):
        for j in range(i + 1, len(rater_columns)):
            col1 = rater_columns[i]
            col2 = rater_columns[j]

            kappa = cohen_kappa_score(data[col1], data[col2])
            results[f"{col1} vs {col2}"] = kappa

    print(f"Rows used: {len(data)}")
    return results


# ===== USAGE =====

result = pairwise_kappa_row_range(
    file_path="3 moods with final mood_edited.xlsx",
    rater_columns=["ChatGPT Mood", "Claude Mood", "deepseek/copilot Mood"],
    start_row=0,     # start row (0 = first data row)
    end_row=1608      # up to but not including row 500
)

for pair, score in result.items():
    print(f"{pair}: {score}")