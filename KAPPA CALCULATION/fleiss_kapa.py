import pandas as pd
import numpy as np

def fleiss_kappa_from_excel(
    file_path,
    rater_columns,
    final_label_column=None,   # optional, not required for kappa
    sheet_name=0
):
    """
    file_path: path to Excel file
    rater_columns: list of column names for LLM outputs (e.g., ["llm1","llm2","llm3"])
    final_label_column: optional column name (not used in kappa, just for reference)
    sheet_name: Excel sheet index or name
    """

    # Load data
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Keep only relevant columns
    data = df[rater_columns].copy()

    # Get all unique categories (moods)
    categories = sorted(pd.unique(data.values.ravel()))

    # Build count matrix (N items × k categories)
    count_matrix = []
    for _, row in data.iterrows():
        counts = [sum(row == cat) for cat in categories]
        count_matrix.append(counts)

    M = np.array(count_matrix)  # shape: (N, k)
    N, k = M.shape
    n = np.sum(M[0])  # number of raters (should be 3)

    # Step 1: Compute P_i (agreement per item)
    P_i = (np.sum(M * (M - 1), axis=1)) / (n * (n - 1))

    # Step 2: Mean agreement
    P_bar = np.mean(P_i)

    # Step 3: Category proportions
    p_j = np.sum(M, axis=0) / (N * n)

    # Step 4: Expected agreement
    P_e = np.sum(p_j ** 2)

    # Step 5: Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e)

    return {
        "kappa": kappa,
        "P_bar": P_bar,
        "P_e": P_e,
        "categories": categories
    }


# ====== HOW TO USE ======

result = fleiss_kappa_from_excel(
    file_path="3 moods with final mood_edited.xlsx",
    rater_columns=["ChatGPT Mood", "Claude Mood", "deepseek/copilot Mood"],   # <-- change to your column names
    final_label_column="FINAL MOOD"                # optional
)

print("Fleiss' Kappa:", result["kappa"])
print("Mean Agreement (P_bar):", result["P_bar"])
print("Expected Agreement (P_e):", result["P_e"])
print("Categories:", result["categories"])