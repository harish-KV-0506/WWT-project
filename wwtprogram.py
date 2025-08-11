# ========================
# 1. IMPORTS & SETUP
# ========================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
import itertools

# ========================
# 2. LOAD DATA
# ========================
order_df = pd.read_csv("order_data.csv")
customer_df = pd.read_csv("customer_data.csv")
store_df = pd.read_csv("store_data.csv")
test_df = pd.read_csv("test_data_question.csv")

# ========================
# 3. PREPROCESS DATA
# ========================
# Extract items list from ORDERS column (assuming comma-separated items)
def extract_items(order_str):
    if pd.isna(order_str): return []
    return [item.strip() for item in str(order_str).split(",")]

order_df["item_list"] = order_df["ORDERS"].apply(extract_items)
test_df["item_list"] = test_df[["item1", "item2", "item3"]].values.tolist()

# ========================
# 4. ITEM CO-OCCURRENCE MATRIX
# ========================
item_counts = Counter()
pair_counts = Counter()

for items in order_df["item_list"]:
    unique_items = list(set(items))
    item_counts.update(unique_items)
    for comb in itertools.combinations(unique_items, 2):
        pair_counts[tuple(sorted(comb))] += 1

# Build co-occurrence dict
co_occurrence = defaultdict(dict)
for (i1, i2), count in pair_counts.items():
    co_occurrence[i1][i2] = count
    co_occurrence[i2][i1] = count

# ========================
# 5. RECOMMENDATION FUNCTION
# ========================
def recommend_items(cart_items, top_n=3):
    scores = Counter()

    # Use co-occurrence
    for item in cart_items:
        for related_item, count in co_occurrence.get(item, {}).items():
            if related_item not in cart_items:
                scores[related_item] += count

    # Add popularity bias
    for item in scores:
        scores[item] += 0.1 * item_counts[item]

    return [item for item, _ in scores.most_common(top_n)]

# ========================
# 6. GENERATE PREDICTIONS FOR TEST SET
# ========================
recommendations = []
for _, row in test_df.iterrows():
    cart_items = [x for x in row["item_list"] if pd.notna(x)]
    recs = recommend_items(cart_items, top_n=3)
    while len(recs) < 3:  # Fill if not enough recs
        recs.append(item_counts.most_common(1)[0][0])
    recommendations.append(recs)

test_df[["RECOMMENDATION_1", "RECOMMENDATION_2", "RECOMMENDATION_3"]] = pd.DataFrame(recommendations)

# ========================
# 7. SAVE OUTPUT FILE
# ========================
test_df.to_csv("TeamName_Recommendation_Output_Sheet.csv", index=False)
print("✅ Recommendation file saved!")