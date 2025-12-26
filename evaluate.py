import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import logging

# 選擇性：如果你想看加載過程
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# 1. 設定檔案路徑
MODEL_PATH = "models/doc2vec.model"
DATA_PATH = "csv/tokenized/shuffled_articles.csv"


def evaluate_model():
    # 2. 載入模型與資料
    print("--- Loading Model ---")
    model = Doc2Vec.load(MODEL_PATH)

    print("--- Loading Data ---")
    df = pd.read_csv(DATA_PATH)

    # 3. 準備評估樣本 (至少 1,000 份文件)
    sample_size = 1000
    if len(df) < sample_size:
        sample_size = len(df)

    # 隨機挑選 1000 個索引
    eval_indices = np.random.choice(len(df), sample_size, replace=False)

    ranks = []
    similarities = []

    print(f"--- Evaluating {sample_size} documents ---")

    for count, doc_id in enumerate(eval_indices):
        # 取得該文本的 tokens (list)
        doc_words = str(df.iloc[doc_id]["Title_Tokens"]).split()

        # 核心測試：透過 infer_vector 取得新向量
        inferred_vector = model.infer_vector(doc_words, epochs=100)

        # 找出與此推論向量最接近的文檔
        # 返回 (ID, similarity) 的列表
        sims = model.dv.most_similar([inferred_vector], topn=10)

        # 檢查該文章自己有沒有在 Top 2 裡面
        # 找出原本的 doc_id 在 sims 列表中的排名
        doc_ids_in_sims = [item[0] for item in sims]

        try:
            rank = doc_ids_in_sims.index(doc_id)
        except ValueError:
            rank = 999  # 不在 Top 10 裡面

        ranks.append(rank)

        # 記錄第一名的相似度分數
        similarities.append(sims[0][1])

        if (count + 1) % 100 == 0:
            print(f"{count + 1}")

    # 4. 計算需求指標
    # 需求：Second Self-Similarity > 80% (這通常指 Top 2 命中率或平均分數)
    hit_top_2 = sum(1 for r in ranks if r <= 1)
    hit_rate_top_2 = (hit_top_2 / sample_size) * 100
    avg_self_sim = np.mean(similarities) * 100

    print("\n" + "=" * 40)
    print(f"Self Similarity: {avg_self_sim:.2f}%")
    print(f"Second Self Similarity: {hit_rate_top_2:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_model()
