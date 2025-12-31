import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

# 設定 logging 格式
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# 1. 讀取 CSV
print("--- 正在讀取資料 ---")
# df = pd.read_csv("csv/tokenized/sample_tokenized_articles.csv")
df = pd.read_csv("csv/tokenized/tokenized_articles.csv")

# 隨機打亂
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# 2. 將字串轉換為 list
def prepare_corpus(df):
    for i, row in df.iterrows():
        tokens = str(
            row["Title_Tokens"]
        ).split()  # "助理 費 不夠".split() -> ['助理', '費', '不夠']
        yield TaggedDocument(tokens, [i])  # 使用 index 作為標籤 (Tags)


print("--- 正在準備語料庫 ---")
train_corpus = list(prepare_corpus(df))

# 3. 初始化並訓練模型
model = Doc2Vec(vector_size=150, dm=0, dm_mean=1, min_count=5, epochs=50, workers=8)

print("--- 正在建立詞彙表 (Build Vocab) ---")
model.build_vocab(train_corpus)

print("--- 開始訓練模型 (請查看下方 INFO 日誌) ---")
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# 4. 驗證
# 存檔
print("--- 訓練完成，正在存檔 ---")
model.save("models/doc2vec.model")

# 同時儲存這份「打亂後」的資料，供 evaluate.py 使用
df.to_csv("csv/tokenized/shuffled_articles.csv", index=False)

print("\n" + "=" * 30)
print("模型驗證結果：")

# 注意：Doc2Vec 同時學了詞向量與文件向量
try:
    print("與 '預算' 最接近的詞:", model.wv.most_similar("預算"))
except KeyError:
    print("詞頻太低，不在詞典中")

# 驗證文件向量 (找跟第 1 篇文章最接近的文章)
sim_docs = model.dv.most_similar(0, topn=3)
print(f"原始標題: {df.iloc[0]['Title_Tokens']}")
print("最相似的文章 ID 與分數:", sim_docs)
print("=" * 30)
