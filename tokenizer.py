import pandas as pd
import jieba
import jieba.posseg as psg
import os
import swifter
import re
import unicodedata


# 換繁體字庫
zh_dict_file = os.path.join(os.path.dirname(__file__), "jieba", "zh", "dict.txt")
jieba.load_userdict(zh_dict_file)


INPUT_FILE = "csv/cleaned/sample_cleaned_articles.csv" # all_cleaned_articles
OUTPUT_FILE = "csv/tokenized/sample_tokenized_articles.csv" # tokenized_articles

# --- 1. 定義詞性過濾列表 ---
STOP_POS = ["p", "c", "u", "d", "e", "w", "r", "y", "m", "q", "o", "t"]
STOP_WORDS = set(
    [
        "了",
        "嗎",
        "吧",
        "啦",
        "的",
        "是",
        "嗎",
        "過",
        "不",
        "都",
        "在",
        "跟",
        "就",
        "有",
        "和",
        "也",
        "但",
        "還",
        "很",
        "這",
        "那",
        "他",
        "她",
        "你",
        "我",
    ]
)


def tokenize_and_filter(text):

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    # 這裡使用正則表達式來替換半形/全形標點、並包含您之前看到的問號
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~，。？！……]+', " ", text)

    if pd.isna(text):  # 處理空值
        return ""

    words_list = []

    for word, flag in psg.cut(text):
        word = word.strip()

        is_stop_pos = flag in STOP_POS
        is_stop_word = word in STOP_WORDS

        if len(word) > 0 and not is_stop_pos and not is_stop_word:
            words_list.append(word)

    # 將結果以空格分隔的字串形式返回
    return " ".join(words_list)


# ----------------------------------------------------
# 1. 載入 Clean Data from CSV file
# ----------------------------------------------------
try:
    print(f"--- 1. 載入數據: {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE)

    # 確認檔案結構
    if "Title" not in df.columns or "Board" not in df.columns:
        print("錯誤：CSV 檔案中缺少 'Title' 或 'Board' 欄位。請檢查檔案。")
        exit()

except FileNotFoundError:
    print(f"錯誤：找不到檔案 {INPUT_FILE}。請檢查路徑是否正確。")
    exit()
except Exception as e:
    print(f"載入檔案時發生錯誤: {e}")
    exit()

# ----------------------------------------------------
# 2. 轉換 each article title into word tokens (包含詞性過濾)
# ----------------------------------------------------
print("--- 2. 開始進行分詞和詞性過濾 (Tokenization) ---")

df["Title_Tokens"] = df["Title"].swifter.apply(tokenize_and_filter)


# ----------------------------------------------------
# 3. 儲存 tokenized data into a single csv file
# ----------------------------------------------------
print(f"--- 3. 儲存數據到: {OUTPUT_FILE} ---")

# 選擇需要儲存的欄位，通常是原始 Title, Board, 和新的 Title_Tokens
output_df = df[["Board", "Title_Tokens"]]

# 儲存為 CSV 檔案，不包含索引 (index=False)
output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("--- 處理完成 ---")
print(f"最終儲存的檔案位於：{os.path.abspath(OUTPUT_FILE)}")
