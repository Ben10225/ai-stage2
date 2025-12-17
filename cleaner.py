import pandas as pd
import os
import re

# --- 配置參數 ---
RAW_DATA_DIR = "csv/raw"
CLEANED_DATA_DIR = "csv/cleaned"
OUTPUT_FILENAME = "all_cleaned_articles.csv"
# 假設您的原始 CSV 檔案中，文章標題所在的欄位名稱為 'title'
TITLE_COLUMN = "Title"
BOARD_COLUMN = "Board"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    對單個 DataFrame 應用清洗規則。
    """

    initial_rows = len(df)
    print(f"  -> 初始行數: {initial_rows}")

    # --- 1. 定義清洗規則 ---

    # 1. 移除開頭和結尾的空白 (Remove leading and trailing spaces)
    df[TITLE_COLUMN] = df[TITLE_COLUMN].str.strip()

    # 2. 轉換字母為小寫 (Convert letters to lowercase)
    df[TITLE_COLUMN] = df[TITLE_COLUMN].str.lower()

    # 3. 移除所有以 "re:" 或 "fw:" 開頭的標題
    # 使用 .str.startswith() 結合布林索引 (Boolean Indexing)
    condition_re = df[TITLE_COLUMN].str.startswith("re:")
    condition_fw = df[TITLE_COLUMN].str.startswith("fw:")
    condition_gonggao = df[TITLE_COLUMN].str.startswith("[公告]")
    rows_to_remove = condition_re | condition_fw | condition_gonggao
    # 應用布林索引：保留不滿足移除條件的行
    df = df[~rows_to_remove].copy()

    # --- 2. 額外定義的清洗規則 (建議加入) ---

    # 4. 移除爬蟲階段可能產生的空值或 NaN 值
    df.dropna(subset=[TITLE_COLUMN], inplace=True)

    # 5. 移除重複資料 (基於標題和看板名稱)
    df.drop_duplicates(subset=[TITLE_COLUMN, BOARD_COLUMN], inplace=True)

    # 6. 移除 PTT 特殊符號 (例如 [新聞] 這種分類標籤)
    # 這個步驟是您在 "初步整理" 中想做的，這裡用正規表達式 (re) 實現更精確。
    # 匹配並移除開頭 [任何文字] 結構
    df[TITLE_COLUMN] = (
        df[TITLE_COLUMN]
        .apply(lambda x: re.sub(r"^\[.*?\]\s*", "", x) if pd.notna(x) else x)
        .str.strip()
    )

    # # 僅移除 PTT 常見分類標籤 (使用 OR 邏輯指定，忽略其他特殊標籤)
    # regex_to_remove = r'^\[(新聞|討論|閒聊|情報|心得|問卦|爆卦|發問)\]\s*'
    # df[TITLE_COLUMN] = df[TITLE_COLUMN].apply(
    #     lambda x: re.sub(regex_to_remove, '', x, flags=re.IGNORECASE) if pd.notna(x) else x
    # ).str.strip()

    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    print(f"  -> 清洗後行數: {final_rows}")
    print(f"  -> 移除總數: {removed_rows}")

    return df


def load_and_clean_all():
    """載入所有 raw 檔案，清洗並合併"""

    # 確保輸出資料夾存在
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

    all_cleaned_dfs = []

    # 遍歷 RAW_DATA_DIR 下的所有檔案
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, filename)

            print(f"處理檔案: {filename}...")

            try:
                # 載入 CSV 檔案 (使用 utf-8-sig 編碼以兼容 Windows 和 Excel)
                # 由於您的 raw 資料可能已經很大，建議使用較低記憶體的 dtype
                df_raw = pd.read_csv(
                    file_path, encoding="utf-8-sig", dtype={TITLE_COLUMN: str}
                )

                # 應用清洗規則
                df_cleaned = clean_data(df_raw)

                all_cleaned_dfs.append(df_cleaned)

            except Exception as e:
                print(f"❌ 處理 {filename} 發生錯誤: {e}")
                continue  # 繼續處理下一個檔案

    # --- 合併所有清洗後的 DataFrame ---
    if not all_cleaned_dfs:
        print("未找到任何 CSV 檔案或所有檔案處理失敗。")
        return

    print("\n--- 開始合併所有看板資料 ---")

    # 使用 pd.concat 合併列表中的所有 DataFrame
    df_final = pd.concat(all_cleaned_dfs, ignore_index=True)

    print(f"✨ 最終總文章數 (合併後): {len(df_final)}")

    # --- 儲存最終結果 ---
    output_path = os.path.join(CLEANED_DATA_DIR, OUTPUT_FILENAME)
    # 儲存到 cleaned 資料夾中
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 資料清洗與合併完成！結果已儲存至: {output_path}")


if __name__ == "__main__":
    load_and_clean_all()
