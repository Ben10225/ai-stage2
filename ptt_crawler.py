import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import concurrent.futures
import os
import random
from requests.exceptions import RequestException


# 設置爬蟲的參數
MAX_RETRIES = 5
MAX_ARTICLES = 200000
PTT_URL = "https://www.ptt.cc"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
# PTT 部分看板（如Gossiping）有「滿18歲」驗證，需要傳遞一個 cookie
COOKIES = {"over18": "1"}


def fetch_page_with_retry(url):
    """嘗試多次連線，以應對暫時的 Connection Aborted 錯誤"""
    for attempt in range(MAX_RETRIES):
        try:
            # 設置超時時間，避免無限等待
            response = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=15)

            # 檢查 HTTP 狀態碼 (4xx 或 5xx 錯誤)
            response.raise_for_status()

            # 請求成功，返回回應物件
            return response

        except RequestException as e:
            # 捕捉所有的 requests 錯誤，包括 Connection Aborted, Reset, Timeout, HTTPError 等
            print(f"❌ 嘗試 {attempt + 1}/{MAX_RETRIES} 失敗: {url}。錯誤: {e}")

            if attempt < MAX_RETRIES - 1:
                # 執行重試前，使用遞增的隨機延遲 (Exponential Backoff with Jitter)
                # 這樣能讓請求看起來更分散，並給伺服器更多恢復時間。
                sleep_time = random.uniform(2**attempt, 2 ** (attempt + 1))
                # 為了避免太久，可以設定一個上限，例如 30 秒
                if sleep_time > 30:
                    sleep_time = random.uniform(20, 30)

                print(f"等待 {sleep_time:.2f} 秒後重試...")
                time.sleep(sleep_time)
            else:
                # 達到最大重試次數，拋出錯誤給上層函式處理
                print(f"🚨 達到最大重試次數，放棄頁面: {url}")
                raise

    return None  # 理論上不會執行到這裡


def get_articles_from_page(url):
    # 從單頁面抓取文章標題和上一頁的連結
    try:
        response = fetch_page_with_retry(url)

    except Exception as e:
        # 如果 fetch_page_with_retry 在達到最大次數後仍然失敗，會拋出異常
        print(f"致命錯誤，無法抓取頁面: {url}")
        return [], None

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    # 文章列表的區塊
    for div in soup.find_all("div", class_="r-ent"):
        # 抓取文章標題
        title_tag = div.find("div", class_="title").find("a")

        # 排除被刪除或無標題的文章 (標題tag為None)
        if title_tag:
            title = title_tag.text.strip()
            articles.append(title)

    # 尋找「上一頁」的連結 (在 PTT 網頁上是「上頁」)
    paging_div = soup.find("div", class_="btn-group btn-group-paging")
    prev_page_link = None
    if paging_div:
        prev_button = paging_div.find("a", string="‹ 上頁")

        if "href" in prev_button.attrs:
            prev_page_link = PTT_URL + prev_button["href"]

    time.sleep(random.uniform(0.5, 1.5))

    return articles, prev_page_link


def crawl_board(initial_url, board_name):
    """遞迴抓取多頁文章標題，直到達到最大數量或沒有上一頁"""
    print(f"--- 開始爬取看板: {board_name} ---")
    current_url = initial_url
    all_titles = []

    page_count = 0

    while current_url and len(all_titles) < MAX_ARTICLES:
        print(f"爬取頁面: {current_url}")

        articles_on_page, next_url = get_articles_from_page(current_url)
        page_count += 1

        # 增加看板標籤
        newly_fetched = 0
        for title in articles_on_page:
            if len(all_titles) < MAX_ARTICLES:
                all_titles.append((title, board_name))
                newly_fetched += 1
            else:
                break

        current_total = len(all_titles)
        print(
            f"[{board_name}] 頁面 {page_count} 爬取完成。本次新增: {newly_fetched} 篇。累積總數: {current_total} 篇 / 目標 {MAX_ARTICLES}"
        )

        # 如果達到上限，則停止
        if len(all_titles) >= MAX_ARTICLES:
            break

        current_url = next_url
        time.sleep(random.uniform(1.5, 2.5))

    # 儲存為 CSV
    filename = f"{board_name}.csv"
    df = pd.DataFrame(all_titles, columns=["Title", "Board"])

    BASE_DIR = "csv/raw"
    os.makedirs(BASE_DIR, exist_ok=True)
    full_path = os.path.join(BASE_DIR, filename)

    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(
        f"看板 {board_name} 爬取完成，共 {len(all_titles)} 篇文章，已保存至 {filename}"
    )
    print("-" * 30)


def run_concurrently(board_list):
    MAX_WORKERS = 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 submit 將每個 crawl_board 任務提交給執行緒池
        future_to_board = {
            executor.submit(crawl_board, url, name): name
            for name, url in board_list.items()
        }

        # 可選：等待所有任務完成並處理結果/錯誤
        for future in concurrent.futures.as_completed(future_to_board):
            board_name = future_to_board[future]
            try:
                # 獲取 crawl_board 的返回結果 (如果有的話)
                data = future.result()
                print(f"看板 {board_name} 已完成並行爬取。")
            except Exception as exc:
                print(f"看板 {board_name} 在爬取時發生錯誤: {exc}")


board_list = {
    "baseball": "https://www.ptt.cc/bbs/Baseball/index.html",
    "boy_girl": "https://www.ptt.cc/bbs/Boy-Girl/index.html",
    "c_chat": "https://www.ptt.cc/bbs/c_chat/index.html",
    "hate_politics": "https://www.ptt.cc/bbs/hatepolitics/index.html",
    "life_is_money": "https://www.ptt.cc/bbs/Lifeismoney/index.html",
    "military": "https://www.ptt.cc/bbs/Military/index.html",
    "pc_shopping": "https://www.ptt.cc/bbs/pc_shopping/index.html",
    "stock": "https://www.ptt.cc/bbs/stock/index.html",
    "tech_job": "https://www.ptt.cc/bbs/Tech_Job/index.html",
}

if __name__ == "__main__":
    run_concurrently(board_list)
