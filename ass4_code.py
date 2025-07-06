from .adapters import *
import pathlib
import os
import random
import torch
import numpy as np
import mmap
from statistics import mean
from collections import Counter
from urllib.parse import urlparse
from warcio.archiveiterator import ArchiveIterator
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "cs336_data"
MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "cs336_model"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_text(file_name: str | os.PathLike) -> list[str]:
    texts = []
    with open(file_name, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                texts.append(run_extract_text_from_html_bytes(record.content_stream().read()))
    return texts

def extract_url(file_name: str | os.PathLike) -> list[str]:
    urls = []
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("WARC-Target-URI:"):
                url = line.split(":", 1)[1].strip()
                urls.append(url)
    return urls

def construct_dataset(data_list, data_set, category):
    for line in data_list:
        try:
            line = line.strip()
            data_set.append("__label__" + str(category) + " " + line)
        except Exception as e:
            print(line)
            continue

HIGH_QUALITY_DOMAINS_TO_SKIP = {
    # 主流新闻
    'nytimes.com', 'bbc.com', 'theguardian.com', 'reuters.com', 'wsj.com',
    'washingtonpost.com', 'cnn.com', 'apnews.com',
    # 学术与百科
    'wikipedia.org', 'arxiv.org', 'nature.com', 'sciencemag.org',
    'acm.org', 'ieee.org',
    # 高质量技术博客/社区
    'stackoverflow.com', 'medium.com', 'github.com',
}

CITATION_PATTERN = re.compile(r'\([^)]*\b(18|19|20)\d{2}\b[^)]*\)')

def extract_eng_text_from_wet(file_path: str, domains_to_skip: set) -> list[str]:
    extracted_texts = []
    with open(file_path, 'rb') as stream:
        # 使用 ArchiveIterator 迭代文件中的每条 WARC 记录
        for record in ArchiveIterator(stream):
            # 条件 1: 记录类型必须是 'conversion'
            if record.rec_type == 'conversion':
                # 获取记录的头部信息
                headers = record.rec_headers
                # 条件 2: 语言必须是 'eng'
                language = headers.get('WARC-Identified-Content-Language')
                # 条件 3: 内容类型必须是 'text/plain'
                content_type = headers.get('Content-Type')
                if language == 'eng' and content_type == 'text/plain':
                    target_uri = headers.get('WARC-Target-URI')
                    if not target_uri:
                        continue
                    try:
                        domain = urlparse(target_uri).netloc
                        # print(target_uri)
                        # print(domain)
                        # 检查域名是否需要被跳过
                        should_skip = False
                        for hq_domain in domains_to_skip:
                            if hq_domain in domain:
                                should_skip = True
                                break
                        # 如果域名在跳过列表中，则进入下一次循环
                        if should_skip:
                            # print(target_uri)
                            continue
                    except Exception as e:
                        continue
                    content_bytes = record.content_stream().read()
                    # 使用 'utf-8' 解码，忽略可能出现的错误
                    text = content_bytes.decode('utf-8', errors='ignore')
                    if CITATION_PATTERN.search(text):
                        continue
                    clean_text = text.replace('\n', ' ').strip()
                    if clean_text:
                        extracted_texts.append(clean_text)
    print(f"处理完成。共提取到 {len(extracted_texts)} 段文本。")
    return extracted_texts

# 正则表达式：匹配 403, 404, 被封爬虫提示等
ERROR_PATTERNS = [
    re.compile(r'403\s+ERROR', re.IGNORECASE),
    re.compile(r'HTTP\s+Error\s+404', re.IGNORECASE),
    re.compile(r'404\s+Not\s+Found', re.IGNORECASE),
    re.compile(r'blocked\s+as\s+a\s+crawler\s+bot', re.IGNORECASE),
]

def clean_and_filter_lines(lines: list[str]) -> list[str]:
    cleaned_lines = []
    for line in lines:
        line = line.replace("Skip to content", " ")
        line = line.replace("Skip to main content", " ")
        line = line.replace("\n", " ").strip()
        if len(line) < 100:
            continue
        if any(p.search(line) for p in ERROR_PATTERNS):
            continue
        cleaned_lines.append(line)
    return cleaned_lines

def filter_by_non_alpha_ratio(texts):
    ratios = []
    index_ratio = {}
    for idx, line in enumerate(texts):
        tmp = line.strip()
        total_len = len(tmp)
        alpha_len = len(re.findall(r'[a-zA-Z]', tmp))
        non_alpha = total_len - alpha_len

        if total_len > 200:
            ratio = non_alpha / total_len
            ratios.append(ratio)
            index_ratio[idx] = ratio
    if not ratios:
        return []
    ratio_mean = mean(ratios)
    filtered_texts = [texts[i] for i, v in index_ratio.items() if v < ratio_mean]
    return filtered_texts

if __name__ == '__main__':
    # set_seed(1337)
    warc_path = DATA_PATH / "CC-MAIN-20250417135010-20250417165010-00065.warc"
    wet_path = DATA_PATH / "CC-MAIN-20250417135010-20250417165010-00065.warc.wet"
    enwiki_path = DATA_PATH / "enwiki-20240420-extracted_urls.txt"
    common_crawl_text = extract_eng_text_from_wet(wet_path, HIGH_QUALITY_DOMAINS_TO_SKIP)

    # with open(enwiki_path, 'rb') as f:
    #     mmapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    #     enwiki_urls = mmapped.read().decode('utf-8').splitlines()
    #     mmapped.close()
    # enwiki_sample = random.sample(enwiki_urls, 20000)
    # with open(DATA_PATH / "positive_urls.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(enwiki_sample))

    enwiki_warc = DATA_PATH / "positive_urls.warc"
    enwiki_text = extract_text(enwiki_warc)
    enwiki_en = []
    for i in range(len(enwiki_text)):
        language, p = run_identify_language(enwiki_text[i])
        if language == 'en':
            enwiki_en.append(enwiki_text[i])
    enwiki_clean = clean_and_filter_lines(enwiki_en)
    alpha_enwiki = filter_by_non_alpha_ratio(enwiki_clean)
    with open(DATA_PATH / "alpha_enwiki.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(alpha_enwiki) + "\n")

    with open(DATA_PATH / "alpha_enwiki.txt", "r", encoding="utf-8") as f:
        alpha_enwiki = f.read().splitlines()
    enwiki_ = random.sample(alpha_enwiki,  min(7500, len(alpha_enwiki)))
    common_crawl_ = random.sample(common_crawl_text, min(7500, len(enwiki_)))
    print(len(common_crawl_))
    all_dataset = []
    construct_dataset(enwiki_, all_dataset, "wiki")
    construct_dataset(common_crawl_, all_dataset, "cc")
    random.shuffle(all_dataset)
    split_index = int(len(all_dataset)*0.8)
    train_data = all_dataset[:split_index]
    val_data = all_dataset[split_index:]
    with open(DATA_PATH / "train_data.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
    with open(DATA_PATH / "val_data.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))
    
    model = fasttext.train_supervised(input=str(DATA_PATH / "train_data.txt"), lr=0.5, dim=100, word_ngrams=2, epoch=10, loss='softmax')
    model.save_model(str(MODEL_PATH / "quality_classifier.bin"))
    val_result = model.test(str(DATA_PATH / "val_data.txt"))
    print("Validation result:")
    print(f"Examples: {val_result[0]}")
    print(f"Precision: {val_result[1]:.4f}")
    print(f"Recall: {val_result[2]:.4f}")
    # 从 warc 文件的 html 代码中提取文本
    # text = extract_text(warc_path)
    # 随机抽取 20 个例子
    # sampled = random.sample(text, 20)
    # 识别这 20 个例子的语言
    # en_cnt = 0
    # for i in range(len(sampled)):
    #     language, p = run_identify_language(sampled[i])
    #     if language == 'en':
    #         en_cnt += 1
    # for i in range(len(text)):
        # flag = run_gopher_quality_filter(sampled[i])
        # print(sampled[i])
        # print(f"************ This example {flag} pass gopher quality filter ************")
        # flag, p = run_classify_nsfw(text[i])
        # flag1, p1 = run_classify_toxic_speech(text[i])
        # if flag == 'nsfw':
        #     print(f"************ NSFW category: {flag}, prob: {p} ************")
        #     print(f"************ Toxic speech: {flag1}, prob: {p1} ************")
        #     print(flag, flag1)
        #     break
        # print("*************** before mask ***************")
        # print(sampled[i])
        # # 先屏蔽字符串的邮箱
        # mask_email, email_cnt = run_mask_emails(sampled[i])
        # # 用屏蔽邮箱后的字符串屏蔽电话号码
        # mask_phone, phone_cnt = run_mask_phone_numbers(mask_email)
        # # 最后屏蔽 IP地址
        # mask_IP, IP_cnt = run_mask_ips(mask_phone)
        # print("*************** after mask ***************")
        # print(f"mask email count:{email_cnt}, mask phone count:{phone_cnt}, mask IP count:{IP_cnt}")
        # print(mask_IP)
        