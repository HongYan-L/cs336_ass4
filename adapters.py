from __future__ import annotations

import os
import re
import sys
from typing import Any
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import pathlib
from pathlib import Path
import hashlib
import unicodedata
import math
import torch
import random
import numpy as np
MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "cs336_model"


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    bytes_type = detect_encoding(html_bytes)
    try:
        unicode_html = html_bytes.decode(bytes_type, errors='replace')
    except (UnicodeDecodeError, TypeError):
        unicode_html = html_bytes.decode('utf-8', errors='replace')
    text = extract_plain_text(unicode_html)
    return text


def run_identify_language(text: str) -> tuple[Any, float]:
    text = text.replace('\n', '')
    model = fasttext.load_model(str(MODEL_PATH / "lid.176.bin"))
    language, p = model.predict(text, k=1)
    return str(language[0]).split('__label__')[1], p[0]


def run_mask_emails(text: str) -> tuple[str, int]:
    mask_email_text, cnt = re.subn(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "|||EMAIL_ADDRESS|||", text)
    return mask_email_text, cnt


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    PHONE_REGEXES = {
        "zh": r"(\+?86)?1[3-9]\d{9}",                           # 中国大陆手机号
        "en": r"(?<!\d)(\s*)((\+?1\s*[-.\s]*)?(\(\d{2,4}\)|\d{2,4})[-\s.]?\d{3,4}[-\s.]?\d{4})(?!\d)",  # 美国/加拿大
        "be": r"(?<!\d)(\s*)((\+?32)?\s*\(?0?\)?\s*(\d{1,2}[-\s.]?\d{2,3}[-\s.]?\d{2,3}[-\s.]?\d{2,3}))(?!\d)",
        "ja": r"(\+?81)?0\d{1,4}[\s-]?\d{1,4}[\s-]?\d{4}",       # 日本
        "ko": r"(\+?82)?0[1-9]\d{1,3}[\s-]?\d{3,4}[\s-]?\d{4}",  # 韩国
        "fr": r"(\+?33)?\s?[67]\d{8}",                          # 法国
        "de": r"(\+?49)?1[5-7]\d{8}",                           # 德国
        "ru": r"(\+?7)?9\d{9}",                                 # 俄罗斯
        "it": r"(\+?39)?3\d{9}",                                # 意大利
        "id": r"(\+?62)?8\d{8,11}",                             # 印尼
        "tr": r"(\+?90)?5\d{9}",                                # 土耳其
        "th": r"(\+?66)?[689]\d{8}",                            # 泰国
        "vi": r"(\+?84)?(3|5|7|8|9)\d{8}",                      # 越南
        "pt": r"(\+?351)?9[1236]\d{7}",                         # 葡萄牙
        "es": r"(\+?34)?6\d{8}",                                # 西班牙
        "uk": r"(\+?380)?(39|50|63|66|67|68|73|91|92|93|94|95|96|97|98|99)\d{7}",  # 乌克兰
        "pl": r"(\+?48)?[5-8]\d{8}",                            # 波兰
        "ro": r"(\+?40)?7[2-8]\d{7}",                           # 罗马尼亚
        "bn": r"(\+?880)?1[3-9]\d{8}",                          # 孟加拉
        "hi": r"(\+?91)?[6789]\d{9}",                           # 印地语（印度）
        "ta": r"(\+?91)?[6789]\d{9}",                           # 泰米尔语（印度）
        "te": r"(\+?91)?[6789]\d{9}",                           # 泰卢固语（印度）
        "gu": r"(\+?91)?[6789]\d{9}",                           # 古吉拉特语（印度）
        "ur": r"(\+?92)?3\d{9}",                                # 乌尔都语（巴基斯坦）
        "fa": r"(\+?98)?9\d{9}",                                # 波斯语（伊朗）
        "he": r"(\+?972)?5\d{8}",                               # 希伯来语（以色列）
        "az": r"(\+?994)?(50|51|70|77)\d{7}",                   # 阿塞拜疆
        "ms": r"(\+?60)?1[0-9]\d{7,8}",                         # 马来语（马来西亚）
        "nl": r"(\+?31)?6\d{8}",                                # 荷兰语
        "sv": r"(\+?46)?7[0236]\d{7}",                          # 瑞典语（瑞典）
    }
    total_cnt = 0
    new_text = text
    for regexe in PHONE_REGEXES.values():
        new_text, cnt = re.subn(regexe, r"\1|||PHONE_NUMBER|||", new_text)
        total_cnt += cnt
    return new_text, total_cnt


def run_mask_ips(text: str) -> tuple[str, int]:
    mask_ip_text, cnt = re.subn(r"\b(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\b", "|||IP_ADDRESS|||", text)
    return mask_ip_text, cnt


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    text = text.replace('\n', '')
    model = fasttext.load_model(str(MODEL_PATH / "jigsaw_fasttext_bigrams_nsfw_final.bin"))
    flag, p = model.predict(text, k=1)
    return str(flag[0]).split('__label__')[1], p[0]


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    text = text.replace('\n', '')
    model = fasttext.load_model(str(MODEL_PATH / "jigsaw_fasttext_bigrams_hatespeech_final.bin"))
    flag, p = model.predict(text, k=1)
    return str(flag[0]).split('__label__')[1], p[0]


def run_classify_quality(text: str) -> tuple[Any, float]:
    text = text.replace('\n', '')
    model = fasttext.load_model(str(MODEL_PATH / "quality_classifier.bin"))
    flag, p = model.predict(text, k=1)
    return str(flag[0]).split('__label__')[1], p[0]


def run_gopher_quality_filter(text: str) -> bool:
    flag = True
    # 按空格切分文本
    words = text.split()
    new_words = []
    # 排除 words 里面非字母的字符
    for word in words:
        new_word = re.sub(r'[^a-zA-Z]', '', word)
        new_words.append(new_word)
    new_words = list(filter(lambda x:x != '', new_words))
    # 统计单词的数量
    word_cnt = len(new_words)
    # 如果单词数量小于 50 或者 大于 100,000，则文本不通过 Gopher 质量过滤器
    if word_cnt < 50 or word_cnt > 100000:
        flag = False
    # 如果单词数量占比少于 80%，则文本不通过 Gopher 质量过滤器
    if len(words) == 0:
        flag = False
    if len(words) > 0 and len(new_words) / len(words) < 0.8:
        flag = False
    # 计算单词的平均长度
    words_len = 0
    for it in new_words:
        words_len += len(it)
    avg_words_len = words_len / len(new_words)
    # 如果单词平均长度小于 3 或者 大于 10，则文本不通过 Gopher 质量过滤器
    if avg_words_len < 3 or avg_words_len > 10:
        flag = False
    lines = text.splitlines()
    if lines:
        end_with_dots = sum(1 for line in lines if line.strip().endswith("..."))
        # 如果超过 30% 的行以 ... 结尾，则文本不通过 Gopher 质量过滤器
        if end_with_dots / len(lines) > 0.3:
            flag = False
    return flag


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    line_cnt = {}
    # 第一次遍历，将目录下所有文件的行用 md5 做哈希映射并统计次数
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            file_lines = f.read().splitlines()
        for line in file_lines:
            line_hash = hashlib.md5(line.encode()).hexdigest()
            line_cnt[line_hash] = line_cnt.get(line_hash, 0) + 1
    output_directory.mkdir(parents=True, exist_ok=True)
    # 第二次遍历，对每一行查看它的 md5 哈希值对应的计数是否大于 1
    for input_file in input_files:
        output_file = []
        with open(input_file, "r", encoding="utf-8") as f:
            file_lines = f.read().splitlines()
        for line in file_lines:
            line_hash = hashlib.md5(line.encode()).hexdigest()
            if (line_cnt[line_hash] > 1):
                continue
            else:
                output_file.append(line)
        content = "\n".join(output_file).rstrip("\n")
        with open(str(output_directory / Path(input_file).name), "w", encoding="utf-8") as f:
            if content:
                f.write(content + "\n")
            else:
                f.write("")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def get_ngrams(text, ngrams):
    n_grams = set([])
    for i in range(len(text) - ngrams + 1):
        n_grams.add(text[i : i+ngrams])
    return n_grams

def get_minhash_sign(S, num_hashes, random_strings):
    signature = []
    for i in range(num_hashes):
        mini_hash = sys.maxsize
        for it in S:
            h_it = abs(hash(it + random_strings[i]))
            if h_it < mini_hash:
                mini_hash = h_it
        signature.append(mini_hash)
    return signature

class UnionFind:
    def __init__(self, elements=None):
        self.parent = {}
        self.rank = {}
        if elements:
            for x in elements:
                self.parent[x] = x
                self.rank[x] = 0

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def groups(self):
        result = {}
        for node in self.parent:
            root = self.find(node)
            if root not in result:
                result[root] = set()
            result[root].add(node)
        return result

def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    assert num_hashes % num_bands == 0
    set_seed(1337)
    # 生成 num_hashes 个哈希函数的 signature
    random_strings = [str(random.random()) for _ in range(num_hashes)]
    file_sign = {}
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        # applying NFD unicode normalization and removing accents
        nfd_text = unicodedata.normalize('NFD', text)
        no_accents_text = ''.join([char for char in nfd_text if unicodedata.category(char) != 'Mn'])
        # lowercasing
        text_lower = no_accents_text.lower()
        # removing punctuation
        text_nopun = re.sub(r'[^\w\s]', ' ', text_lower, flags=re.UNICODE)
        # normalizing whitespaces
        text_nospace = re.sub(r'\s+', ' ', text_nopun).strip()

        # 将文件分解成 n-grams 集合 S
        S = get_ngrams(text_nospace, ngrams)
        
        minihash_sign = get_minhash_sign(S, num_hashes, random_strings)
        file_sign[str(input_file)] = minihash_sign

    # LSH
    r = int(num_hashes / num_bands)
    band_bucket = {i: {} for i in range(num_bands)}
    for file_name, signature in file_sign.items():
        for num_band in range(num_bands):
            left_idx = num_band * r
            right_idx = left_idx + r
            sign_vector = tuple(signature[left_idx : right_idx])
            sign_hash = hashlib.sha1(str(sign_vector).encode("utf-8")).hexdigest()
            if sign_hash not in band_bucket[num_band]:
                band_bucket[num_band][sign_hash] = []
            band_bucket[num_band][sign_hash].append(file_name)

    dup_doc = set()
    for num_band in range(num_bands):
        for sign_hash, file_list in band_bucket[num_band].items():
            if len(file_list) > 1:
                for i in range(len(file_list)):
                    for j in range(i + 1, len(file_list)):
                        pair = tuple(sorted((file_list[i], file_list[j])))
                        dup_doc.add(pair)

    # 使用所有文档的名称初始化并查集
    uf = UnionFind(list(file_sign.keys()))

    for pair in dup_doc:
        file_sign1 = file_sign[pair[0]]
        file_sign2 = file_sign[pair[1]]
        jaccard_sim = (sum(1 for x, y in zip(file_sign1, file_sign2) if x == y)) / len(file_sign1)
        # 如果两个文档的 Jaccard 相似度超过设定阈值
        if jaccard_sim > jaccard_threshold:
            # 则将这两个文档 union 成一个更大的簇
            uf.union(pair[0], pair[1])

    output_directory.mkdir(parents=True, exist_ok=True)
    # 遍历所有簇，因为 union 操作后，所有相似的文档都合并到一个簇，所有只需将所有的 keys 写入新的目录即可

    for k in uf.groups().keys():
        in_path = Path(k)
        out_path = str(output_directory / in_path.name)
        with open(in_path, "r", encoding="utf-8") as f_in:
            text = f_in.read()
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write(text)
