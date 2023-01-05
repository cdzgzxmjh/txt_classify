import os
import time
from collections import Counter
from itertools import chain

import jieba


def sort_and_write_words(all_words, file_path):
    words = list(chain(*all_words))
    words_vocab = Counter(words).most_common()
    with open(file_path, "w", encoding="utf8") as f:
        f.write('[UNK]\n[PAD]\n')
        # filter the count of words below 5
        # 过滤低频词，词频<5
        for word, num in words_vocab:
            if num < 5:
                continue
            f.write(word + "\n")


(root, directory, files), = list(os.walk("./work/data"))
all_words = []
for file_name in files:
    with open(os.path.join(root, file_name), "r", encoding="utf8") as f:
        for line in f:
            if file_name in ["train.txt", "dev.txt"]:
                text, label = line.strip().split("\t")
            elif file_name == "test.txt":
                text = line.strip()
            else:
                continue
            words = jieba.lcut(text)
            words = [word for word in words if word.strip() !='']
            all_words.append(words)

# 写入词表
sort_and_write_words(all_words, "work/data/vocab.txt")
