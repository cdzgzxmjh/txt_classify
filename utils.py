# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jieba
import numpy as np


def read_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            word = line.strip("\n")
            vocab[word] = idx

    return vocab


def convert_example(example, vocab, stop_words, is_test=False, is_api=False):
    """
    Builds model inputs from a sequence for sequence classification tasks. 
    It use `jieba.cut` to tokenize text.

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        valid_length(obj:`int`): The input sequence valid length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    if is_api:
        text = example
    elif is_test:
        text, = example
    else:
        text, label = example

    input_ids = []
    for word in jieba.cut(text):
        if word in vocab and word not in stop_words:
            word_id = vocab[word]
            input_ids.append(word_id)
        elif word in vocab and word in stop_words:
            continue
        elif word not in vocab:
            word_id = vocab["[UNK]"]
            input_ids.append(word_id)

    # assert len(input_ids) != 0
    valid_length = np.array(len(input_ids), dtype='int64')
    input_ids = np.array(input_ids, dtype='int64')

    if not is_test:
        label = np.array(label, dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def preprocess_prediction_data(data, tokenizer):
    """
    It process the prediction data as the format used as training.

    Args:
        data (obj:`List[str]`): The prediction data whose each element is  a tokenized text.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.

    Returns:
        examples (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).

    """
    examples = []
    for text in data:
        ids = tokenizer.encode(text)
        examples.append([ids, len(ids)])
    return examples


def write_results(labels, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.writelines("\n".join(labels))