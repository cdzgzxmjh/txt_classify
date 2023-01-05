import paddle
import numpy as np

from functools import partial

import paddlenlp
from paddlenlp.datasets import MapDataset

from utils import convert_example, read_vocab, write_results
from paddlenlp.data import Stack, Pad, Tuple


class NewsData(paddle.io.Dataset):
    def __init__(self, data_path, mode="train"):
        is_test = True if mode == "test" else False
        self.label_map = { item:index for index, item in enumerate(self.label_list)}
        self.examples = self._read_file(data_path, is_test)

    def _read_file(self, data_path, is_test):
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if is_test:
                    text = line.strip()
                    examples.append((text,))
                else:
                    text, label = line.strip('\n').split('\t')
                    label = self.label_map[label]
                    examples.append((text, label))
        return examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    @property
    def label_list(self):
        return ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']


# Loads dataset.
train_ds = NewsData("work/data/train.txt", mode="train")
dev_ds = NewsData("work/data/dev.txt", mode="dev")
test_ds = NewsData("work/data/test.txt", mode="test")

print("Train data:")
for text, label in train_ds[:5]:
    print(f"Text: {text}; Label ID {label}")

print()
print("Test data:")
for text, in test_ds[:5]:
    print(f"Text: {text}")



def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      use_gpu=False,
                      batchify_fn=None):
    if trans_fn:
        dataset = MapDataset(dataset)
        dataset = dataset.map(trans_fn)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=batchify_fn)
    return dataloader

vocab = read_vocab("work/data/vocab.txt")
stop_words = read_vocab("work/data/stop_words.txt")

batch_size = 128
epochs = 2

trans_fn = partial(convert_example, vocab=vocab, stop_words=stop_words, is_test=False)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab.get('[PAD]', 0)),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
train_loader = create_dataloader(
    train_ds,
    trans_fn=trans_fn,
    batch_size=batch_size,
    mode='train',
    use_gpu=True,
    batchify_fn=batchify_fn)
dev_loader = create_dataloader(
    dev_ds,
    trans_fn=trans_fn,
    batch_size=batch_size,
    mode='validation',
    use_gpu=True,
    batchify_fn=batchify_fn)

import paddle.nn as nn
import paddle.nn.functional as F

class CNNModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)

        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        # self.lstm_encoder = paddlenlp.seq2vec.LSTMEncoder(
        #     emb_dim,
        #     lstm_hidden_size,
        #     num_layers=lstm_layers,
        #     direction=direction,
        #     dropout=dropout_rate,
        #     pooling_type=pooling_type)

        # CNN
        self.cnn_encoder = paddlenlp.seq2vec.CNNEncoder(
            emb_dim,
            num_filter=128
        )

        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        self.fc = nn.Linear(self.cnn_encoder.get_output_dim(), fc_hidden_size)

        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirectional' else 1
        text_repr = self.cnn_encoder(embedded_text)


        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))

        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


if __name__ == '__main__':
    model = CNNModel(
        len(vocab),
        len(train_ds.label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
    model = paddle.Model(model)

    print(type(model))

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=5e-4)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Starts training and evaluating.
    # model.fit(train_loader, dev_loader, epochs=epochs, save_dir='./ckpt')
    model.fit(train_loader, dev_loader, epochs=epochs, save_dir='./log_cnn')
