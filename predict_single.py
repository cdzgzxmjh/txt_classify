import numpy as np
import paddle

from functools import partial

from train import create_dataloader, NewsData, LSTMModel
from utils import convert_example, read_vocab, write_results
from paddlenlp.data import Stack, Pad, Tuple

vocab = read_vocab("work/data/vocab.txt")
stop_words = read_vocab("work/data/stop_words.txt")

label_list = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']


# test_ds = NewsData("work/data/test.txt", mode="test")

batch_size = 128
test_batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab.get('[PAD]', 0)),  # input_ids
    Stack(dtype="int64"),  # seq len
): [data for data in fn(samples)]


class ArrayData(paddle.io.Dataset):
    def __init__(self, examples):
        self.label_map = { item:index for index, item in enumerate(label_list)}
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


state_dict = paddle.load('./log/final.pdparams')
model = LSTMModel(len(vocab),
        len(label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
model.set_state_dict(state_dict)
model = paddle.Model(model)
model.prepare(loss=paddle.nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy())


# title = 新闻标题
def get_news_type(title):
    # 数据集：其中标签为空
    example = ArrayData([[title, ]])

    # 定义数据加载器
    test_loader = create_dataloader(
        example,
        trans_fn=partial(convert_example, vocab=vocab, stop_words=stop_words, is_test=True),
        batch_size=batch_size,
        mode='test',
        use_gpu=True,
        batchify_fn=test_batchify_fn)
    # 执行模型预测，模型已经预先加载
    results = model.predict(test_loader)
    inverse_label_map = {value: key for key, value in example.label_map.items()}
    all_labels = []
    for batch_results in results[0]:
        label_ids = np.argmax(batch_results, axis=1).tolist()
        labels = [inverse_label_map[label_id] for label_id in label_ids]
        all_labels.extend(labels)

    # 返回结果
    return all_labels[0]

