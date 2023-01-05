import numpy as np
import paddle

from functools import partial

from train_cnn import create_dataloader, NewsData, CNNModel
from utils import convert_example, read_vocab, write_results
from paddlenlp.data import Stack, Pad, Tuple

vocab = read_vocab("work/data/vocab.txt")
stop_words = read_vocab("work/data/stop_words.txt")
# Loads dataset.
train_ds = NewsData("work/data/train.txt", mode="train")
dev_ds = NewsData("work/data/dev.txt", mode="dev")
test_ds = NewsData("work/data/test.txt", mode="test")

batch_size = 128

test_batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab.get('[PAD]', 0)),  # input_ids
    Stack(dtype="int64"),  # seq len
): [data for data in fn(samples)]
test_loader = create_dataloader(
    test_ds,
    trans_fn=partial(convert_example, vocab=vocab, stop_words=stop_words, is_test=True),
    batch_size=batch_size,
    mode='test',
    use_gpu=True,
    batchify_fn=test_batchify_fn)

state_dict = paddle.load('./log_cnn/final.pdparams')
model = CNNModel(len(vocab),
        len(train_ds.label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
model.set_state_dict(state_dict)
model = paddle.Model(model)
model.prepare(loss=paddle.nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy())
print(type(model))

# Does predict.

results = model.predict(test_loader)
inverse_lable_map = {value:key for key, value in test_ds.label_map.items()}
all_labels = []
for batch_results in results[0]:
    label_ids = np.argmax(batch_results, axis=1).tolist()
    labels = [inverse_lable_map[label_id] for label_id in label_ids]
    all_labels.extend(labels)

write_results(all_labels, "./result_CNN.txt")
