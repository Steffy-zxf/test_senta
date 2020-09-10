# coding:utf-8
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
from typing import Dict, List, Optional, Union, Tuple
import csv
import io
import os

import paddle
import tqdm

from tokenizer import CustomTokenizer


class InputExample(object):
    """
    The input data structure of Transformer modules (BERT, ERNIE and so on).
    """

    def __init__(self, guid: int, text_a: str, text_b: Optional[str] = None, label: Optional[str] = None):
        """
        The input data structure.

        Args:
          guid (:obj:`int`):
              Unique id for the input data.
          text_a (:obj:`str`, `optional`, defaults to :obj:`None`):
              The first sequence. For single sequence tasks, only this sequence must be specified.
          text_b (:obj:`str`, `optional`, defaults to :obj:`None`):
              The second sequence if sentence-pair.
          label (:obj:`str`, `optional`, defaults to :obj:`None`):
              The label of the example.

        Examples:
            .. code-block:: python
                from paddlehub.datasets.base_nlp_dataset import InputExample

                example = InputExample(guid=0,
                                text_a='15.4寸笔记本的键盘确实爽，基本跟台式机差不多了',
                                text_b='蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错',
                                label='1')
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b is None:
            return "text={}\tlabel={}".format(self.text_a, self.label)
        else:
            return "text_a={}\ttext_b={},label={}".format(self.text_a, self.text_b, self.label)


class ChnSentiCorp(paddle.io.Dataset):
    """
    ChnSentiCorp is a dataset for chinese sentiment classification,
    which was published by Tan Songbo at ICT of Chinese Academy of Sciences.
    """

    def __init__(self, tokenizer: Union[CustomTokenizer], max_seq_len: int = 128, mode: str = 'train'):
        """
        Args:
            tokenizer (:obj:`BertTokenizer` or :obj:`CustomTokenizer`):
                It tokenizes the text and encodes the data as model needed.
            max_seq_len (:obj:`int`, `optional`, defaults to :128):
                The maximum length (in number of tokens) for the inputs to the selected module,
                such as ernie, bert and so on.
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train, test or dev).

        Examples:
            .. code-block:: python
                from paddlehub.datasets.chnsenticorp import ChnSentiCorp
                from paddlehub.tokenizer.bert_tokenizer import BertTokenizer

                tokenizer = BertTokenizer(vocab_file='./vocab.txt')

                train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=120, mode='train')
                dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=120, mode='dev')
                test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=120, mode='test')
        """
        base_path = '/mnt/zhangxuefei/.paddlehub/dataset/chnsenticorp/'
        if mode == 'train':
            data_file = 'token_train.tsv'
        elif mode == 'test':
            data_file = 'token_test.tsv'
        else:
            data_file = 'token_dev.tsv'

        self.data_file = os.path.join(base_path, data_file)
        self.label_list = ["0", "1"]
        self.label_map = {item: index for index, item in enumerate(self.label_list)}

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.examples = self._read_file(self.data_file, is_file_with_header=True)

        self.records = self._convert_examples_to_records(self.examples)

    def _read_file(self, input_file, is_file_with_header: bool = False) -> List[InputExample]:
        """
        Reads a tab separated value file.

        Args:
            input_file (:obj:str) : The file to be read.
            is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
                Whether or not the file is with the header introduction.

        Returns:
            examples (:obj:`List[InputExample]`): All the input data.
        """
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    example = InputExample(guid=seq_id, label=line[0], text_a=line[1])
                    seq_id += 1
                    examples.append(example)
                return examples

    def _convert_examples_to_records(self, examples: List[InputExample]) -> List[dict]:
        """
        Converts all examples to records which the model needs.
        Args:
            examples(obj:`List[InputExample]`): All data examples returned by _read_file.
        Returns:
            records(:obj:`List[dict]`): All records which the model needs.
        """
        records = []
        for example in examples:
            record = self.tokenizer.encode(
                text=example.text_a, text_pair=example.text_b, max_seq_len=self.max_seq_len)
            # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
            # When all words are not found in the vocab, the text will be dropped.
            if not record:
                logger.info("The text %s has been dropped as it has no words in the vocab after tokenization." %
                            example.text_a)
                continue
            if example.label:
                record['label'] = self.label_map[example.label]
            records.append(record)
        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        if 'label' in record.keys():
            return record['text'], record['label']
        else:
            return record['text']

    def __len__(self):
        return len(self.records)


if __name__ == "__main__":
    tokenizer = CustomTokenizer(vocab_file='/mnt/zhangxuefei/.paddlehub/modules/senta_bow/assets/vocab.txt')
    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=10, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=10, mode='dev')
    test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=10, mode='test')

    index = 0
    while index < 3:
        record = train_dataset.__getitem__(index)
        print("train record: ", record)
        record = dev_dataset.__getitem__(index)
        print("dev record: ", record)
        record = test_dataset.__getitem__(index)
        print("test record: ", record)
        index += 1
