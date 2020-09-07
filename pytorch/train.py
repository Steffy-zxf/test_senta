import argparse
import ast
import time

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chnsenticorp import ChnSentiCorp
from model import BowTextClassifier
from tokenizer import CustomTokenizer



# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.

def generate_batch(batch):
    label = torch.tensor([entry[1] for entry in batch])
    text = [entry[0] for entry in batch]
    # offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.tensor(text)
    return text, label

def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()



def train():

    tokenizer = CustomTokenizer(vocab_file='/mnt/zhangxuefei/.paddlehub/modules/senta_bow/assets/vocab.txt')
    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='dev')
    # test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=generate_batch)
    dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.batch_size, shuffle=False, collate_fn=generate_batch)

    model = BowTextClassifier(
        dict_dim=tokenizer.vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        fc_hidden_dim=96, 
        num_labels=len(train_dataset.label_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epoch):
        model.train()

        for i, (texts, labels) in enumerate(train_loader):
            # texts = np.array([[417311, 93118, 173182, 117997, 213052, 595755, 1106222, 940440, 947651, 169200]], dtype='int64')
            # print(texts)
            # labels = np.array([1], dtype='int64')
            # print(labels)
            # texts = torch.from_numpy(texts)
            # labels = torch.from_numpy(labels)
            outputs = model(texts, labels)
            optimizer.zero_grad()
            outputs.get('loss').backward()
            optimizer.step()

            ground_truth = labels.data.cpu()
            predictions = torch.max(outputs['probs'].data, 1)[1].cpu()
            # print(outputs['probs'], predictions)
            # print(ground_truth.numpy(), predictions[1].cpu().numpy())
            train_acc = metrics.accuracy_score(ground_truth, predictions)
            print('train step: %d, train_loss: %.5f, train_acc: %.5f' % (i, outputs['loss'].item(), train_acc))


train()