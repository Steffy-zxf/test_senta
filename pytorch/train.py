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
parser.add_argument("--num_epoch", type=int, default=20, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate used to train with warmup.")
parser.add_argument("--checkpoint_dir", type=str, default='saved_model.pt', help="Directory to model checkpoint")
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


def train():

    tokenizer = CustomTokenizer(vocab_file='/mnt/zhangxuefei/.paddlehub/modules/senta_bow/assets/vocab.txt')
    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='dev')
    test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=generate_batch)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=generate_batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=generate_batch)

    model = BowTextClassifier(
        dict_dim=tokenizer.vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        fc_hidden_dim=96, 
        num_labels=len(train_dataset.label_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_steps = 0
    best_acc = -1
    for epoch in range(args.num_epoch):
        model.train()
        start_time = time.time()
        for i, (texts, labels) in enumerate(train_loader):
            # texts = np.array([[417311, 93118, 173182, 117997, 213052, 595755, 1106222, 940440, 947651, 169200]], dtype='int64')
            # print(texts)
            # labels = np.array([1], dtype='int64')
            # print(labels)
            # texts = torch.from_numpy(texts)
            # labels = torch.from_numpy(labels)
            reader_end_time = time.time()
            reader_time = reader_end_time - start_time
            outputs = model(texts, labels)
            optimizer.zero_grad()
            outputs.get('loss').backward()
            optimizer.step()
            run_model_time = time.time() - reader_end_time

            ground_truth = labels.numpy()
            predictions = torch.max(outputs['probs'], 1)[1].numpy()
            # print(outputs['probs'], predictions)
            # print(ground_truth.numpy(), predictions[1].cpu().numpy())
            train_acc = metrics.accuracy_score(ground_truth, predictions)
            print('epoch: %d, train step: %d, train_loss: %.5f, train_acc: %.5f, reader time cost: %.5f, model time cost: %.5f' 
                % (epoch, i, outputs['loss'].item(), train_acc, reader_time, run_model_time))
            total_steps += 1
            if total_steps % 10 == 0:
                dev_acc, loss_avg = eval(model, dev_loader)
                msg = 'dev avg loss: %.5f, dev acc: %.5f' % (dev_acc, loss_avg)
                if best_acc < dev_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(), args.checkpoint_dir)
                    msg += ', best acc: %.5f' % best_acc
                print(msg)
            start_time = time.time()
        
        test_acc, loss_avg = eval(model, test_loader)
        print('Finanly')
        print("test acc: %.5f", test_acc)
        

def eval(model, valid_dataloader):
    model.eval()
    loss_all = 0
    predition_all = np.array([], dtype=np.int32)
    truth_all = np.array([], dtype=np.int32)
    with torch.no_grad():
        for texts, labels in valid_dataloader:
            outputs = model(texts, labels)
            loss = outputs.get('loss')
            loss_all += loss
            labels = labels.numpy()
            preditions = torch.max(outputs['probs'], 1)[1].numpy()
            truth_all = np.append(truth_all, labels)
            predition_all = np.append(predition_all, preditions)

    dev_acc = metrics.accuracy_score(truth_all, predition_all)
    loss_avg = loss_all / len(valid_dataloader) 
    return dev_acc, loss_avg



train()
