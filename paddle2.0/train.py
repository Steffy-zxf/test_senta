import argparse
import ast
import os
import time

from sklearn import metrics
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from chnsenticorp import ChnSentiCorp
from model import BowTextClassifier
from tokenizer import CustomTokenizer

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=5, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate used to train with warmup.")
parser.add_argument("--checkpoint_dir", type=str, default='saved_model.pt', help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.

def generate_batch(batch):
    label = [entry[1] for entry in batch]
    text = [entry[0] for entry in batch]
    return text, label


def train():

    tokenizer = CustomTokenizer(vocab_file='/mnt/zhangxuefei/.paddlehub/modules/senta_bow/assets/vocab.txt')
    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='dev')
    test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='test')

    place = paddle.CPUPlace() # paddle.CUDAPlace(ParallelEnv().dev_id)
    paddle.disable_static(place)

    # train dataset
    # train_sampler = paddle.io.BatchSampler(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = paddle.io.DataLoader(train_dataset, places=place, batch_size=args.batch_size, shuffle=True, drop_last=False, return_list=True, collate_fn=generate_batch)
    # dev dataset
    # dev_sampler = paddle.io.BatchSampler(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, places=place, drop_last=False, return_list=True, collate_fn=generate_batch)
    # test dataset
    # test_sampler = paddle.io.BatchSampler(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = paddle.io.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, places=place, drop_last=False, return_list=True, collate_fn=generate_batch)

    model = BowTextClassifier(
        dict_dim=tokenizer.vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        fc_hidden_dim=96, 
        num_labels=len(train_dataset.label_list))

    if os.path.exists(args.checkpoint_dir + ".pdparams"):
        para_state_dict, opti_state_dict = paddle.load(args.checkpoint_dir)
        model.set_dict(para_state_dict)
        print("Loaded checkpoint from %s" % (args.checkpoint_dir+'.pdparams'))

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.learning_rate)

    total_steps = 0
    best_acc = -1
    for epoch in range(args.num_epoch):
        start_time = time.time()
        for i, (texts, labels) in enumerate(train_loader):
            model.train()
            reader_end_time = time.time()
            reader_time = reader_end_time - start_time
            outputs = model(texts, labels)
            optimizer.clear_gradients()
            outputs.get('loss').backward()
            optimizer.step()
            run_model_time = time.time() - reader_end_time

            ground_truth = labels.numpy()
            predictions = np.argmax(outputs['probs'].numpy(), axis=1)
            train_acc = metrics.accuracy_score(ground_truth, predictions)
            print('epoch: %d, train step: %d, train_loss: %.5f, train_acc: %.5f, reader time cost: %.5f, model time cost: %.5f' 
                % (epoch, i, outputs['loss'].numpy(), train_acc, reader_time, run_model_time))
            total_steps += 1
            if total_steps % 10 == 0:
                dev_acc, loss_avg = eval(model, dev_loader)
                msg = 'dev avg loss: %.5f, dev acc: %.5f' % (loss_avg, dev_acc)
                if best_acc < dev_acc:
                    best_acc = dev_acc
                    paddle.save(model.state_dict(), args.checkpoint_dir)
                    msg += ', best acc: %.5f' % best_acc
                print(msg)
            start_time = time.time()
        
    test_acc, loss_avg = eval(model, test_loader)
    print('Finanly')
    print("test acc: %.5f" % test_acc)
        

def eval(model, valid_dataloader):
    model.eval()
    loss_all = 0
    predition_all = np.array([], dtype=np.int32)
    truth_all = np.array([], dtype=np.int32)
    # with paddle.no_grad():
    for texts, labels in valid_dataloader:
        outputs = model(texts, labels)
        loss = outputs.get('loss')
        loss_all += loss
        labels = labels.numpy()
        preditions = np.argmax(outputs['probs'].numpy(), axis=1)
        truth_all = np.append(truth_all, labels)
        predition_all = np.append(predition_all, preditions)

    dev_acc = metrics.accuracy_score(truth_all, predition_all)
    loss_avg = loss_all / len(valid_dataloader) 
    return dev_acc, loss_avg


if __name__ == "__main__":
    train()