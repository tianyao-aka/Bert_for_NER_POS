import numpy as np
import pandas as pd
from conllu import parse_incr,parse
import torch
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from seqeval.metrics import f1_score
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM,AdamW, WarmupLinearSchedule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from sklearn.metrics import  confusion_matrix,accuracy_score
import pycuda.driver as cuda
#bert-base-multilingual-cased
import nltk
import seaborn as sns
import os
import stanfordnlp
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from seqeval.metrics import classification_report,f1_score
import lookahead as lk



## 本项目使用的数据集链接如下：

#   https://github.com/ialfina/ner-dataset-modified-dee/tree/master/singgalang


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')  # 加载BERT的tokenizer
model = BertModel.from_pretrained('bert-base-multilingual-cased') #加载预训练好的多语言BERT Model


#加载已处理好的数据集，分为两个部分：1）sentences，每个句子格式为：['i','am','working','at','school']，sentences： [[sent1],[sent2],,..]
# labels = [0,1,0,0,0,1], labels:[[sent1_labels],[sent2_labels],[sent3_labels]]

#----------------------------------------------------------------------------------------------------------------

sentences = pickle.load( open( "sentences.pickle", "rb" ) )
labels = pickle.load( open( "labels.pickle", "rb" ) )
entity2idx = {'O':0,'Place':1,'Person':2,'Organisation':3}
def str2idx(x):
    return [entity2idx[i] for i in x]
idx2entity = {0:'O',1:'Place',2:'Person',3:'Company'}

labels = list(map(lambda x:str2idx(x),labels))

#----------------------------------------------------------------------------------------------------------------


def get_device(use_gpu=True):    #utility function：get cpu or cuda device
    if torch.cuda.is_available() and use_gpu:
        return torch.device('cuda:0')
    else:
        return  torch.device('cpu')

def train_loss_diff(history):    # utility function:  返回连续 {len(history)}次training loss的绝对值平均值，用于训练阶段的early stopping
    diffs = []
    for i in range(len(history)-1):
        diffs.append(abs(history[i+1]-history[i]))
    return np.mean(np.asarray(diffs))


def make_dataset(sents, labels):
    #   用于制造适合于BERT的数据集
    #   dataset: list of lists, 用于存放tokenize以后的句子
    #   ner_labels: 用于存放对应tokenize以后的句子关联的标记
    #   data_ids: 每一个单词对应一个globally unique的id，用于之后的数据操作

    dataset = []
    ner_labels = []
    data_ids = []
    ids = 1
    for s, l in zip(sents, labels):
        sent = []
        sent_tag = []
        sent_ids = []
        for idx in range(len(s)):
            w = tokenizer.tokenize(s[idx])
            sent.extend(w)
            sent_tag.extend([l[idx]] * len(w))
            sent_ids.extend([ids] * len(w))
            ids += 1
        dataset.append(sent)
        ner_labels.append(sent_tag)
        data_ids.append(sent_ids)

    return (dataset, ner_labels, data_ids)


def change_dataset(dataset, labels, sent_ids, max_len=172):

    # 该函数进一步处理数据集，由于BERT的输入不是等长的，需要添加[CLS],[PAD]等相应特殊token。
    # dataset_tensor: 经过padding处理过后的dataset，数据结构为long tensor
    # labels_tensor: 经过padding处理后的labels，数据结构为long tensor
    # attn_mask: 作为输入的一部分在BERT Model 前向计算时需要用到的tensor， 用于指示哪些为padding token。 注：0为padded token
    # sent_id_tensor: 不同的token可能对应一个词，比如 love 经过BERT Tokenizer处理后，变为 lov, ##e。 sent_id用于给同一个单词打上同一个标记，用于后续NER模型的优化处理

    sent_id_tensor = []
    label_tensor = []
    dataset_tensor = []
    padded_data = []
    padded_labels = []
    padded_ids = []
    for idx, d in enumerate(dataset):
        labl = labels[idx]
        ids = sent_ids[idx]
        if len(d) >= max_len - 2:
            d = d[:max_len - 2]
            d = ['[CLS]'] + d + ['[SEP]']
            padded_data.append(d)

            labl = labl[:max_len - 2]
            labl = [0] + labl + [0]
            padded_labels.append(labl)

            ids = ids[:max_len - 2]
            ids = [-1] + ids + [-1]
            padded_ids.append(ids)


        else:
            d = ['[CLS]'] + d + ['[SEP]']
            labl = [0] + labl + [0]
            ids = [-1] + ids + [-1]

            while len(d) < max_len:
                d.append('[PAD]')
                labl.append(0)
                ids.append(-1)
            padded_data.append(d)
            padded_labels.append(labl)
            padded_ids.append(ids)

    for d in padded_data:
        dataset_tensor.append(tokenizer.convert_tokens_to_ids(d))
    dataset_tensor = torch.tensor(dataset_tensor).long()

    label_tensor = torch.tensor(padded_labels).long()

    sent_id_tensor = torch.tensor(padded_ids).long()

    attn_mask = dataset_tensor != 0
    attn_mask = attn_mask

    return dataset_tensor, label_tensor, attn_mask, sent_id_tensor


class Multiclass_Focal_Loss(nn.Module):

    # 该函数用于NER的自定义loss function， 主要用于解决类的不平衡问题，数据集中'O'的数量为其他样本类型数量的20倍左右，因此为了让模型能够收敛，
    # 本项目中使用了两种方法：
    # 1）采用focal-loss进行计算，减弱'O'标记样本的影响。本项目中我将focal loss扩展成适用于multi-class
    # 2） 另外一种解决样本不平衡的方法采用了hard negative mining
    # 1）中方法详见 https://arxiv.org/abs/1708.02002-Focal Loss for Dense Object Detection
    # 2—）中方法详见https://arxiv.org/pdf/1512.02325.pdf- SSD: Single Shot MultiBox Detector

    # 经过实践，最终采用hard negative mining的方法，由于可以达到更高的f1 score


    def __init__(self, alpha=2):
        super(Multiclass_Focal_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, labels):
        outputs = outputs.to(device)
        labels = labels.to(device)
        type_i_mask = labels > 0
        type_ii_mask = labels == 0
        #         print ('labels:',labels[:5])

        labels = labels.view(-1, 1)
        costs = torch.gather(outputs, 1, labels)
        costs = costs.view(-1)
        costs = -1. * torch.log(costs)
        type_i_loss = costs[type_i_mask]
        type_ii_loss = costs[type_ii_mask]
        N = len(type_i_loss)
        type_ii_loss_truncated = torch.sort(type_ii_loss, descending=True)[0][:int(2.5 * N)]
        total_costs = (type_i_loss.sum() + type_ii_loss_truncated.sum()) / int((3.5 * N)) * 1.

        #         N = len(labels)
        #         labels = labels.view(-1,1)
        #         costs = torch.gather(outputs,1,labels)
        #         costs = costs.view(-1)
        #         log_costs = -1.*torch.log(costs)
        #         squared_cost = (1-costs)**self.alpha
        #         total_cost = torch.sum(log_costs*squared_cost)/N

        return total_costs


class NER_Model(nn.Module):
    # 该NER_Model基于多语言的BERT模型，但是用于NER的任务进行了优化改进。原paper中用每个token对该单词的label进行训练和预测，本模型结合local context
    # 和global context一起对每个单词的分类进行预测。h_cls为每个句子的hidden embedding， h_{token}为对应token的embedding， 则举例对于love这个词的词性分类
    #采用如下特征进行描述： h_cls||AGG(h_{lov},h_{##e}) 作为love这个单词的特征，这里的AGGREGATION FUNCTION采用average operator
    #以下是该模型的具体实现：

    def __init__(self, model, alpha):
        super(NER_Model, self).__init__()
        self.model = model
        self.linear = nn.Linear(768 * 2, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=1)
        self.lossfunc = Multiclass_Focal_Loss(alpha)


    def forward(self, inputs, labels, attn_mask, sent_ids, extract_feats=False):
        out = self.model(inputs, attention_mask=attn_mask)
        out = out[0]
        cls_dict = self.build_dict(out, sent_ids)
        data = []
        label_list = []
        for k in np.unique(sent_ids.cpu().numpy()):
            if k == -1:
                continue
            cls_vector = cls_dict[k]
            mask = sent_ids == k
            temp = out[mask]

            data.append(self.avg_vector(cls_vector, temp))
            label_list.append(labels[mask][0])

        data = list(map(lambda x: x.view(1, -1), data))
        data = torch.cat(data, dim=0)
        data = data.float()

        label_list = torch.tensor(list(map(lambda x: x.item(), label_list))).long().to(device)
        output = self.linear(data)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.softmax(output)
        cost = self.lossfunc(output, label_list)
        if not extract_feats:
            return cost
        else:
            #             print (label_list.shape,label_list[:10])

            out = torch.argmax(output, dim=1).to(device)
            #             print (out[:6])
            return cost, label_list, out

    def build_dict(self, out, sent_ids):
        sent_ids = sent_ids.cpu().numpy()
        cls_dict = dict()
        N = sent_ids.shape[0]
        for i in range(N):
            for j in set(list(sent_ids[i, :])):
                if j == -1: continue
                cls_dict[j] = out[i][0]
        return cls_dict

    def avg_vector(self, cls_vector, inputs):
        if len(inputs) == 1:
            return torch.cat((cls_vector, inputs.squeeze()))

        return torch.cat((cls_vector, torch.mean(inputs, dim=0)))


def eval_model(ner_model, dev_data_gen):

    # utility function: 用于验证模型的性能，输入为我们的模型和dev data loader，输出为entity-level f1-score.
    #具体可参见：http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/


    print('eval.........')
    #     torch.cuda.empty_cache()
    ner_model = ner_model.to(device)
    #     ner_model = nn.DataParallel(ner_model,device_ids)
    ner_model.eval()
    y_trues = []
    y_preds = []
    losses = []
    with torch.no_grad():
        for inputs, labels, attn_mask, sent_ids in dev_data_gen:

            rand_num = np.random.uniform()
            if rand_num > 0.4:
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)
            attn_mask = attn_mask.to(device)
            sent_ids = sent_ids.to(device)

            cost, y_true, y_pred = ner_model(inputs, labels, attn_mask, sent_ids, extract_feats=True)
            y_trues.append([idx2entity[x] for x in list(y_true.cpu().numpy())])
            y_preds.append([idx2entity[x] for x in list(y_pred.cpu().numpy())])
        #             losses.append(cost.item())
        #             print (y_trues)
        #             print (y_preds)
        #             print (losses)
        #         eval_loss = np.sum(np.asarray(losses))/len(losses)
        #         print ('----------------------------------')
        #         print (y_trues)
        print(classification_report(y_trues, y_preds))
    #         con_mat = confusion_matrix(y_trues,y_preds)
    #         acc_score = accuracy_score(y_true,y_pred)

    del inputs
    del labels
    del attn_mask
    del sent_ids
    return f1_score(y_trues, y_preds)


def train(ner_model, train_dset, dev_data_gen, batch_size=124, step_every=60, lr=2e-4, warmup_steps=900,total_steps=9000):

    # 用于训练ner模型，本函数最终采用Adam optimizer， lr= 2e-4, epoch=30, batch_size = 124, 用4块GPU并行训练
    # 由于该数据集较大，且噪声较大，当batch size较小时（16或者32）， 模型无法收敛，只有当batch size >= 96时，模型才收敛。


    torch.cuda.empty_cache()
    #     ner_model =ner_model.to(device)
    ner_model = nn.DataParallel(ner_model, device_ids)
    ner_model.train()
    history = []
    print('go')
    best_f1 = 0.
    min_training_error = 1.

    #     optimizer =  AdamW(ner_model.parameters(), lr=lr, correct_bias=False)
    adam_optim = optim.Adam(ner_model.parameters(), lr=lr)
    #     lookahead = lk.Lookahead(adam_optim, k=5, alpha=0.5)

    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 8}
    train_data_gen = data.DataLoader(train_dset, **params)
    steps = 0
    for e in range(30):
        print('epoch: ', e)
        for inputs, labels, attn_mask, sent_ids in train_data_gen:
            #             inputs = inputs.to(device)
            #             labels = labels.to(device)
            #             attn_mask = attn_mask.to(device)
            #             sent_ids = sent_ids.to(device)

            steps += 1

            loss = ner_model(inputs, labels, attn_mask, sent_ids)

            loss.sum().backward()
            adam_optim.step()
            #             scheduler.step()
            adam_optim.zero_grad()
            history.append(loss.sum().item())
            if steps % 20 == 0:
                print('training error: ', loss.sum().item())
                print('step:', steps)

            if loss.sum().item() < min_training_error and loss.sum().item() < 0.05 and best_f1 > 0.85:
                p = np.random.uniform()
                if p > 0.6:
                    continue
                min_training_error = loss.sum().item()
                print('-----------------eval mode------------------')
                b = eval_model(ner_model, dev_data_gen, loss_func=None)
                print('eval f1_score', b)
                print('------------------end ----------------------')

                if b >= best_f1 and b > 0.85:
                    print('----------saving model-----------------')
                    path = 'best_NLER_model_f1_score_' + str(b)
                    torch.save(ner_model.state_dict(), path)
                    files = current_best_f1_measure()
                    if len(files) >= 8:
                        for i in files[:3]:
                            os.remove(i)

                    print('')
                    print('')
                    print('----------end saving model-------------')
                if b > best_f1:
                    best_f1 = b
            #             if d>0.94:
            #                 print ('----------saving model-----------------')
            #                 path = 'best_ner_model_acc_score_'+str(d)
            #                 torch.save(ner_model.state_dict(), path)
            #                 print ('')
            #                 print ('')
            #                 print ('----------end saving model-------------')

            if steps % step_every == 0 and steps > 0:
                print('-----------------eval mode------------------')
                b = eval_model(ner_model, dev_data_gen, loss_func=None)
                print('eval f1_score', b)
                print('------------------end ----------------------')

                if b >= best_f1 and b > 0.85:
                    print('----------saving model-----------------')
                    path = 'best_NLER_model_f1_score_' + str(b)
                    torch.save(ner_model.state_dict(), path)

                    print('')
                    print('')
                    print('----------end saving model-------------')
                if b > best_f1:
                    best_f1 = b

    #             if d>0.94:
    #                 print ('----------saving model-----------------')
    #                 path = 'best_ner_model_acc_score_'+str(d)
    #                 torch.save(ner_model.state_dict(), path)
    #                 print ('')
    #                 print ('')
    #                 print ('----------end saving model-------------')

    #                 diff = train_loss_diff(history[::-1][:10])
    #                 if diff<0.00000005:
    #                     return  history
    return history

## indicate devices to use, here we use 4 GPUs

# training scripts
#---------------------------------------------------------------------------------------------------
device = get_device(use_gpu = True)
print (device)
cuda1 = torch.device('cuda:0')
cuda2 = torch.device('cuda:1')
cuda3 = torch.device('cuda:2')
cuda4 = torch.device('cuda:3')
device_ids = [cuda1,cuda2,cuda3,cuda4]




a,b,c = make_dataset(sentences[:],labels[:])
dataset_tensor,label_tensor,attn_mask, sent_id_tensor = change_dataset(a,b,c)   # process dataset

model = BertModel.from_pretrained('bert-base-multilingual-cased')
ner_model = NER_Model(model,3) # load model
ner_model.cuda(0)  # move model to cuda device
params = {'batch_size': 32,
          'shuffle': False,
          'num_workers': 8 }

N = dataset_tensor.size(0)
temp = np.arange(N)
np.random.shuffle(temp)
train_idx = torch.tensor(temp[:int(0.8*N)]).long()   # 80% for training data
test_idx = torch.tensor(temp[int(0.8*N):]).long()    # 20% for test data
train_dataset_tensor,train_label_tensor,train_attn_mask, train_sent_id_tensor = dataset_tensor[train_idx],label_tensor[train_idx],attn_mask[train_idx], sent_id_tensor[train_idx]
test_dataset_tensor,test_label_tensor,test_attn_mask, test_sent_id_tensor = dataset_tensor[test_idx],label_tensor[test_idx],attn_mask[test_idx], sent_id_tensor[test_idx]

train_dset = data.TensorDataset(train_dataset_tensor,train_label_tensor,train_attn_mask, train_sent_id_tensor ) # 将dataset变成pytorch的tensordataset

dev_dset = data.TensorDataset(test_dataset_tensor,test_label_tensor,test_attn_mask, test_sent_id_tensor) # 将dataset变成pytorch的tensordataset

dev_loader = data.DataLoader(dev_dset,**params)   # 形成test dataset的data loader


loss_history = train(ner_model,train_dset,dev_loader,batch_size=120,step_every=50,lr = 2e-5, warmup_steps = 950, total_steps = 9500)

#-------------------------------------------------------------------------------------------------


# inference- using trained model to do inference on unseen data
#-------------------------------------------------------------------------------------------------

from collections import OrderedDict

device = get_device(False)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
ner_model = NER_Model(model,3)

params = OrderedDict()
path_name = 'best_NLER_model_f1_score_0.8714718053239453'
s = torch.load(path_name)
for k in s:
    params[k[7:]] = s[k]
ner_model.load_state_dict(params)



def extract_labels(outputs, cum_sents_length, idx2entity):
    s = 0
    final_outputs = []
    for i in range(len(cum_sents_length) - 1):
        if i == 0:
            temp = outputs[:cum_sents_length[i]].cpu().numpy()
            final_outputs.append([idx2entity[t] for t in temp])
            s += 1
        temp = outputs[cum_sents_length[i]:cum_sents_length[i + 1]].cpu().numpy()
        final_outputs.append([idx2entity[t] for t in temp])

    return final_outputs


def inference_sents(sents, ner_model, idx2entity):
    # format of sents: sents::List[String]   e.g.:['i love studying .', 'good job, nice work.','deep learning is fun !',.....]
    if len(sents) > 32:
        print('number of sentences must be less than 33')
        return
    ret = copy.deepcopy(sents)
    sents = [s.split() for s in sents]
    sent_length = [len(s) for s in sents]
    labels = [[0] * t for t in sent_length]
    cum_sents_length = np.cumsum(np.asarray(sent_length))

    a, b, c = make_dataset(sents[:], labels[:])
    dataset_tensor, label_tensor, attn_mask, sent_id_tensor = change_dataset(a, b, c)
    _, _, o2 = ner_model(dataset_tensor, label_tensor, attn_mask, sent_id_tensor, extract_feats=True)
    labels = extract_labels(o2, cum_sents_length, idx2entity)
    return ret, labels


sents = ['Anda bisa juga langsung melakukan prediksi dengan menggunakan model yang telah saya buat , yaitu','Ngurusin data lagi untuk kerjaan suatu kementerian .']
sents,labels = inference_sents(sents,ner_model,idx2entity)

