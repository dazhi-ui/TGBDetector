
import torch.utils.data
import torch.nn as nn
import numpy as np
import argparse
import random
import torch
import os
from ADOA_cadets_1 import ADOA
from sklearn.metrics import *
from lightgbm import LGBMClassifier
import copy
from os import path
import torch.nn.functional as F

# 给Transformer定义一些全局变量
d_ff = 2048  # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6  # 有多少个encoder和decoder
n_heads = 8  # Multi-Head Attention设置为8
torch.cuda.set_device(0)

# 归一化
def min_max_scaler(data_list, min_=-1, max_=-1):
    if min_ < 0:
        min_, max_ = min_max(data_list[0])
        for data in data_list:
            minT, maxT = min_max(data)
            if minT < min_:
                min_ = minT
            if maxT > max_:
                max_ = maxT
    data_list_new = []
    for data in data_list:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min_) / (max_ - min_)
        data_list_new.append(data)
    return min_, max_, data_list_new
def min_max(data):
    max_ = data[0][0]
    min_ = data[0][0]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if max_ < data[i][j]:
                max_ = data[i][j]
            if min_ > data[i][j]:
                min_ = data[i][j]
    return min_, max_

def extract_label(filename):
    if filename.find('wget-normal') > -1:
        return 0
    if filename.find('benign') > -1:
        return 0
    if filename.find('attack') > -1:
        return 1
    else:
        return 0

def load_sketches(fh):
    sketches = list()
    for line in fh:
        sketch = list(map(int, line.strip().split()))
        sketches.append(sketch)
    return np.array(sketches)

def prepare_data_1(files, max_len=1000):
    sketches_list = []
    train_label = []
    for file_one in files:
        with open(file_one, 'r') as f:
            sketches = load_sketches(f)
            sketches_list.append(sketches)
            train_label.append(extract_label(file_one))
    min_, max_data, sketches_list = min_max_scaler(sketches_list)
    stream_dataset = StreamDataset(sketches_list, max_len, train_label)
    train_data_loader = torch.utils.data.DataLoader(stream_dataset, batch_size=1) #--------------------------------------batch_size=2
    return train_data_loader

class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, seq, max_len, label):
        self.max_len = max_len
        seq_pad = np.zeros((len(seq), max_len, seq[0].shape[1]))
        self.length = []
        for index, each in enumerate(seq):
            seq_pad[index, :each.shape[0], :] = each
            self.length.append(each.shape[0])
        self.seq = seq_pad
        self.label = label

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {'seq': self.seq[idx, :, :],
                'label': self.label[idx]}



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()  # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_k, d_v, d_model=512):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff, d_model=512):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.d_model = d_model

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, max_len=1000, d_model=512, n_layers=6):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Linear(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs).cuda()
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).cuda()
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Decoder_GRU(nn.Module):
    def __init__(self, inputSize_GRU, hiddenSize_GRU, numLayers_GRU=1):
        super(Decoder_GRU, self).__init__()

        self.inputSize_GRU = inputSize_GRU
        self.hiddenSize_GRU = hiddenSize_GRU
        self.numLayers_GRU = numLayers_GRU
        self.decoder_GRU = nn.GRU(hiddenSize_GRU, hiddenSize_GRU, num_layers=numLayers_GRU)

    def forward(self, seq_GRU, hidden_GRU):
        output_GRU, hidden_GRU = self.decoder_GRU(seq_GRU, hidden_GRU)
        return output_GRU, hidden_GRU

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q, column_q = seq_q.size()
    batch_size, len_k, column_k = seq_k.size()

    zero_row = [0] * column_k
    pad_attn_mask = torch.zeros(batch_size, len_k).to(dtype=torch.bool).cuda()
    for idx, rows in enumerate(seq_k):
        for row_i, row in enumerate(rows):
            if row == zero_row:
                pad_attn_mask[idx][row_i] = True
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, max_len=1000, d_model=512, n_layers=6, numLayers_GRU=1):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)  # 加入位置信息
        self.layers_1 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        self.inputSize_GRU = src_vocab_size
        self.hiddenSize_GRU = d_model
        self.numLayers_GRU = numLayers_GRU
        self.encoder_GRU = nn.GRU(d_model, d_model, num_layers=numLayers_GRU)


    def forward(self, enc_inputs):


        enc_outputs = self.src_emb(enc_inputs)
        enc_inputs_GRU = enc_outputs
        enc_outputs = self.pos_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers_1:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        output_GRU, hidden_GRU = self.encoder_GRU(enc_inputs_GRU.permute(1,0,2))
        # GRU的----------------------------------------------------------------------------------------------------------   --结束

        enc_outputs = enc_outputs + output_GRU.permute(1, 0, 2)
        output_GRU = output_GRU + enc_outputs.permute(1, 0, 2)

        return enc_outputs, enc_self_attns, output_GRU, hidden_GRU


class Attention_GRU(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers=1):
        super(Attention_GRU, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.GRU = nn.GRU(inputSize, hiddenSize, num_layers=numLayers)

    def forward(self, seq):
        output, hidden = self.GRU(seq)
        return output.permute(1, 0, 2)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_len=1000, d_model=512):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(src_vocab_size, max_len).cuda()
        self.Decoder = Decoder(tgt_vocab_size, max_len).cuda()
        self.Decoder_GRU = Decoder_GRU(tgt_vocab_size, d_model, 1).cuda()
        self.embedding_GRU = nn.Linear(src_vocab_size, d_model)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
        self.local_GRU = True
        self.att = Attention_GRU(2 * d_model, 1)
        self.localLen = 3

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs, enc_self_attns, encoderOutput_GRU, hidden_GRU = self.Encoder(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model],
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection(dec_outputs)


        feature__GRU = encoderOutput_GRU[-1]
        decoderInput_GRU = encoderOutput_GRU[-1].unsqueeze(0)
        seq1_GRU = self.embedding_GRU(enc_inputs)
        seq_GRU = seq1_GRU.permute(1, 0, 2)
        length_GRU = seq_GRU.shape[0]
        batchSize_GRU = seq_GRU.shape[1]
        featureDim_GRU = seq_GRU.shape[2]
        res_GRU = torch.zeros(length_GRU, batchSize_GRU, featureDim_GRU)
        if self.local_GRU:
            for i in range(length_GRU):
                output_GRU, hidden_GRU = self.Decoder_GRU(decoderInput_GRU, hidden_GRU)
                left = max(0, i - self.localLen)
                right = min(length_GRU - 1, i + self.localLen)
                outputs = output_GRU.repeat(1, right - left + 1, 1)
                outputs = torch.cat([outputs, seq1_GRU[:, left:right + 1, :]], dim=2)
                weight = F.softmax(self.att(outputs.permute(1, 0, 2)), dim=1)
                res_GRU[i] = output_GRU[0] + torch.sum(seq1_GRU[:, left:right + 1, :] * weight, dim=1)

                decoderInput_GRU = seq_GRU[i].unsqueeze(0)

        # GRU的----------------------------------------------------------------------------------------------------------   --结束
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns, res_GRU.permute(1, 0, 2), seq1_GRU, feature__GRU

# 获取特征
def extract_feature(dataset, model, gpu):
    model.eval()
    feature = []
    labels = []

    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_outputs, enc_self_attns, encoderOutput_GRU, hidden_GRU= model.Encoder(seq)

        enc_outputs_trans = enc_outputs.permute(0, 2, 1)
        pool = nn.AdaptiveAvgPool1d(1)
        # batch
        pool_res = pool(enc_outputs_trans)
        for one_i, one in enumerate(pool_res):
            poll_trans = one.permute(1, 0)
            feature.append(0.5*(poll_trans.data.cpu().numpy()[0]) +  0.5*(np.squeeze(encoderOutput_GRU[-1].cpu().detach().numpy())))

        for label in data['label']:
            labels.append(label)

    return np.array(feature), labels

# 评估模型
def evaluate_model(y_true, y_pred, y_prob):
    assert len(y_true) == len(y_pred)
    assert len(y_true) == len(y_prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)# AUC 代表不同类别的区分度，这个值越大代表模型分类正确的可能性越大。AUC越接近1.0，检测方法真实性越高;
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    gmean = np.sqrt(recall * precision)

    eval_frame ='  ACCuracy（准确性评分） ' + str(round(acc,4)) +'  Precision（精确率） ' + str(round(precision,4)) + '  Recall（召回率） ' + str(round(recall,4)) + '  F1（f1_score） ' + str(round(f1,4))
    return eval_frame,precision,auc


def train_valid(ADOA_test_files,ADOA_train_files,model,max_len,gpu):

    # 使用ADOA来进行判断
    ADOA_test_files_dataset = prepare_data_1(ADOA_test_files, max_len)
    ADOA_train_files_dataset = prepare_data_1(ADOA_train_files, max_len)

    ADOA_test_files_feature, ADOA_test_files_label = extract_feature(ADOA_test_files_dataset, model, gpu)
    ADOA_train_files_feature, ADOA_train_files_label = extract_feature(ADOA_train_files_dataset, model, gpu)

    data = ADOA_train_files_feature
    sample_num = int(0.9 * len(data))  # --------------------------------------------------------------------------------这里取的是正常的相关数据
    sample_list = [i for i in range(len(data))]  # [0, 1, 2, 3]
    sample_list1 = sample_list
    sample_list = random.sample(sample_list, sample_num)  # [1, 2]
    sample_list3 = []
    label0 = ADOA_train_files_label[0]
    ADOA_train_files_label = []
    for each in sample_list1:
        if each not in sample_list:
            sample_list3.append(each)
            ADOA_train_files_label.append(label0)
    ADOA_untrain_files_dataset_some = data[sample_list3, :]
    ADOA_train_files_dataset_some = data[sample_list, :]

    ADOA_yuce_some = np.array((ADOA_test_files_feature.tolist()) + (ADOA_untrain_files_dataset_some.tolist()))
    ADOA_yuce_label_some = ADOA_test_files_label + ADOA_train_files_label

    clf = LGBMClassifier(learning_rate=0.2, num_leaves=200, n_estimators=300)

    result = []
    values = []
    for i in range(1, 1000):
        values.append(0.001 * i)
    percents = [45,55,65,75,85]

    gailv = [2.5,4,5.5,7]


    for value in values:
        for percent in percents:
            for g in gailv:
                print("value:----", value, "   percent:-----", percent, "gailv------",g)
                tester = ADOA(ADOA_train_files_dataset_some, ADOA_yuce_some, clf, return_proba=True,
                              contamination=(len(ADOA_train_files_label) / len(
                                  ADOA_yuce_label_some)), theta=value,percent=percent,gailv=g)  # contamination 数据集中异常值的比例。
                a_pred, a_prob = tester.predict()  # 表示的是预测标签和可能性
                for wen in range(0, len(a_pred)):
                    if a_pred[wen] == 0:
                        a_pred[wen] = 1
                    else:
                        a_pred[wen] = 0
                TP, TN, FN, FP = 0, 0, 0, 0
                for mm in range(0, len(a_pred)):
                    if ADOA_yuce_label_some[mm] == 1 and a_pred[mm] == 1:
                        TP += 1
                    if ADOA_yuce_label_some[mm] == 0 and a_pred[mm] == 0:
                        TN += 1
                    if ADOA_yuce_label_some[mm] == 0 and a_pred[mm] == 1:
                        FP += 1
                    if ADOA_yuce_label_some[mm] == 1 and a_pred[mm] == 0:
                        FN += 1
                print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
                try:
                    if TN == 0 and FN ==0:
                        break
                except:
                    None
                metrics_adoa, _, _ = evaluate_model(ADOA_yuce_label_some, a_pred, a_prob)
                result.append(metrics_adoa)
                print(metrics_adoa)
    return result



# 主函数执行操作。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainPath', help='absolute path to the directory that contains all training sketches',
                        required=True)
    parser.add_argument('-u', '--testPath', help='absolute path to the directory that contains all test sketches',
                        required=True)
    parser.add_argument('-c', '--cross-validation',
                        help='number of cross validation we perform (use 0 to turn off cross validation)', type=int,
                        default=20)
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-s', '--split', help='split rate', type=float, default=0.1)
    parser.add_argument('-a', '--sample', help='sample', type=float, default=1.0)

    # 获取设置参数的相应数值。
    args = parser.parse_args()
    trainPath = args.trainPath
    testPath = args.testPath
    dataset = args.dataset
    splitRate = args.split
    sample = args.sample
    cross_validation = args.cross_validation
    print("训练集占正常样本的比例：", str(1-splitRate))

    # 设置随机种子。
    SEED = 98765432
    random.seed(SEED)
    np.random.seed(SEED)

    train = os.listdir(trainPath)
    test = os.listdir(testPath)
    train_files = [os.path.join(trainPath, f) for f in train]
    print("正常样本数据集个数：", len(train_files))
    test_files_temp = [os.path.join(testPath, f) for f in test]
    print("异常样本数据集个数", len(test_files_temp))

    cv = 1
    ADOA_test_files = copy.deepcopy(test_files_temp)
    ADOA_train_files = copy.deepcopy(train_files)
    ADOA_results = []
    print("\x1b[6;30;42m[STATUS]\x1b[0m Performing {} cross validation".format(cross_validation))

    modelPath = 'model/Tf_wget_baseline_正常-优质-' + str(round(float(1 - splitRate), 2))  # ---------------------------------------------------------------------------模型的存放位置
    result_place = "results/wget_baseline+3/"
    if not os.path.exists(result_place):
        os.makedirs(result_place)
    file = os.listdir(modelPath)
    for f2 in range(len(file)):
        f = file[len(file) - f2 - 1]
        real_url = path.join(modelPath, f)
        print('当前使用的模型文件：', real_url)
        gpu = torch.cuda.is_available()
        max_len = 340  # -------------------------------------------------------------------------------------------------maxlen=400
        input_size = 2000
        model = Transformer(input_size, input_size)
        if gpu:
            model = model.cuda()
        model.load_state_dict(torch.load(real_url))
        for kk in range(0, cross_validation):
            wen_xieru = open(result_place + f + "_wget_baseline.txt",
                             "a")  # -------------------------------------------------------------------保存文件的名称
            print("\x1b[6;30;42m[STATUS] Test {}--{}/{}\x1b[0m:".format(cv, kk, args.cross_validation))
            result = train_valid(ADOA_test_files, ADOA_train_files, model, max_len, gpu)
            for res in result:
                ADOA_results.append(res)
                wen_xieru.write(res + '\n')
                print(res)
            wen_xieru.close()
        cv += 1
        print("部分结束")
    print("")
    print("开始输出最终结果：")
    for i in ADOA_results:
        print(i)
