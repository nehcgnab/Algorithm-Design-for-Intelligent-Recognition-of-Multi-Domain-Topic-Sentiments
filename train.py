import logging
import argparse
import math
import os
import sys
import random
import numpy
from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import AOAN
import matplotlib.pyplot as plt


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Instructor 类的作用是组织和管理整个模型的训练和评估过程
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        #根据model类型选择embedded输入
        if 'bert' in opt.model_name:
            # tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            tokenizer = Tokenizer4Bert(opt.max_seq_len, 'models/bert-base-chinese')
            # tokenizer = AutoTokenizer.from_pretrained(opt.max_seq_len,opt.pretrained_bert_name)
            # bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            bert = BertModel.from_pretrained('models/bert-base-chinese')
            # bert = AutoModelForMaskedLM.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                # 指定位置占位 {0}, your balance is {1}.".format("Adam", 230.2346))
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))

            # 词嵌入矩阵 embedding_matrix ： 通过build_embedding_matrix函数获得embedding_matrix=（3600，300）---->从glove.42B.300d.txt'中得到
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            # 选择模型并加载参数
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        # 记录使用cuda的内存分配地址
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    #打印模型的可训练参数数量以及训练参数的配置信息
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # 该方法用于重置模型的参数。它遍历模型的所有子模块，对于不是 BertModel 的子模块，对其参数进行初始化。
    # 如果参数的形状大于 1，则采用指定的初始化器进行初始化；否则，采用均匀分布进行初始化
    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # 该方法是整个训练过程的核心部分。它接受损失函数、优化器以及训练集和验证集的数据加载器作为输入，然后进行模型训练，
    # 并在每个 epoch 结束后评估验证集的性能，保存表现最佳的模型参数。
    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0

        train_losses = []       ###
        val_losses = []
        train_accs = []
        val_accs = []
        train_f1s = []
        val_f1s = []            ###

        path = None
        # a = train_data_loader
        # 按照epoch循环
        for i_epoch in range(self.opt.num_epoch):
            # 在每个 epoch 开始时输出日志，其中 logger.info 用于输出信息到日志文件
            logger.info('>' * 100)
            # logger.info()其中{}表示占位符
            logger.info('epoch: {}'.format(i_epoch))
            # 跟踪每个 epoch 的累积正确预测数、总样本数以及累积损失
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            # 将模型设置为训练模式，这样在训练过程中可以启用特定于训练的功能，例如 dropout
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                # 计算全局步数，用于日志记录和调整学习率等
                global_step = global_step+1
                # clear gradient accumulators
                # 将优化器的梯度清零，以便进行新一轮的梯度计算
                optimizer.zero_grad()

                # 将 batch 数据传递给模型进行前向传播，并将输出与真实标签进行比较以计算损失
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]



                # torch.autograd.set_detect_anomaly(True)
                outputs= self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                # 计算损失，反向传播计算梯度并使用优化器来更新模型参数
                loss = criterion(outputs, targets)
                #with torch.autograd.detect_anomaly():
                loss.backward()
                optimizer.step()

                # 更新统计指标，如正确预测数量、总样本数量和累积损失
                n_correct = n_correct + (torch.argmax(outputs, -1) == targets).sum().item()
                n_total = n_total + len(outputs)
                loss_total = loss_total + loss.item() * len(outputs)

                # 用于在每个记录步数时输出训练损失和准确率,周期性地打印
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            # 在每个 epoch 结束后，在验证集上评估模型性能，记录并打印验证集的准确率和 F1 值
            train_acc, train_f1, train_loss = self._evaluate_acc_f1(train_data_loader)   ###
            val_acc, val_f1, val_loss= self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

            train_losses.append(train_loss)   ###
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)          ###

            # 如果当前模型在验证集上取得了比历史最优模型更好的性能，则保存当前模型参数
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_{2}_val_acc_{3}_f1_{4}'.format(self.opt.model_name, self.opt.dataset, self.opt.threshold, round(val_acc, 4), round(val_f1, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
            logger.info('> best_val_acc: {:.4f}, val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
            # 如果验证集性能在连续多个 epoch 上没有明显改善，则提前结束训练
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        self.plot_graph(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)     ###

        return path

    def _evaluate_acc_f1(self, data_loader,_istest=False):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        # 将模型设置为评估模式
        self.model.eval()
        incorr=[]
        # 上下文管理器，用于确保在评估过程中不计算梯度，以节省内存和计算资源
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text=t_batch['text']
                # 将 batch 数据传递给模型进行前向传播，并获取模型的预测输出
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols[:len(self.opt.inputs_cols)]]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                # pre 变量将保存模型对于当前 batch 中每个样本的预测结果，每个预测结果是对应类别的索引
                pre = torch.argmax(t_outputs, -1)
                # 更新正确预测的数量和总样本数量
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                #print(t_outputs)
                # 将每个 batch 的真实标签和模型输出连接起来，以便在整个数据集上计算准确率和 F1 分数
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        # 计算模型在给定数据上的准确率和 F1 分数，并将其返回
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')

        loss = nn.CrossEntropyLoss()(t_outputs_all, t_targets_all).item()    ###

        return acc, f1, loss

    def plot_graph(self, train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='train_acc')
        plt.plot(val_accs, label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_f1s, label='train_f1')
        plt.plot(val_f1s, label='val_f1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run(self):
        # Loss and Optimizer
        # 交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 从模型的参数中筛选出需要进行梯度更新的参数 _params
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # 根据指定的优化器类型和学习率、权重衰减率创建了优化器 optimizer
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        # 创建了训练集、测试集和验证集的数据加载器，用于对数据进行批处理
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        # 调用 _reset_params 方法重置模型的参数，根据指定的初始化策略对参数进行初始化
        self._reset_params()
        # 调用 _train 方法开始训练模型，并传入损失函数、优化器以及训练集和验证集的数据加载器
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        # 加载在验证集上表现最佳的模型参数
        self.model.load_state_dict(torch.load(best_model_path))
        # 调用 _evaluate_acc_f1 方法评估测试集上的准确率和 F1 分数，并输出结果
        test_acc, test_f1,test_loss = self._evaluate_acc_f1(test_data_loader,True)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))




def main():
     # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='AOAN_bert', type=str)
    parser.add_argument('--dataset', default='毕设', type=str, help='guan, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=0.00001, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=30, type=int, help='try larger number for non-BERT models')#default=30
    parser.add_argument('--batch_size', default=4, type=int, help='try 16, 32, 64 for BERT models')#default=32
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)#default=100
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=2123, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--threshold', default=10, type=int, help='hyperparameter L, see the paper of AOAN model')
    opt = parser.parse_args()

    # 设置随机种子，以确保实验的可重复性
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'AOAN_bert': AOAN,
    }
    dataset_files = {
        '毕设': {
            'train': './datasets/dataset/train_data.xml.seg',
            'test': './datasets/dataset/test_data.xml.seg'
        },

    }
    input_colses = {
        'AOAN_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices','aspect_boundary','domain'],
    }

    # 定义了几种参数初始化方法，用于初始化神经网络中的权重参数,Xavier均匀初始化（xavier_uniform_）、正态初始化（xavier_normal_）和正交初始化（orthogonal_）
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
