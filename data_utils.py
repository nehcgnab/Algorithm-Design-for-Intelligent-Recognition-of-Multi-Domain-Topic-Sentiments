import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer



# build_tokenizer函数用于构建分词器，它会读取数据集文件并构建词典。
# 如果已经存在预训练好的分词器，就直接加载，否则会遍历数据集文件，将文本转换为小写并进行分词处理
def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        # 从train和test数据集中加载数据进行构建
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 4):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        # Tokenizer类：包括fit_on_text函数与text_to_sequence
        tokenizer = Tokenizer(max_seq_len)
        # 构建词典key-value
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


# _load_word_vec函数用于从预训练的词向量文件中加载词向量
# 从glove.42B.300d文件将word的编码embedding读取出来，存在word_vec中
def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        # Python rstrip() 删除 string 字符串末尾的指定字符（默认为空格)
        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        tokens = line.rstrip().split()
        # 第一个token为word；第二个开始（-embed_dim）到最后一个为该word的embedding矩阵
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        # Python 字典(Dictionary) keys() 函数以列表返回一个字典所有的键值
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


# build_embedding_matrix函数用于构建词嵌入矩阵，它会根据词典中的单词查找对应的词向量并构建嵌入矩阵
def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'

        # _load_word_vec函数从glove.42B.300d文件将word的embedding读取出来，存在word_vec中
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)

        # word2idx=3598；embedding_matrix=（3600，300）
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

# pad_and_truncate函数用于对序列进行填充和截断操作，确保序列长度不超过设定的最大长度
# padding='post', truncating='post', value=0 表示对转换后的序列进行填充和截断操作时，填充和截断的方式是在序列末尾用 0 来进行填充和截断
def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    #  truncating='pre'，表示在序列的开头进行截断
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


# Tokenizer类用于构建通用分词器
class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        # 定义两个字典
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    # 将文本输入拟合到分词器中，以构建词汇表。构建词典key-value
    def fit_on_text(self, text):
        if self.lower:
            # Python lower()方法转换字符串中所有大写字符为小写
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    # 将文本转换为序列。
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


# Tokenizer4Bert类用于构建BERT分词器
class Tokenizer4Bert:
    #
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        # bath = './models/bert/'
        # self.tokenizer = BertTokenizer.from_pretrained('bert')
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


# ABSADataset类用于构建数据集。在初始化时，它会读取数据集文件，并根据分词器将文本转换为序列。
# 同时，它会加载依存图数据，并将文本序列和依存图数据整合成一个样本。
class ABSADataset(Dataset):
    # fname：选择输入的数据集    tokenizer：
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname + '.graph', 'rb')
        # 反序列化对象。将文件中的数据解析为一个Python对象。其中要注意的是，在load(file)的时候，要让python能够找到类的定义，否则会报错：
        # 加载依存图：通过open()和pickle.load()函数加载与数据文件相关联的依存图数据。依存图通常是由另外的工具生成，并用于帮助模型理解文本中单词之间的关系。
        idx2graph = pickle.load(fin)
        fin.close()
        category_dict = {
            "菲军舰": 0,
            "疯狂的大葱": 1,
            "官员财产公示": 2,
            "官员调研": 3,
            "教育制度": 4,
            "韩寒方舟子之争": 5,
            "假和尚": 6,
            "奖状植入广告": 7,
            "90后暴打老人": 8,
            "90后当教授": 9,
            "六六叫板小三": 10,
            "南京大屠杀": 11,
            "南京老太": 12,
            "皮鞋果冻": 13,
            "苹果封杀360": 14,
            "三亚春节宰客": 15,
            "食用油涨价": 16,
            "洗碗工人": 17,
            "钓鱼执法": 18,
            "中国教师收入": 19
        }
        all_data = []
        # range(start, stop[, step])：数据集按照第一行为句子、第二行为方面词、第三行为情感极性标签
        for i in range(0, len(lines), 4):
            # print(len(lines))
            # lines[i].partition("$T$")按照$T$字符分为三份：左边内容、$T$、右边内容
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text = text_left + " " + aspect + " " + text_right
            polarity = lines[i + 2].strip()

            domain = lines[i + 3].strip()    # 修改
            domain = int(category_dict[domain])   # 修改

            # text_indices:整个句子内容  ，context_indices：除去aspect词的内容
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            right_len = np.sum(right_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity)

            # domain = tokenizer.text_to_sequence(domain)     # 修改

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP] " + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
            concat_bert_contex = tokenizer.text_to_sequence("[CLS] " + text_left + " " + text_right + " [SEP]")

            text_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
            context_len = np.sum(context_indices != 0)

            # 构建依赖图（dependency graph）
            dependency_graph = np.pad(idx2graph[i], \
                                      ((0, tokenizer.max_seq_len - idx2graph[i].shape[0]),
                                       (0, tokenizer.max_seq_len - idx2graph[i].shape[0])), 'constant')

            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,
                'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_boundary': aspect_boundary,
                'dependency_graph': dependency_graph,
                'polarity': polarity,
                'domain':domain,    # 修改
                'concat_bert_contex': concat_bert_contex,
                'text':text_left + " " + aspect + " " + text_right,
                'bert_input':"[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]"
            }

            all_data.append(data)
        self.data = all_data

    # __getitem__和__len__方法，通过索引获取样本和获取数据集的长度
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
