# import torch
# import torch.nn as nn
# import copy
# import numpy as np
#
# from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention
# # from transformers.modeling_bert import BertPooler, BertSelfAttention
#
# # 自注意力机制
# class SelfAttention(nn.Module):
#     def __init__(self, config, opt):
#         super(SelfAttention, self).__init__()
#         self.opt = opt
#         self.config = config
#         # self.SA是BERT自注意力层的实例，用于处理输入序列
#         self.SA = BertSelfAttention(config)
#         # tanh激活函数
#         self.tanh = torch.nn.Tanh()
#
#     # forward方法定义了自注意力层的前向传播逻辑。它接收一个名为inputs的输入张量，并生成一个全零张量作为注意力遮罩。
#     # 然后，调用BERT自注意力层self.SA处理输入，并将结果通过tanh激活函数进行处理后返回。
#     def forward(self, inputs):
#         zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
#                                             dtype=np.float32), dtype=torch.float32).to(self.opt.device)
#         SA_out = self.SA(inputs, zero_tensor)
#         return self.tanh(SA_out[0])
#
# class MaskAttention(nn.Module):
#     """
#     Compute attention layer
#     """
#     def __init__(self, input_shape):
#         super(MaskAttention, self).__init__()
#         self.attention_layer = torch.nn.Linear(input_shape, 1)
#
#     def forward(self, inputs, mask=None):
#         scores = self.attention_layer(inputs).view(-1, inputs.size(1))
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float("-inf"))
#         scores = torch.softmax(scores, dim=-1).unsqueeze(1)
#         outputs = torch.matmul(scores, inputs).squeeze(1)
#         return outputs, scores
#
#
# class AOAN(nn.Module):
#     def __init__(self, bert, opt):
#         super(AOAN, self).__init__()
#
#         # 初始化模型的各个组件，包括BERT模型、Dropout层、自注意力层、线性层、BERT池化器和全连接层
#         self.bert_spc = bert
#         self.opt = opt
#         self.dropout = nn.Dropout(opt['dropout'])
#         self.bert_SA = SelfAttention(bert.config, opt)
#         self.linear_double = nn.Linear(opt['bert_dim'] * 2, opt['bert_dim'])
#         self.linear_single = nn.Linear(opt['bert_dim'], opt['bert_dim'])
#         self.bert_pooler = BertPooler(bert.config)
#         self.dense = nn.Linear(opt['bert_dim'], opt['polarities_dim'])
#         self.pool=nn.AvgPool1d(opt['threshold']+1)
#
#         # 修改
#         self.domain_num = 20
#         self.gate = nn.Sequential(nn.Linear(opt['bert_dim'] * 2, opt['bert_dim']),
#                                   nn.ReLU(),
#                                   nn.Linear(opt['bert_dim'], self.opt['threshold']+1),
#                                   nn.Softmax(dim=1))
#         self.attention = MaskAttention(opt['bert_dim'])
#         self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=opt['bert_dim'])
#
#
#     # moving_mask方法用于生成动态遮罩，以根据方面的位置动态遮罩文本向量
#     def moving_mask(self, text_local_indices, aspect_indices, mask_len):
#         # 将文本局部索引和方面索引转移到CPU上，并将其转换为NumPy数组。
#         texts = text_local_indices.cpu().detach().numpy()
#         asps = aspect_indices.cpu().numpy()
#         # 创建一个全为1的张量作为遮罩张量
#         masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
#                                           dtype=np.float32)
#         # 对于每个样本，遍历文本和方面的索引。计算方面的长度，并找到方面的开始位置。根据遮罩长度确定遮罩的开始位置。
#         # 接着，将遮罩张量的相应位置置为0，即遮罩掉方面周围的文本部分。
#         for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
#             asp_len = np.count_nonzero(asps[asp_i]) - 2
#             try:
#                 asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
#             except:
#                 continue
#             if asp_begin >= mask_len:
#                 mask_begin = asp_begin - mask_len
#             else:
#                 mask_begin = 0
#             for i in range(mask_begin):
#                 masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
#             for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
#                 masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
#         # 将生成的遮罩张量转换为PyTorch张量，并返回至设备上
#         masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
#         return masked_text_raw_indices.to(self.opt.device)
#
#     def forward(self, inputs):
#         text_bert_indices = inputs[0]     # 从输入中获取文本BERT索引
#         bert_segments_ids = inputs[1]     # BERT片段ID
#         text_local_indices = inputs[2]    # 本地文本索引
#         aspect_indices = inputs[3]        # 方面索引
#         category = inputs[5]
#
#         # 将文本BERT索引和BERT片段ID传递给BERT模型以获取BERT的输出,并对输出进行Dropout处理。
#         bert_spc_out, _ = self.bert_spc(text_bert_indices,token_type_ids=bert_segments_ids,return_dict=False)
#         bert_spc_out = self.dropout(bert_spc_out)
#
#         # 修改
#         init_feature = bert_spc_out
#         feature, _ = self.attention(init_feature)
#         idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
#         # squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉
#         domain_embedding = self.domain_embedder(idxs).squeeze(1)
#
#         gate_input_feature = feature
#         gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
#         gate_value = self.gate(gate_input)
#
#
#         # 使用本地文本索引传递给BERT模型以获取邻域范围，并对其进行Dropout处理
#         neighboring_span, _ = self.bert_spc(text_local_indices,return_dict=False)
#         neighboring_span = self.dropout(neighboring_span)
#
#         out_list=[]
#
#         # 模型根据一定的阈值生成不同的注意力遮罩，并通过moving_mask方法获得遮罩后的文本向量。随后，将经过遮罩后的文本向量与BERT输出拼接起来，
#         # 然后通过线性层和自注意力层处理，最终通过BERT池化器和全连接层生成模型的输出。最后，通过均值池化层对输出进行池化，并返回模型的输出结果。
#         for i in range(self.opt.threshold+1):
#             masked_local_text_vec = self.moving_mask(bert_spc_out, aspect_indices,i)
#             neighboring_span = torch.mul(neighboring_span, masked_local_text_vec)
#             enhanced_text = torch.cat((neighboring_span, bert_spc_out), dim=-1)
#             mean_pool = self.linear_double(enhanced_text)
#             self_attention_out= self.bert_SA(mean_pool)
#             pooled_out = self.bert_pooler(self_attention_out)
#             dense_out = self.dense(pooled_out)
#             out_list.append(dense_out )
#
#
#         # 修改
#         shared_feature = 0
#         for i in range(self.opt.threshold+1):
#             tmp_feature = out_list[i]
#             shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))
#         # out = shared_feature.view(dense_out.shape[0],2,-1)
#         # out = shared_feature.unsqueeze(2)
#         return shared_feature
#
#
#         # out=torch.cat(out_list,dim=-1)
#         # out=out.view(dense_out.shape[0],2,-1)
#         #
#         # ensem_out=self.pool(out)
#         # ensem_out=ensem_out.squeeze(-1)
#         # return ensem_out


# =====================================================================================
import torch
import torch.nn as nn
import copy
import numpy as np

from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention
# from transformers.modeling_bert import BertPooler, BertSelfAttention

# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        # self.SA是BERT自注意力层的实例，用于处理输入序列
        self.SA = BertSelfAttention(config)
        # tanh激活函数
        self.tanh = torch.nn.Tanh()

    # forward方法定义了自注意力层的前向传播逻辑。它接收一个名为inputs的输入张量，并生成一个全零张量作为注意力遮罩。
    # 然后，调用BERT自注意力层self.SA处理输入，并将结果通过tanh激活函数进行处理后返回。
    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class MaskAttention(nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class AOAN(nn.Module):
    def __init__(self, bert, opt):
        super(AOAN, self).__init__()

        # 初始化模型的各个组件，包括BERT模型、Dropout层、自注意力层、线性层、BERT池化器和全连接层
        self.bert_spc = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.pool=nn.AvgPool1d(opt.threshold+1)

        # 修改
        self.domain_num = 20
        self.gate = nn.Sequential(nn.Linear(opt.bert_dim * 2, opt.bert_dim),
                                  nn.ReLU(),
                                  nn.Linear(opt.bert_dim, self.opt.threshold+1),
                                  nn.Softmax(dim=1))
        self.attention = MaskAttention(opt.bert_dim)
        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=opt.bert_dim)


    # moving_mask方法用于生成动态遮罩，以根据方面的位置动态遮罩文本向量
    def moving_mask(self, text_local_indices, aspect_indices, mask_len):
        # 将文本局部索引和方面索引转移到CPU上，并将其转换为NumPy数组。
        texts = text_local_indices.cpu().detach().numpy()
        asps = aspect_indices.cpu().numpy()
        # 创建一个全为1的张量作为遮罩张量
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        # 对于每个样本，遍历文本和方面的索引。计算方面的长度，并找到方面的开始位置。根据遮罩长度确定遮罩的开始位置。
        # 接着，将遮罩张量的相应位置置为0，即遮罩掉方面周围的文本部分。
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        # 将生成的遮罩张量转换为PyTorch张量，并返回至设备上
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]  # 从输入中获取文本BERT索引
        bert_segments_ids = inputs[1]  # BERT片段ID
        text_local_indices = inputs[2]  # 本地文本索引
        aspect_indices = inputs[3]  # 方面索引
        category = inputs[5]

        # print(input.shape)

        # 将文本BERT索引和BERT片段ID传递给BERT模型以获取BERT的输出,并对输出进行Dropout处理。
        bert_spc_out, _ = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids, return_dict=False)
        bert_spc_out = self.dropout(bert_spc_out)

        # 修改
        init_feature = bert_spc_out
        feature, _ = self.attention(init_feature)
        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        # squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input_feature = feature
        gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        gate_value = self.gate(gate_input)

        # 使用本地文本索引传递给BERT模型以获取邻域范围，并对其进行Dropout处理
        neighboring_span, _ = self.bert_spc(text_local_indices, return_dict=False)
        neighboring_span = self.dropout(neighboring_span)

        out_list = []

        # 模型根据一定的阈值生成不同的注意力遮罩，并通过moving_mask方法获得遮罩后的文本向量。随后，将经过遮罩后的文本向量与BERT输出拼接起来，
        # 然后通过线性层和自注意力层处理，最终通过BERT池化器和全连接层生成模型的输出。最后，通过均值池化层对输出进行池化，并返回模型的输出结果。
        for i in range(self.opt.threshold + 1):
            masked_local_text_vec = self.moving_mask(bert_spc_out, aspect_indices, i)
            neighboring_span = torch.mul(neighboring_span, masked_local_text_vec)
            enhanced_text = torch.cat((neighboring_span, bert_spc_out), dim=-1)
            mean_pool = self.linear_double(enhanced_text)
            self_attention_out = self.bert_SA(mean_pool)
            pooled_out = self.bert_pooler(self_attention_out)
            dense_out = self.dense(pooled_out)
            out_list.append(dense_out)

        # 修改
        shared_feature = 0
        for i in range(self.opt.threshold + 1):
            tmp_feature = out_list[i]
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))
        return shared_feature

