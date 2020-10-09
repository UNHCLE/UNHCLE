from data_cook_sup import CookData

# from soft_dtw import SoftDTW
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

import random
import math
import time
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
# from tslearn.metrics import dtw, dtw_path
# from numba import jit
from transformers import *
from data_cook_sup import MAX_EVENTS


use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_cuda = False
# device = "cpu"
d_model = 768
# bs = 4
sos_idx = 0
# vocab_size = 10000
input_len = 1000
output_len2 = 5
output_len1 = 9
skill_max_len = 5
t_ratio = 0.5
ngpu = 7
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

import sys
# print('__Python VERSION:', sys.version)
# print('__pyTorch VERSION:', torch.__version__)
# print('__CUDA VERSION')
# from subprocess import call
# # call(["nvcc", "--version"]) does not work
# # ! nvcc --version
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())

# print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device())
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


class CenteredBatchNorm2d(nn.BatchNorm2d):
    """It appears the only way to get a trainable model with beta (bias) but not scale (weight
     is by keeping the weight data, even though it's not used"""

    def __init__(self, channels):
        super().__init__(channels, affine=True)
        self.weight.data.fill_(1)
        self.weight.requires_grad = False


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, input_channels, output_channels=None):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.conv1_bn = CenteredBatchNorm2d(output_channels)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.relu(out, inplace=True)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv1_bn = CenteredBatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv2_bn = CenteredBatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.conv2_bn(self.conv2(out))
        out += x
        out = F.relu(out, inplace=True)
        return out


class LeelaModel(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        # 112 input channels
        self.conv_in = ConvBlock(kernel_size=3,
                               input_channels=15,
                               output_channels=channels)
        self.residual_blocks = []
        for idx in range(blocks):
            block = ResidualBlock(channels)
            self.residual_blocks.append(block)
            self.add_module('residual_block{}'.format(idx+1), block)
        self.conv_pol = ConvBlock(kernel_size=1,
                                   input_channels=channels,
                                   output_channels=32)
        self.affine_pol = nn.Linear(32*8*8, 1858)
        self.conv_val = ConvBlock(kernel_size=1,
                                 input_channels=channels,
                                 output_channels=32)
        self.affine_val_1 = nn.Linear(32*8*8, 128)
        self.affine_val_2 = nn.Linear(128, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
#             if next(self.parameters()).is_cuda:
#                 x = x.cuda()
        x = x.view(-1, 15, 8, 8)
        out = self.conv_in(x)
        for block in self.residual_blocks:
            out = block(out)
#         print(out.shape)
        # out_pol = self.conv_pol(out).view(-1, 32*8*8)
        # out_pol = self.affine_pol(out_pol)
        out_val = self.conv_val(out).view(-1, 32*8*8)
        out_val = F.relu(self.affine_val_1(out_val), inplace=True)
        # out_val = self.affine_val_2(out_val).tanh()
        # return out_pol, out_val
        return out_val


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DecoderRNNBoard(nn.Module):
    
    def __init__(self, output_size, hidden_size, n_layers=1):
        super(DecoderRNNBoard, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.hid2out = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(output_size, hidden_size, self.n_layers)#output size = input size
        

    def forward(self, input, hidden):
#         print('input', input.shape)
#         print('hidden', hidden.shape)
        output, hn = self.gru(input, hidden)#output = (1, batch, hidden), output = (1, batch, hidden)
        output = self.hid2out(output)#reshaping output
        return output, hn


class Net(nn.Module):
    def __init__(self, input_channel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 30, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 16, 2)
        self.adapt = nn.AdaptiveAvgPool2d((1,16))
        self.fc1 = nn.Linear(144, 84)
        #self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        #x = x.double()
        batch_size = x.size()[0]
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        #x = self.adapt(x)

        x = x.view(-1, 144)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = x.view(batch_size, -1)
        #print(x.size())
        return x

class CommentEncodedtoSkills(nn.Module):
    """docstring for CommenttoSkill"""
    def __init__(self, input_len, output_len, d_model, vocab_size):
        super(CommentEncodedtoSkills, self).__init__()
        #self.arg = arg
        self.output_len = output_len
        self.input_len = input_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8).to(device)

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                        num_layers=8).to(device)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8).to(device)

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,
                                        num_layers=8).to(device)

        self.decoder_emb = nn.Embedding(self.vocab_size, self.d_model)

#         self.predictor = nn.Linear(self.input_len, self.output_len)
        self.predictor = nn.Linear(self.d_model, self.vocab_size)
        self.soft = nn.Softmax(dim=0)
        self.probHead = nn.Linear(self.d_model, 1)

    def forward(self, x):
        #x = torch.randn(bs, input_len, d_model).to(device)
        bs = x.size()[0]
        pos_x = x.transpose(0, 1)
#         x_enc = self.decoder_emb(x)

#         pos_x = self.pos_encoder(x_enc.transpose(0, 1)) 
#         print(pos_x.shape)
        encoder_output = self.encoder(pos_x).to(device)  # (input_len, bs, d_model)
        
        #print('encoder_output', encoder_output.shape)
        # initialized the input of the decoder with sos_idx (start of sentence token idx)
        output = torch.ones(self.output_len, bs).long().to(device)*sos_idx
        predictions = torch.zeros(self.output_len, bs, self.vocab_size).to(device)
        # tgt_emb = self.decoder_emb(output.clone()[:output_len-1].transpose(0, 1)).transpose(0, 1)
        # decoder_output = self.decoder(tgt = tgt_emb, memory = encoder_output)
        for t in range(self.output_len - 1):
            tgt_emb = self.decoder_emb(output[:t+1].transpose(0,1).clone()).transpose(0, 1)
#             print(tgt_emb.shape)
            
            tgt_emb = self.pos_encoder(tgt_emb)
#             print('tgt_emb', tgt_emb.shape)
            # tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(device)
#             print(tgt_emb.shape)
            decoder_output = self.decoder(tgt=tgt_emb, memory=encoder_output)
            
            prediction = self.predictor(decoder_output[-1])
            # predictions - [t, bs, vocab]
            predictions[t] = prediction
            
#             if random.random() < 0.4:
#                 return decoder_output
            one_hot_idx = prediction.argmax(1)

            # output  = [output len, batch size]
            output[t+1] = one_hot_idx            
            
        return decoder_output.transpose(0, 1)
    


class BoardEncodedtoSkills(nn.Module):
    """docstring for BoardtoSkill"""
    def __init__(self, input_len, output_len, d_model, vocab_size):
        super(BoardEncodedtoSkills, self).__init__()
        #self.arg = arg
        self.output_len = output_len
        self.input_len = input_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8).to(device)

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                        num_layers=8).to(device)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8).to(device)

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,
                                        num_layers=8).to(device)

        self.decoder_emb = nn.Embedding(self.vocab_size, self.d_model)

#         self.predictor = nn.Linear(self.input_len, self.output_len)
        self.predictor = nn.Linear(self.d_model, self.vocab_size)
        self.soft = nn.Softmax(dim=0)
        self.probHead = nn.Linear(self.d_model, 1)

    def forward(self, x):
        #x = torch.randn(bs, input_len, d_model).to(device)
        bs = x.size()[0]
#         print('input shape',x.shape)
        pos_x = self.pos_encoder(x.transpose(0, 1)) 
#         print('pos_input shape', pos_x.shape)
#         print(pos_x.shape)
        encoder_output = self.encoder(pos_x).to(device)  # (input_len, bs, d_model)
#         print('encoder shape', encoder_output.shape)
#         skill_output = self.predictor(encoder_output.transpose(0, 2))
#         skill_output = torch.transpose(skill_output, 0, 2)
        
#         return skill_output
#         print(encoder_output.shape)
        
        #print('encoder_output', encoder_output.shape)
        # initialized the input of the decoder with sos_idx (start of sentence token idx)
        output = torch.ones(self.output_len, bs).long().to(device)*sos_idx
        predictions = torch.zeros(self.output_len, bs, self.vocab_size).to(device)
#         tgt_emb = self.decoder_emb(output.clone()[:output_len-1].transpose(0, 1)).transpose(0, 1)
# #         print('tgt_emb',tgt_emb.shape)
#         decoder_output = self.decoder(tgt = tgt_emb, memory = encoder_output)
#         print('decoder', decoder_output.shape)
        for t in range(self.output_len - 1):
            tgt_emb = self.decoder_emb(output[:t+1].transpose(0,1).clone()).transpose(0, 1)
#             print(tgt_emb.shape)
            
            tgt_emb = self.pos_encoder(tgt_emb)
#             print('tgt_emb', tgt_emb.shape)
            tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(device)
#             print(tgt_emb.shape)
            decoder_output = self.decoder(tgt=tgt_emb, memory=encoder_output)
            
            prediction = self.predictor(decoder_output[-1])
            # predictions - [t, bs, vocab]
            predictions[t] = prediction
            
#             if random.random() < 0.4:
#                 return decoder_output
            one_hot_idx = prediction.argmax(1)

            # output  = [output len, batch size]
            output[t+1] = one_hot_idx            
#         print('decoder_output', decoder_output.shape)
        return decoder_output.transpose(0, 1)
    

class WeightMultInverse(nn.Module):
    def __init__(self, in_features, out_features):
        super(WeightMultInverse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_features)
        #self.m = nn.LayerNorm()
        

        self.matrix = nn.Linear(in_features, out_features)
    
    def forward(self, x):
#         if use_cuda:
#             x = x.to(device)
#         result = self.matrix(x)
#         if use_cuda:
#             result = result.cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
#         if use_cuda:
#             return x.to(device)
#         else:
        return x

class DecoderRNNComment(nn.Module):
    
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNNComment, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        
#         print('input', input.shape)
#         print('hidden', hidden.shape)
        hidden = torch.transpose(hidden, 0, 1)
        bs = input.shape[0]
#         print(input.shape)
        output = self.embedding(input).view(1, bs, self.hidden_size)
        # output = input.view(1, bs, self.hidden_size)
        for i in range(self.n_layers):
            output = F.relu(output)
            # output = output.cuda()
            # hidden = hidden.cuda()
            output, hidden = self.gru(output, hidden)
            
            # print(self.out(output[0]))
            # print(self.out(output[0]).shape)
        output = self.softmax(self.out(output[0]))
        return output, hidden.transpose(0, 1)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def train(boardSeq, comment, bert_emb, pred_masks, time_gt , w_hidden2hidden, seg_enc_dec, w_hidden2hidden_opt, seg_enc_dec_opt , criteriaSeg):

    w_hidden2hidden_opt.zero_grad()
    seg_enc_dec_opt.zero_grad()
    # boardenc2skill1_opt.zero_grad()
    # boardenc2skill2_opt.zero_grad()
    # decoderBoard1_opt.zero_grad()
    # decoderBoard2_opt.zero_grad()
#     w_hidden2board_opt.zero_grad()
#     decoderComment_opt.zero_grad()
    # comment2skill_opt.zero_grad()
        
    lossBoard = 0
    lossComment = 0
    loss = 0
    #move = (batch, seq_len, 5)
    ##################Board Encoding ############################################################################
    #boardSeq = torch.randn(bs, input_len, 8, 8, 15).to(device)
#     print('board1',boardSeq1.shape)#batch, 5000, 512
    
    
    boardSeq = boardSeq.permute(1,0,2) #batch, 500, 512-> 500, batch, 512
    boardSeq = boardSeq.to(dtype = torch.float32)
    seq_len = boardSeq.size()[0]
    batch_size = boardSeq.size()[1]

#     print('boardseq', boardSeq.shape)
#     del boardSeq1
#     del temp
#     board_seq_encoded = torch.zeros(seq_len, batch_size, 128)
#     board_seq_encoded = board_seq_encoded.to(dtype = torch.float64)
#     if use_cuda:
#         board_seq_encoded = board_seq_encoded.cuda()

#     for i in range(seq_len):
#         board_seq_encoded[i] = cnn1(boardSeq[i].view(batch_size, 15, 8, 8)).view(batch_size, 128)
#     seq_len = boardSeq.size()[0]
#     batch_size = boardSeq.size()[1]
#     print(boardSeq.shape)
    # print("start")
    board_seq_encoded_enlarged = torch.zeros(seq_len, batch_size, d_model)
#     if use_cuda:
#         board_seq_encoded_enlarged = board_seq_encoded_enlarged.to(device)
    for i in range(seq_len):
        board_seq_encoded_enlarged[i] = w_hidden2hidden(boardSeq[i])

    # print("hidden2hidden done!")
    board_seq_encoded_enlarged = board_seq_encoded_enlarged.permute(1,0,2) #(input_len, bs, 768)-> (bs, input_len, 768)
#     print('board_seq_encoded_enlarged', board_seq_encoded_enlarged.shape)
    time_preds = seg_enc_dec(board_seq_encoded_enlarged)
    #x = torch.randn(bs, input_len, d_model).to(device)
    # print(time_gt)
    loss = criteriaSeg(time_preds.squeeze(-1), time_gt)
    # print(time_preds.size(), time_gt.size(), loss.size())
    # loss = torch.mean(torch.sum(loss, dim=1))

    if torch.isnan(loss):
        pass
    else:
        loss.backward()
#         cnn1_opt.step()
        w_hidden2hidden_opt.step()
        seg_enc_dec_opt.step()
#         w_hidden2board_opt.step()
        # decoderComment_opt.step()
        # comment2skill_opt.step()
#     print(lossBoard, lossComment)
    # print(loss)
    return float(loss.item())

class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if not isinstance(criterion, torch.nn.Module):
            criterion = ModularizedFunction(criterion)
        self.criterion = torch.nn.DataParallel(criterion)
    def forward(self, *args, **kwargs):
        return self.criterion(*args, **kwargs).mean()


def trainIters(w_hidden2hidden, seg_enc_dec, epochs, train_loader, test_loader, learning_rate=0.001):
    start = time.time()

    # cnn1_opt = None#optim.Adam(filter(lambda x: x.requires_grad, cnn1.parameters()),
#                                   lr=learning_rate)
    w_hidden2hidden_opt = optim.Adam(filter(lambda x: x.requires_grad, w_hidden2hidden.parameters()),
                                  lr=learning_rate)
    seg_enc_dec_opt = optim.Adam(filter(lambda x: x.requires_grad, seg_enc_dec.parameters()),
                                     lr=learning_rate)
#     boardenc2skill1_opt = optim.Adam(filter(lambda x: x.requires_grad, boardenc2skill1.parameters()),
#                                   lr=learning_rate)
#     boardenc2skill2_opt = optim.Adam(filter(lambda x: x.requires_grad, boardenc2skill2.parameters()),
#                                   lr=learning_rate)
#     decoderBoard1_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderBoard1.parameters()),
#                                   lr=learning_rate)
#     decoderBoard2_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderBoard2.parameters()),
#                                   lr=learning_rate)
#     w_hidden2board_opt = None#optim.Adam(filter(lambda x: x.requires_grad, w_hidden2board.parameters()),
# #                                   lr=learning_rate)
#     decoderComment_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderComment.parameters()),
#                                   lr=learning_rate)
#     comment2skill_opt = None#optim.Adam(filter(lambda x: x.requires_grad, comment2skill.parameters()),
# #                                   lr=learning_rate)
    
    # criteriaBoard = nn.BCEWithLogitsLoss()
    # criteriaComment = nn.NLLLoss()
    critera_segment = nn.MSELoss()
    if torch.cuda.device_count() > 1:
        # criteriaBoard = CriterionParallel(criteriaBoard)
        # criteriaComment = CriterionParallel(criteriaComment)
        criteraSegment = CriterionParallel(critera_segment)

#     print("Initialised optimisers")
#     # before = list(cnn.parameters())[0].clone()
#     for epoch in range(epochs):
#         print_loss_total = 0.
#         count = 0
#         for entry in train_loader:
#             loss = train(entry[0], entry[1], entry[2], entry[4], entry[5],  w_hidden2hidden, seg_enc_dec, w_hidden2hidden_opt, seg_enc_dec_opt,
#                          criteraSegment)
#             print_loss_total = print_loss_total + loss
#             count += 1
#             if count % 10 == 0:
#                 print(entry[4][0], entry[5][0])
#                 print(count, print_loss_total/count)
#
# # # # # # #         after = list(cnn.parameters())[0].clone()
# # # # # # #         for i in range(len(before)):
# # # # # # #             print(torch.equal(before[i].data, after[i].data))
#
#         print('epochs: '+str(epoch))
#         print('cumm_loss : ', str(float(print_loss_total)/count))
#         print('total loss: '+str(print_loss_total),'\n')
# #         torch.save(cnn1, "./cnn1_siam_rev_lowLR_random2.pth")
#         torch.save(w_hidden2hidden, "./models/w_hidden2hidden_cook_maximalSkillDTW_200_16_no_" + str(epoch) + ".pth")
#         torch.save(seg_enc_dec, "./models/seg_enc_dec_cook_maximalSkillDTW_200_16_no_" + str(epoch) + ".pth")
#         torch.save(boardenc2skill1, "./models/boardenc2skill1_cook_maximalSkillDTW_200_16_no.pth")
#         torch.save(boardenc2skill2, "./models/boardenc2skill2_cook_maximalSkillDTW_200_16_no.pth")
#         torch.save(decoderBoard1, "./models/decoderBoard1_cook_maximalSkillDTW_200_16_no.pth")
#         torch.save(decoderBoard2, "./models/decoderBoard2_cook_maximalSkillDTW_200_16_no.pth")
# #         torch.save(w_hidden2board, "./w_hidden2board_siam_rev_lowLR_random2.pth")
#         torch.save(decoderComment, "./models/decoderComment_cook_maximalSkillDTW_200_16_no.pth")
        # torch.save(comment2skill, "/trainman-mount/trainman-scratch-trainman2-job-1c0042f5-b217-4d14-b5b9-08eb9544d789/models/comment2skill_siam_cook.pth")
        print("Models saved for epoch")
    tot1 = 0
    # tot2 = 0
    cnt = 0
    t_num1 = t_den1 = 0
    for entry in train_loader:
        al1, iou11, iou12 = evaluate(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], w_hidden2hidden, seg_enc_dec)
        # TODO : remove this
        cnt += 1
        # if cnt > 20:
        #     break
        # continue
        tot1 += al1

        # t_num1 += iou11
        # t_num2 += iou21
        # t_den1 += iou12
        # t_den2 += iou22
        t_num1 += iou11/iou12
        # t_num2 += iou21/iou22

        print(cnt)
        # break
    print("Alignment score 1:", tot1/cnt)
    # print("Alignment score 2:", tot2/cnt)
    print("IoU score 1:", t_num1/cnt)
    # print("IoU score 2:", t_num2/cnt)
        

def evaluate(boardSeq, comment, bert_emb, intvals, masks, time_gt, w_hidden2hidden, seg_enc_dec):

    boardSeq = boardSeq.permute(1,0,2) #batch, 500, 512-> 500, batch, 512
    boardSeq = boardSeq.to(dtype = torch.float32)
    seq_len = boardSeq.size()[0]
    batch_size = boardSeq.size()[1]

#     print('boardseq', boardSeq.shape)
#     del boardSeq1
#     del temp
#     board_seq_encoded = torch.zeros(seq_len, batch_size, 128)
#     board_seq_encoded = board_seq_encoded.to(dtype = torch.float64)
#     if use_cuda:
#         board_seq_encoded = board_seq_encoded.cuda()

#     for i in range(seq_len):
#         board_seq_encoded[i] = cnn1(boardSeq[i].view(batch_size, 15, 8, 8)).view(batch_size, 128)
#     seq_len = boardSeq.size()[0]
#     batch_size = boardSeq.size()[1]
#     print(boardSeq.shape)
    # print("start")
    board_seq_encoded_enlarged = torch.zeros(seq_len, batch_size, d_model)
#     if use_cuda:
#         board_seq_encoded_enlarged = board_seq_encoded_enlarged.to(device)
    for i in range(seq_len):
        board_seq_encoded_enlarged[i] = w_hidden2hidden(boardSeq[i])

    # print("hidden2hidden done!")
    board_seq_encoded_enlarged = board_seq_encoded_enlarged.permute(1,0,2) #(input_len, bs, 768)-> (bs, input_len, 768)
#     print('board_seq_encoded_enlarged', board_seq_encoded_enlarged.shape)

    time_preds = seg_enc_dec(board_seq_encoded_enlarged)
    time_preds_cpu = time_preds.squeeze(-1).detach().cpu().numpy()[0].tolist()



    truncated_time_preds_cpu = []
    for index,curr_time in enumerate(time_preds_cpu):
        if curr_time <= 0.:
            break
        if index > 0:
            if curr_time < time_preds_cpu[index-1]:
                break
        truncated_time_preds_cpu.append(float(curr_time))


    intvals = intvals.detach().cpu().numpy()

    intvals = intvals[0]
    # print(intvals)
    prefix = np.zeros(len(intvals))
    # print(intvals)
    tot_time = sum(intvals)
    prefix[0] = float(intvals[0])
    for i in range(1, len(intvals)):
        prefix[i] = float(prefix[i-1] + intvals[i])
    prefix = prefix / 500
    # multi = tot_time / 200.0
    # multi1 = tot_time / 200.0

    prefix = np.unique(prefix)
    pred = np.unique(np.asarray(truncated_time_preds_cpu))
    # TODO : remove this
    # print(prefix)
    # print()
    # print(pred)
    # print()
    # print()
    # return None, None, None

    # print(prefix)
    # print(pred)
    # print(pred1)

    paths1, dists1 = dtw_path(prefix, pred)
    # paths2, dists2 = dtw_path(prefix, pred1)
    # print(paths1)
    # print(paths2)
    curr = 0
    tot_den = 0
    tot_num = 0
    prev_d = -1
    prev_n = -1
    track = 0
    for nums in paths1:
        fi = nums[0]
        se = nums[1]
        x2 = prefix[fi]
        y2 = pred[se]
        if fi == 0:
            x1 = 0.0
        else:
            x1 = prefix[fi-1]
        if se == 0:
            y1 = 0.0
        else:
            y1 = pred[se-1]
        tot_num += max(0.0, min(x2,y2) - max(x1,y1))
        if prev_d != x2 and prev_n != x1:
            tot_den += max(0.0, max(x2,y2) - min(x1,y1))
            prev_d = x2
            prev_n = x1

    return dists1, tot_num, tot_den
    # print(path1)
    print("----------------------------------")

        

class VideoEncDec(nn.Module):
    def __init__(self, enc_lstm_hidden_size=d_model):
        super(VideoEncDec, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        # use multiInverse to bring the input to the correct dimension
        self.enc_lstm_hidden_size = enc_lstm_hidden_size
        self.dec_lstm_hidden_size = enc_lstm_hidden_size
        self.enc_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.dec_gru = nn.GRU(1 + d_model, d_model, batch_first=True)
        self.attn_query = nn.Linear(d_model, d_model)
        self.attn_memory = nn.Linear(d_model, d_model)
        self.attn_proj = nn.Linear(d_model, 1)
        self.time_pred = nn.Linear(d_model, 1)
        self.max_time_decode_len = MAX_EVENTS
    def forward(self, x):
        bs, input_len, feat_dim = x.size()
        # put x on the device as well, model and model optimizers as well
        pos_inp = self.pos_encoder(x.transpose(0,1)).transpose(0,1).to(device)
        enc_hidden_start = torch.zeros(1, bs, self.enc_lstm_hidden_size).to(device)
        enc_outputs, _ = self.enc_gru(pos_inp, enc_hidden_start)
        # enc_outputs is BS x len x feat_dim
        dec_hidden = torch.zeros(1, bs, self.dec_lstm_hidden_size).to(device)
        last_output = torch.zeros(bs, 1).to(device)
        decoded_steps = []
        for i in range(self.max_time_decode_len):
            attn_logits = self.attn_proj(
                torch.tanh(self.attn_memory(enc_outputs) + self.attn_query(dec_hidden[0]).unsqueeze(1))).squeeze(-1)
            attn_weights = torch.softmax(attn_logits,dim=1).unsqueeze(-1)
            context_vector = torch.sum(attn_weights * enc_outputs, dim=1)
            dec_input = torch.cat([last_output, context_vector],dim=-1)
            curr_dec_gru_output, dec_hidden = self.dec_gru(dec_input.unsqueeze(1), dec_hidden)
            time_preds = self.time_pred(curr_dec_gru_output.squeeze(1))
            decoded_steps.append(time_preds)
            last_output = time_preds
        return torch.stack(decoded_steps, dim=1)

train_data = CookData(data_path="./feat_comment_interval_200_validation.npy")
dataload = DataLoader(train_data, batch_size=1, num_workers=64, pin_memory=True)
print('Data loaded')

# TRAIN
# cnn1 = torchvision.models.resnet18()
# cnn1 = torch.load('./AE_enc_newloss-Copy1.pth')
# cnn1 = cnn1.eval()
# cnn1.conv1 = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3,bias=False)
# cnn1 = Net(15)
# cnn1 = LeelaModel(100, 30)
# cnn1.double()

# w_hidden2hidden.double()
# w_hidden2hidden = WeightMultInverse(512, d_model)
# seg_enc_dec = VideoEncDec()
# cnn1 = None
# w_hidden2board = None
# boardenc2skill1 = BoardEncodedtoSkills(input_len, output_len1, d_model, d_model)
# boardenc2skill2 = BoardEncodedtoSkills(output_len1, output_len2, d_model, d_model)

# comment2skill = None#CommentEncodedtoSkills(input_len, output_len, d_model, train_data.n_words).to(device)

# decoderBoard1 = DecoderRNNBoard(768, 768, 1)
# decoderBoard2 = DecoderRNNBoard(512, 768, 1)

# decoderBoard1.double()
# decoderBoard2.double()
# w_hidden2hidden.double()
# decoderBoard.double()
# w_hidden2board = WeightMultInverse(128, 15*8*8)
# decoderComment = DecoderRNNComment(d_model, train_data.n_words, n_layers=1)


# EVAL

# cnn1 = None#torch.load('./cnn1_siam_rev_lowLR_random1.pth')
w_hidden2hidden = torch.load('./models_sup/w_hidden2hidden_cook_maximalSkillDTW_200_16_no_50.pth')
seg_enc_dec = torch.load('./models_sup/seg_enc_dec_cook_maximalSkillDTW_200_16_no_50.pth')
w_hidden2hidden = w_hidden2hidden.eval()
seg_enc_dec = seg_enc_dec.eval()

# boardenc2skill1 = torch.load('./models/boardenc2skill1_cook_maximalSkillDTW_200_16.pth')
# boardenc2skill2 = torch.load('./models/boardenc2skill2_cook_maximalSkillDTW_200_16.pth')
#
# decoderBoard1 = torch.load('./models/decoderBoard1_cook_maximalSkillDTW_200_16.pth')
# decoderBoard2 = torch.load('./models/decoderBoard2_cook_maximalSkillDTW_200_16.pth')
# w_hidden2board = None#torch.load('./w_hidden2board_siam_rev_lowLR_random1.pth')
# decoderComment = torch.load('./models/decoderComment_cook_maximalSkillDTW_200_16.pth')
# comment2skill = None#torch.load('./comment2skill_siam_cook.pth')
# # cnn1 = cnn1.eval()
# w_hidden2hidden = w_hidden2hidden.eval()
# boardenc2skill1 = boardenc2skill1.eval()
# boardenc2skill2 = boardenc2skill2.eval()
# # w_hidden2board = w_hidden2board.eval()
# decoderComment = decoderComment.eval()
# decoderBoard1 = decoderBoard1.eval()
# decoderBoard2 = decoderBoard2.eval()
# # comment2skill = comment2skill.eval()
print("Let's use", torch.cuda.device_count(), "GPUs!")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    w_hidden2hidden = nn.DataParallel(w_hidden2hidden)
    seg_enc_dec = nn.DataParallel(seg_enc_dec)
    # boardenc2skill1 = nn.DataParallel(boardenc2skill1)
    # decoderBoard1 = nn.DataParallel(decoderBoard1)
    # boardenc2skill2 = nn.DataParallel(boardenc2skill2)
    # decoderBoard2 = nn.DataParallel(decoderBoard2)
    # decoderComment = nn.DataParallel(decoderComment)


    # comment2skill = nn.DataParallel(comment2skill)
if use_cuda:
#     cnn1 = cnn1.cuda()
    w_hidden2hidden = w_hidden2hidden.cuda()
    seg_enc_dec = seg_enc_dec.cuda()
#     boardenc2skill1 = boardenc2skill1.cuda()
#     decoderBoard1 = decoderBoard1.cuda()
#     boardenc2skill2 = boardenc2skill2.cuda()
#     decoderBoard2 = decoderBoard2.cuda()
# #     w_hidden2board = w_hidden2board.cuda()
#     decoderComment = decoderComment.cuda()
    # comment2skill = comment2skill.cuda()

#     w_hidden2hidden = nn.DataParallel(w_hidden2hidden)
#     boardenc2skill = nn.DataParallel(boardenc2skill)
#     decoderBoard = nn.DataParallel(decoderBoard)
#     decoderComment = nn.DataParallel(decoderComment)
#     comment2skill = nn.DataParallel(comment2skill)

# print(len(train_data.word2idx))
print("Training starts")
trainIters(w_hidden2hidden, seg_enc_dec, 100, dataload, dataload)


#w = nn.Linear(768, 84)
#decoder_input = w(output)
#print('decoder_input', decoder_input.shape)



# encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4).to(device)

# encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
#                                 num_layers=6).to(device)

# decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4).to(device)

# decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
#                                 num_layers=6).to(device)

# decoder_emb = nn.Embedding(vocab_size, d_model)

# predictor = nn.Linear(d_model, vocab_size)

# # for a single batch x
# x = torch.randn(bs, input_len, d_model).to(device)
# encoder_output = encoder(x)  # (bs, input_len, d_model)
# print('encoder_output', encoder_output.shape)
# # initialized the input of the decoder with sos_idx (start of sentence token idx)
# output = torch.ones(bs, output_len).long().to(device)*sos_idx
# for t in range(1, output_len):
#     tgt_emb = decoder_emb(output[:, :t]).transpose(0, 1)
#     print('tgt_emb', tgt_emb.shape)
#     tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(
#         t).to(device).transpose(0, 1)
#     print('tgt_mask', tgt_mask.shape)
#     decoder_output = decoder(tgt=tgt_emb,
#                              memory=encoder_output,
#                              tgt_mask=tgt_mask)
#     print('decoder_output', decoder_output.shape)
#     pred_proba_t = predictor(decoder_output)[-1, :, :]
#     output_t = pred_proba_t.data.topk(1)[1].squeeze()
#     output[:, t] = output_t
