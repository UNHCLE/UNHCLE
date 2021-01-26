from data_cook import CookData

from soft_dtw import SoftDTW
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
from tslearn.metrics import dtw, dtw_path
from numba import jit
from transformers import *

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

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


# norm = torch.normal(0,1,size=(1,))
# # norm = nn.DataParallel(norm)
# if use_cuda:
#     norm = norm.cuda()

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
    
    
# class BoardEncodedtoSkills(nn.Module):
#     """docstring for BoardtoSkill"""
#     def __init__(self, input_len, output_len, d_model, vocab_size):
#         super(BoardEncodedtoSkills, self).__init__()
#         #self.arg = arg
#         self.output_len = output_len
#         self.input_len = input_len
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4).to(device)

#         self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
#                                         num_layers=4).to(device)

#         self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=4).to(device)

#         self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,
#                                         num_layers=4).to(device)

#         self.decoder_emb = nn.Embedding(self.vocab_size, self.d_model)

# #         self.predictor = nn.Linear(self.input_len, self.output_len)
#         self.predictor = nn.Linear(self.d_model, self.vocab_size)
#         self.soft = nn.Softmax(dim=0)
#         self.mean = nn.Linear(self.vocab_size, 1)
#         self.var = nn.Linear(self.vocab_size, 1)
    
#     def forward(self, x):
#         #x = torch.randn(bs, input_len, d_model).to(device)
#         bs = x.size()[0]
#         # print('input shape',x.shape)
#         pos_x = self.pos_encoder(x.transpose(0, 1)) 
# #         print('pos_input shape', pos_x.shape)
# #         print(pos_x.shape)
#         encoder_output = self.encoder(pos_x).to(device)  # (input_len, bs, d_model)
# #         print('encoder shape', encoder_output.shape)
# #         skill_output = self.predictor(encoder_output.transpose(0, 2))
# #         skill_output = torch.transpose(skill_output, 0, 2)
        
# #         return skill_output
# #         print(encoder_output.shape)
        
#         #print('encoder_output', encoder_output.shape)
#         # initialized the input of the decoder with sos_idx (start of sentence token idx)
#         output = torch.ones(self.output_len, bs).long().to(device)*sos_idx
#         predictions = torch.zeros(self.output_len, bs, self.vocab_size).to(device)
#         # norm = torch.normal(0, 1, size=(1, )).to(device)
# #         tgt_emb = self.decoder_emb(output.clone()[:output_len-1].transpose(0, 1)).transpose(0, 1)
# # #         print('tgt_emb',tgt_emb.shape)
# #         decoder_output = self.decoder(tgt = tgt_emb, memory = encoder_output)
# #         print('decoder', decoder_output.shape)
#         for t in range(self.output_len - 1):
#             tgt_emb = self.decoder_emb(output[:t+1].transpose(0,1).clone()).transpose(0, 1)
# #             print(tgt_emb.shape)
            
#             tgt_emb = self.pos_encoder(tgt_emb)
# #             print('tgt_emb', tgt_emb.shape)
#             # tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(device)
# #             print(tgt_emb.shape)
#             decoder_output = self.decoder(tgt=tgt_emb, memory=encoder_output)
            
#             prediction = self.predictor(decoder_output[-1])
#             # predictions - [t, bs, vocab]
#             predictions[t] = prediction
            
# #             if random.random() < 0.4:
# #                 return decoder_output
#             # one_hot_idx = prediction.argmax(1)
#             # print(one_hot_idx.shape)
#             mean_skill = self.mean(prediction)
#             var_skill = torch.exp(self.var(prediction))
#             # print(norm.shape)
#             # print(decoder_output.shape)
#             # print(var_skill.shape)
           
#             # # print(type(self.normal))
#             one_hot_idx = mean_skill
#             # if random.random() > 0.5: 
#             one_hot_idx += (var_skill*np.random.normal())
#             # else:
#             #     one_hot_idx -= (var_skill*0.5)

#             # output  = [output len, batch size]
#             output[t+1] = one_hot_idx.squeeze()        
# #         print('decoder_output', decoder_output.shape)
#         decoder_output = torch.transpose(decoder_output, 0, 1)
#         # print(decoder_output.shape)
#         return decoder_output


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

        # self.decoder_emb = nn.Embedding(self.vocab_size, self.d_model)

#         self.predictor = nn.Linear(self.input_len, self.output_len)
        # self.predictor = nn.Linear(self.d_model, self.vocab_size)
        # self.soft = nn.Softmax(dim=0)
        # self.probHead = nn.Linear(self.d_model, 1)
        self.mean = nn.Linear(self.d_model, self.d_model)
        self.var = nn.Linear(self.d_model, self.d_model)

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
        predictions = torch.ones(1, bs, self.d_model).to(device)
#         tgt_emb = self.decoder_emb(output.clone()[:output_len-1].transpose(0, 1)).transpose(0, 1)
# #         print('tgt_emb',tgt_emb.shape)
#         decoder_output = self.decoder(tgt = tgt_emb, memory = encoder_output)
#         print('decoder', decoder_output.shape)
        for t in range(self.output_len - 1):
            # tgt_emb = self.decoder_emb(output[:t+1].transpose(0,1).clone()).transpose(0, 1)
            # print(tgt_emb.shape)
            
            tgt_emb = self.pos_encoder(predictions)
#             print('tgt_emb', tgt_emb.shape)
            # tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(device)
#             print(tgt_emb.shape)
            decoder_output = self.decoder(tgt=tgt_emb, memory=encoder_output)
            mean_skill = self.mean(decoder_output[-1]) 
            var_skill = self.var(decoder_output[-1])
            
            sampled_skill = mean_skill + var_skill * np.random.normal()
            # print(sampled_skill.shape)
            # sampled_skill = sampled_skill.squeeze()
            predictions = torch.cat((predictions, sampled_skill.unsqueeze(0)), 0)
            
            # print(decoder_output.shape)
#             prediction = self.predictor(decoder_output[-1])
#             # predictions - [t, bs, vocab]
#             predictions[t] = prediction
            
# #             if random.random() < 0.4:
# #                 return decoder_output
#             one_hot_idx = prediction.argmax(1)

#             # output  = [output len, batch size]
#             output[t+1] = one_hot_idx            
        # print('decoder_output', decoder_output.shape)
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


def train(boardSeq, comment, bert_emb, cnn1, w_hidden2hidden, boardenc2skill1, boardenc2skill2, comment2skill, decoderBoard1, decoderBoard2, w_hidden2board, decoderComment,  
cnn1_opt, w_hidden2hidden_opt, boardenc2skill1_opt, boardenc2skill2_opt, comment2skill_opt, decoderBoard1_opt, decoderBoard2_opt, w_hidden2board_opt, decoderComment_opt, 
criteriaBoard, criteriaComment,
):


#     cnn1_opt.zero_grad()
    w_hidden2hidden_opt.zero_grad()
    boardenc2skill1_opt.zero_grad()
    boardenc2skill2_opt.zero_grad()
    decoderBoard1_opt.zero_grad()
    decoderBoard2_opt.zero_grad() 
#     w_hidden2board_opt.zero_grad()
    decoderComment_opt.zero_grad()
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
    # print('board_seq_encoded_enlarged', board_seq_encoded_enlarged.shape)
    
    #x = torch.randn(bs, input_len, d_model).to(device)

    # norm = torch.normal(0, 1, size=(batch_size, 1))
    skills = boardenc2skill1(board_seq_encoded_enlarged)
    # print("1 skill")
    skills2 = boardenc2skill2(skills)
    # print("2 skill")
    skill_len = skills.size()[1]
    # print('skill1', skills.shape)
    # print('skill2', skills2.shape)

#     print(skill_len)
    board_per_skill = 25 #max(1, int(10/skill_len))
#     print(skills.shape)
    

    
    #################Comment Encoding ##############################
    #comment = torch.randn(batch_size, 12)

    max_length = comment.size()[1]

    
    ###################### SKILLS GENERATION 8#######################

    decoder_input = torch.ones(1, batch_size, 768)# <SOS_index>
    skills2 = skills2.to(dtype = torch.float32)
    decoder_input = decoder_input.to(dtype = torch.float32)
#     print(type(skills2), type(skills2))
#     decoder_inputB1 = board_seq_encoded.clone()[0, :, :].view(1, batch_size, 128) #game start positions as start token
    # if use_cuda:
    #     decoder_input = decoder_input.cuda()
#     print('decoder_input', decoder_input.shape)
    
    
    #print('decoder_hidden', decoder_hidden.shape)

    # output, hn = decoder(decoder_input, decoder_hidden)

    # print('output', output.shape)#1,batch,84
    # print('hn', hn.shape)#1,4,768
    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_skills = torch.ones(16, batch_size, 768)
#     print(seq_len)
    if use_cuda:
        generated_skills = generated_skills.cuda()
    for j in range(skills2.size()[1]):
        decoder_hidden = skills2[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
        # if use_cuda:
        #     decoder_hidden = decoder_hidden.cuda()
        for i in range(4):
            output, decoder_hidden = decoderBoard1(decoder_input, decoder_hidden)
#             if use_teacher_forcing and j*board_per_skill+i+1 < 10:
#                 decoder_inputB = board_seq_encoded.clone()[i+1+j*board_per_skill,:,:].view(1, batch_size, 128)
#             else:
            decoder_input = output
            #print(output.shape)
            generated_skills[count] = output.view(batch_size, 768)
            count=count + 1
#             if count == 10:
#                 break
#         if count == 10:
#             break
    generated_skills = generated_skills.permute(1,0,2)
    #######################################################BOARD GENERATION FROM 8 ############################\
    #decoder_input = torch.ones(1, bs, 84) # <SOS_index>
    decoder_inputB = boardSeq.clone()[0, :, :].view(1, batch_size, 512) #game start positions as start token
#     if use_cuda:
#         decoder_inputB = decoder_inputB.to(device)
    #print('decoder_input', decoder_input.shape)
    
    
    #print('decoder_hidden', decoder_hidden.shape)

    # output, hn = decoder(decoder_input, decoder_hidden)

    # print('output', output.shape)#1,batch,84
    # print('hn', hn.shape)#1,4,768
    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_boards = torch.ones(board_per_skill * 16, batch_size, 512)
#     print(seq_len)
#     if use_cuda:
#         generated_boards = generated_boards.to(device)
    for j in range(generated_skills.size()[1]):
        decoder_hiddenB = generated_skills.clone()[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
#         if use_cuda:
#             decoder_hiddenB = decoder_hiddenB.to(device)
        for i in range(board_per_skill):
            outputB, decoder_hiddenB = decoderBoard2(decoder_inputB, decoder_hiddenB)
            if use_teacher_forcing and j*board_per_skill+i+1 < seq_len:
                decoder_inputB = boardSeq.clone()[i+1+j*board_per_skill,:,:].view(1, batch_size, 512)
            else:
                decoder_inputB = outputB
            #print(output.shape)
            generated_boards[count] = outputB.view(batch_size, 512)
            count=count + 1
#             if count == 10:
#                 break
#         if count == 10:
#             break
    
    #######################################################BOARD GENERATION FROM 2 ############################\
    #decoder_input = torch.ones(1, bs, 84) # <SOS_index>
    decoder_inputB1 = boardSeq.clone()[0, :, :].view(1, batch_size, 512) #game start positions as start token
#     if use_cuda:
#         decoder_inputB = decoder_inputB.to(device)
    #print('decoder_input', decoder_input.shape)
    
    
    #print('decoder_hidden', decoder_hidden.shape)

    # output, hn = decoder(decoder_input, decoder_hidden)

    # print('output', output.shape)#1,batch,84
    # print('hn', hn.shape)#1,4,768
    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_boards1 = torch.ones(board_per_skill * 4, batch_size, 512)
#     print(seq_len)
#     if use_cuda:
#         generated_boards = generated_boards.to(device)
    for j in range(skills2.size()[1]):
        decoder_hiddenB1 = skills2.clone()[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
#         if use_cuda:
#             decoder_hiddenB = decoder_hiddenB.to(device)
        for i in range(board_per_skill):
            outputB1, decoder_hiddenB1 = decoderBoard2(decoder_inputB1, decoder_hiddenB1)
            if use_teacher_forcing and j*board_per_skill+i+1 < seq_len:
                decoder_inputB1 = boardSeq.clone()[i+1+j*board_per_skill,:,:].view(1, batch_size, 512)
            else:
                decoder_inputB1 = outputB1
            #print(output.shape)
            generated_boards1[count] = outputB1.view(batch_size, 512)
            count=count + 1
#             if count == 10:
#                 break
#         if count == 10:
#             break
    
    
#     generated_boards_real = torch.ones(board_per_skill * skill_len, batch_size, 15*8*8)
#     if use_cuda:
#         generated_boards_real = generated_boards_real.cuda()
#     for i in range(board_per_skill * skill_len):
#         generated_boards_real[i] = w_hidden2board(generated_boards[i])
    
    criterion = SoftDTW(gamma=1.0, normalize=True)
    if torch.cuda.device_count() > 1:
        criterion = CriterionParallel(criterion)
    
    generated_boards = generated_boards.permute(1,0,2)
    generated_boards1 = generated_boards1.permute(1,0,2)
    
#     lnorm = nn.LayerNorm(generated_boards.size()[1:])
#     generated_boards1 = (generated_boards)
#     print(generated_boards1.shape)
#     print(generated_boards)
    # boardSeq = torch.transpose(boardSeq, 0, 1)#.view(batch_size, seq_len, 8*8*15)
    tar = boardSeq.permute(1,0,2).clone()[:, 1:, :]

    loss_dtw = criterion(generated_boards, tar)
    loss_dtw1 = criterion(generated_boards1, tar)
    loss_skills = criterion(generated_skills.permute(1,0,2), skills.permute(1,0,2))

#     print(loss_dtw.shape)
    
    
    
#     generated_boards = torch.transpose(generated_boards, 0, 1).view(batch_size, board_per_skill * skill_len, 1000)
#     tar = torch.transpose(board_seq_encoded, 0, 1).view(batch_size, seq_len, 1000)
#     boardSeq = torch.transpose(boardSeq, 0, 1).view(batch_size, 10, 8*8*15)

#     loss_dtw = criterion(tar, generated_boards)
    loss1 = torch.sum(loss_dtw)
    loss2 = torch.sum(loss_dtw1)
    loss3 = torch.sum(loss_skills)

    
#     for i in range(seq_len-1):
#         lossBoard = lossBoard + criteriaBoard(generated_boards_real[i].view(batch_size, 8*8*15), boardSeq[i+1].view(batch_size, 8*8*15))
        

    # print(loss)


    loss = loss1 + loss2 + loss3 #+ lossComment
    # print(loss)
# #     print(lossComment)
#     loss += lossComment
#     loss += ((skills[0].view(batch_size, d_model) - comment_skill[0].view(batch_size, d_model))**2).mean()
#     loss += ((skills[1].view(batch_size, d_model) - comment_skill[1].view(batch_size, d_model))**2).mean()
#     print(lossComment)
    # print(lossBoard)
#     loss = lossBoard + lossComment
#     print(lossComment)

    if torch.isnan(loss):
        pass
    else:
        loss.backward()
#         cnn1_opt.step()
        w_hidden2hidden_opt.step()
        boardenc2skill1_opt.step()
        decoderBoard1_opt.step()
        boardenc2skill2_opt.step()
        decoderBoard2_opt.step()
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


def trainIters(cnn1, w_hidden2hidden, boardenc2skill1, boardenc2skill2, comment2skill, decoderBoard1, decoderBoard2, w_hidden2board, decoderComment, epochs, train_loader, test_loader, learning_rate=0.00001):
    start = time.time()

    cnn1_opt = None#optim.Adam(filter(lambda x: x.requires_grad, cnn1.parameters()),
#                                   lr=learning_rate)
    w_hidden2hidden_opt = optim.Adam(filter(lambda x: x.requires_grad, w_hidden2hidden.parameters()),
                                  lr=learning_rate)
    boardenc2skill1_opt = optim.Adam(filter(lambda x: x.requires_grad, boardenc2skill1.parameters()),
                                  lr=learning_rate)
    boardenc2skill2_opt = optim.Adam(filter(lambda x: x.requires_grad, boardenc2skill2.parameters()),
                                  lr=learning_rate)
    decoderBoard1_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderBoard1.parameters()),
                                  lr=learning_rate)
    decoderBoard2_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderBoard2.parameters()),
                                  lr=learning_rate)
    w_hidden2board_opt = None#optim.Adam(filter(lambda x: x.requires_grad, w_hidden2board.parameters()),
#                                   lr=learning_rate)
    decoderComment_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderComment.parameters()),
                                  lr=learning_rate)
    comment2skill_opt = None#optim.Adam(filter(lambda x: x.requires_grad, comment2skill.parameters()),
#                                   lr=learning_rate)
    
    criteriaBoard = nn.BCEWithLogitsLoss()
    criteriaComment = nn.NLLLoss()
    if torch.cuda.device_count() > 1:
        criteriaBoard = CriterionParallel(criteriaBoard)
        criteriaComment = CriterionParallel(criteriaComment)

    print("Initialised optimisers")
    # before = list(cnn.parameters())[0].clone()
#     for epoch in range(epochs):
#         print_loss_total = 0.
#         for entry in train_loader:
#             loss = train(entry[0], entry[1], entry[2], cnn1, w_hidden2hidden, boardenc2skill1, boardenc2skill2, comment2skill, decoderBoard1, decoderBoard2, w_hidden2board, decoderComment,  
#                             cnn1_opt, w_hidden2hidden_opt, boardenc2skill1_opt, boardenc2skill2_opt, comment2skill_opt, decoderBoard1_opt, decoderBoard2_opt, w_hidden2board_opt, decoderComment_opt, 
#                                     criteriaBoard, criteriaComment)            
#             print_loss_total = print_loss_total + loss
# # # # # # #         after = list(cnn.parameters())[0].clone()
# # # # # # #         for i in range(len(before)):
# # # # # # #             print(torch.equal(before[i].data, after[i].data))

#         print('epochs: '+str(epoch))
#         print('total loss: '+str(print_loss_total))
# #         torch.save(cnn1, "./cnn1_siam_rev_lowLR_random2.pth")
#         torch.save(w_hidden2hidden, "./models/w_hidden2hidden_cook_maximalSkillDTW_200_vaegauss.pth")
#         torch.save(boardenc2skill1, "./models/boardenc2skill1_cook_maximalSkillDTW_200_vaegauss.pth")
#         torch.save(boardenc2skill2, "./models/boardenc2skill2_cook_maximalSkillDTW_200_vaegauss.pth")
#         torch.save(decoderBoard1, "./models/decoderBoard1_cook_maximalSkillDTW_200_vaegauss.pth")
#         torch.save(decoderBoard2, "./models/decoderBoard2_cook_maximalSkillDTW_200_vaegauss.pth")
# #         torch.save(w_hidden2board, "./w_hidden2board_siam_rev_lowLR_random2.pth")
#         torch.save(decoderComment, "./models/decoderComment_cook_maximalSkillDTW_200_vaegauss.pth")
#         # torch.save(comment2skill, "/trainman-mount/trainman-scratch-trainman2-job-1c0042f5-b217-4d14-b5b9-08eb9544d789/models/comment2skill_siam_cook.pth")
#         print("Models saved for epoch")
    tot1 = 0
    tot2 = 0
    cnt = 0
    t_num1 = t_den1 = t_num2 = t_den2 = 0
    for entry in train_loader:
        al1, al2, iou11, iou12, iou21, iou22, skillsLow, skillsHigh = evaluate(entry[0], entry[1], entry[2], entry[3], cnn1, w_hidden2hidden, boardenc2skill1, boardenc2skill2, comment2skill, decoderBoard1, decoderBoard2, w_hidden2board, decoderComment)
        tot1 += al1
        tot2 += al2
        # t_num1 += iou11
        # t_num2 += iou21
        # t_den1 += iou12
        # t_den2 += iou22
        t_num1 += iou11/iou12
        t_num2 += iou21/iou22
        cnt += 1
        print(cnt)
        # skillsLowN = skillsLow.detach().cpu().numpy()
        # skillsHighN = skillsHigh.detach().cpu().numpy()
        # np.save(f'./skills_vae/{cnt}_low.npy', skillsLowN)
        # np.save(f'./skills_vae/{cnt}_high.npy', skillsHighN)

        # break
    print("Alignment score 1:", tot1/cnt)
    print("Alignment score 2:", tot2/cnt)
    print("IoU score 1:", t_num1/cnt)
    print("IoU score 2:", t_num2/cnt)

        

def evaluate(boardSeq, comment, bert_emb, intvals, cnn1, w_hidden2hidden, boardenc2skill1, boardenc2skill2, comment2skill, decoderBoard1, decoderBoard2, w_hidden2board, decoderComment):
    #
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
    
    #x = torch.randn(bs, input_len, d_model).to(device)

    skills = boardenc2skill1(board_seq_encoded_enlarged)
    # print("1 skill")
    skills2 = boardenc2skill2(skills)
    # print("2 skill")
    skill_len = skills.size()[1]
    # print('skill1', skills.shape)
    # print('skill2', skills2.shape)

#     print(skill_len)
    board_per_skill = 25 #max(1, int(10/skill_len))
#     print(skills.shape)
    
#     skillsB = skills.clone().detach()
#     skillsC = skills.clone().detach()
#     if use_cuda:
#         skills = skills.to(device)
#         skillsB = skillsB.cuda()
#         skillsC = skillsC.cuda()
    #print(skills.shape)#2, bs, 768


    
    #################Comment Encoding ##############################
    #comment = torch.randn(batch_size, 12)

    max_length = comment.size()[1]

#     if use_cuda:
#         comment = comment.to(device)
    
    # comment_skill = comment2skill(bert_emb.view(batch_size, 22, d_model)) # expected n_skills * bs * 768
    
    # comm_skill_len = comment_skill.size()[1]


#     print('comment skill', comment_skill.shape)
    
    ###################### SKILLS GENERATION 8#######################

    decoder_input = torch.ones(1, batch_size, 768)# <SOS_index>
    skills2 = skills2.to(dtype = torch.float32)
    decoder_input = decoder_input.to(dtype = torch.float32)
#     print(type(skills2), type(skills2))
#     decoder_inputB1 = board_seq_encoded.clone()[0, :, :].view(1, batch_size, 128) #game start positions as start token
    # if use_cuda:
    #     decoder_input = decoder_input.cuda()
#     print('decoder_input', decoder_input.shape)
    
    
    #print('decoder_hidden', decoder_hidden.shape)

    # output, hn = decoder(decoder_input, decoder_hidden)

    # print('output', output.shape)#1,batch,84
    # print('hn', hn.shape)#1,4,768
    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_skills = torch.ones(16, batch_size, 768)
#     print(seq_len)
    if use_cuda:
        generated_skills = generated_skills.cuda()
    for j in range(skills2.size()[1]):
        decoder_hidden = skills2[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
        # if use_cuda:
        #     decoder_hidden = decoder_hidden.cuda()
        for i in range(4):
            output, decoder_hidden = decoderBoard1(decoder_input, decoder_hidden)
#             if use_teacher_forcing and j*board_per_skill+i+1 < 10:
#                 decoder_inputB = board_seq_encoded.clone()[i+1+j*board_per_skill,:,:].view(1, batch_size, 128)
#             else:
            decoder_input = output
            #print(output.shape)
            generated_skills[count] = output.view(batch_size, 768)
            count=count + 1
#             if count == 10:
#                 break
#         if count == 10:
#             break
    generated_skills = generated_skills.permute(1,0,2)
    #######################################################BOARD GENERATION FROM 8 ############################\
    #decoder_input = torch.ones(1, bs, 84) # <SOS_index>
    decoder_inputB = boardSeq.clone()[0, :, :].view(1, batch_size, 512) #game start positions as start token
#     if use_cuda:
#         decoder_inputB = decoder_inputB.to(device)
    #print('decoder_input', decoder_input.shape)
    
    
    #print('decoder_hidden', decoder_hidden.shape)

    # output, hn = decoder(decoder_input, decoder_hidden)

    # print('output', output.shape)#1,batch,84
    # print('hn', hn.shape)#1,4,768
    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_boards = torch.ones(board_per_skill * 16, batch_size, 512)
#     print(seq_len)
#     if use_cuda:
#         generated_boards = generated_boards.to(device)
    for j in range(generated_skills.size()[1]):
        decoder_hiddenB = generated_skills.clone()[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
#         if use_cuda:
#             decoder_hiddenB = decoder_hiddenB.to(device)
        for i in range(board_per_skill):
            outputB, decoder_hiddenB = decoderBoard2(decoder_inputB, decoder_hiddenB)
            if use_teacher_forcing and j*board_per_skill+i+1 < seq_len:
                decoder_inputB = boardSeq.clone()[i+1+j*board_per_skill,:,:].view(1, batch_size, 512)
            else:
                decoder_inputB = outputB
            #print(output.shape)
            generated_boards[count] = outputB.view(batch_size, 512)
            count=count + 1
#             if count == 10:
#                 break
#         if count == 10:
#             break
    
    #######################################################BOARD GENERATION FROM 2 ############################\
    #decoder_input = torch.ones(1, bs, 84) # <SOS_index>
    decoder_inputB1 = boardSeq.clone()[0, :, :].view(1, batch_size, 512) #game start positions as start token
#     if use_cuda:
#         decoder_inputB = decoder_inputB.to(device)
    #print('decoder_input', decoder_input.shape)
    
    
    #print('decoder_hidden', decoder_hidden.shape)

    # output, hn = decoder(decoder_input, decoder_hidden)

    # print('output', output.shape)#1,batch,84
    # print('hn', hn.shape)#1,4,768
    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_boards1 = torch.ones(board_per_skill * 4, batch_size, 512)
#     print(seq_len)
#     if use_cuda:
#         generated_boards = generated_boards.to(device)
    for j in range(skills2.size()[1]):
        decoder_hiddenB1 = skills2.clone()[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
#         if use_cuda:
#             decoder_hiddenB = decoder_hiddenB.to(device)
        for i in range(board_per_skill):
            outputB1, decoder_hiddenB1 = decoderBoard2(decoder_inputB1, decoder_hiddenB1)
            if use_teacher_forcing and j*board_per_skill+i+1 < seq_len:
                decoder_inputB1 = boardSeq.clone()[i+1+j*board_per_skill,:,:].view(1, batch_size, 512)
            else:
                decoder_inputB1 = outputB1
            #print(output.shape)
            generated_boards1[count] = outputB1.view(batch_size, 512)
            count=count + 1
#             if count == 10:
#                 break
#         if count == 10:
#             break
    
    
#     generated_boards_real = torch.ones(board_per_skill * skill_len, batch_size, 15*8*8)
#     if use_cuda:
#         generated_boards_real = generated_boards_real.cuda()
#     for i in range(board_per_skill * skill_len):
#         generated_boards_real[i] = w_hidden2board(generated_boards[i])
    
    
    
#     print(generated_boards.shape)
#     print(boardSeq.shape)
#     criterion = SoftDTW(gamma=1.0, normalize=True)
#     print(torch.isnan(generated_boards).any())
#     print(torch.isnan(boardSeq).any())
#     print(generated_boards[0][0])
#     print(boardSeq[0][0])
    
    generated_boards = generated_boards.permute(1,0,2)
    generated_boards1 = generated_boards1.permute(1,0,2)
    
#     lnorm = nn.LayerNorm(generated_boards.size()[1:])
#     generated_boards1 = (generated_boards)
#     print(generated_boards1.shape)
#     print(generated_boards)
    # boardSeq = torch.transpose(boardSeq, 0, 1)#.view(batch_size, seq_len, 8*8*15)
    tar = boardSeq.permute(1,0,2).clone()[:, 1:, :]

    generated_boards_cpu = generated_boards.detach().cpu().numpy()
    generated_boards1_cpu = generated_boards1.detach().cpu().numpy()
    tar_cpu = tar.detach().cpu().numpy()
    intvals = intvals.detach().cpu().numpy()
    intvals = intvals[0]
    prefix = np.zeros(len(intvals))
    # print(intvals)
    tot_time = sum(intvals)
    prefix[0] = intvals[0]
    for i in range(1, len(intvals)):
        prefix[i] = prefix[i-1] + intvals[i]

    multi = tot_time / 200.0
    # multi1 = tot_time / 200.0

    pred = []
    prev = 0
    path, dist = dtw_path(generated_boards_cpu[0], tar_cpu[0])
    for prs in (path):
        if (prs[0]+1) % board_per_skill == 0 and prs[0] != 0:
            pred.append((prs[1]-prev) * multi)
            prev = prs[1]
        
    # print(path)
    pred1 = []
    prev = 0
    path1, dist1 = dtw_path(generated_boards1_cpu[0], tar_cpu[0])
    for prs in (path1):
        if (prs[0]+1) % board_per_skill == 0 and prs[0] != 0:
            pred1.append((prs[1]-prev) * multi)
            prev = prs[1]
    for i in range(1, len(pred)):
        pred[i] += pred[i-1]
    for i in range(1, len(pred1)):
        pred1[i] += pred1[i-1]
    pred = np.array(pred)
    pred1 = np.array(pred1)

    
    prefix = np.array(prefix)
    pred = np.array(pred)
    pred1 = np.array(pred1)


    print(prefix)
    print(pred)
    print(pred1)

    paths1, dists1 = dtw_path(prefix, pred)
    paths2, dists2 = dtw_path(prefix, pred1)
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

    curr = 0
    tot_den1 = 0
    tot_num1 = 0
    prev_d = -1
    prev_n = -1
    track = 0
    for nums in paths2:
        fi = nums[0]
        se = nums[1]
        x2 = prefix[fi]
        y2 = pred1[se]
        if fi == 0:
            x1 = 0.0
        else:
            x1 = prefix[fi-1]
        if se == 0:
            y1 = 0.0
        else:
            y1 = pred1[se-1]
        tot_num1 += max(0.0, min(x2,y2) - max(x1,y1))
        if prev_d != x2 and prev_n != x1:
            tot_den1 += max(0.0, max(x2,y2) - min(x1,y1))
            prev_d = x2
            prev_n = x1
        # print(tot_den1)
        # print(tot_num1)
    # print("Alignment Score 1:",dists1)
    # print("Alignment Score 2:",dists2)
    # print("IoU Score 1:", tot_num/tot_den)
    # print("IoU Score 2:", tot_num1/tot_den1)
    return dists1, dists2, tot_num, tot_den, tot_num1, tot_den1, generated_skills, skills2
    # print(path1)
    print("----------------------------------")
    # loss_dtw = criterion(generated_boards, tar)
    # loss_dtw1 = criterion(generated_boards1, tar)
    # loss_skills = criterion(generated_skills.permute(1,0,2), skills.permute(1,0,2))
        



train_data = CookData(data_path="./feat_csv/feat_comment_interval_200_validation.npy")
dataload = DataLoader(train_data, batch_size=1, num_workers=4, pin_memory=False)
print('Data loaded')

# TRAIN
# # cnn1 = torchvision.models.resnet18()
# # cnn1 = torch.load('./AE_enc_newloss-Copy1.pth')
# # cnn1 = cnn1.eval()
# # cnn1.conv1 = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3,bias=False)
# # cnn1 = Net(15)
# # cnn1 = LeelaModel(100, 30)
# # cnn1.double()

# # w_hidden2hidden.double()
# w_hidden2hidden = WeightMultInverse(512, d_model)
# cnn1 = None
# w_hidden2board = None
# boardenc2skill1 = BoardEncodedtoSkills(input_len, output_len1, d_model, d_model)
# boardenc2skill2 = BoardEncodedtoSkills(output_len1, output_len2, d_model, d_model)

# comment2skill = None#CommentEncodedtoSkills(input_len, output_len, d_model, train_data.n_words).to(device)

# decoderBoard1 = DecoderRNNBoard(768, 768, 1)
# decoderBoard2 = DecoderRNNBoard(512, 768, 1)
# # decoderBoard1.double()
# # decoderBoard2.double()
# # w_hidden2hidden.double()
# # decoderBoard.double()
# # w_hidden2board = WeightMultInverse(128, 15*8*8)
# decoderComment = DecoderRNNComment(d_model, train_data.n_words, n_layers=1)


# EVAL

cnn1 = None#torch.load('./cnn1_siam_rev_lowLR_random1.pth')
w_hidden2hidden = torch.load('./models/w_hidden2hidden_cook_maximalSkillDTW_200_vaegauss.pth')
boardenc2skill1 = torch.load('./models/boardenc2skill1_cook_maximalSkillDTW_200_vaegauss.pth')
boardenc2skill2 = torch.load('./models/boardenc2skill2_cook_maximalSkillDTW_200_vaegauss.pth')

decoderBoard1 = torch.load('./models/decoderBoard1_cook_maximalSkillDTW_200_vaegauss.pth')
decoderBoard2 = torch.load('./models/decoderBoard2_cook_maximalSkillDTW_200_vaegauss.pth')
w_hidden2board = None#torch.load('./w_hidden2board_siam_rev_lowLR_random1.pth')
decoderComment = torch.load('./models/decoderComment_cook_maximalSkillDTW_200_vaegauss.pth')
comment2skill = None#torch.load('./comment2skill_siam_cook.pth')
# cnn1 = cnn1.eval()
w_hidden2hidden = w_hidden2hidden.eval()
boardenc2skill1 = boardenc2skill1.eval()
boardenc2skill2 = boardenc2skill2.eval()
# w_hidden2board = w_hidden2board.eval()
decoderComment = decoderComment.eval()
decoderBoard1 = decoderBoard1.eval()
decoderBoard2 = decoderBoard2.eval()
# comment2skill = comment2skill.eval()
print("Let's use", torch.cuda.device_count(), "GPUs!")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    w_hidden2hidden = nn.DataParallel(w_hidden2hidden)
    boardenc2skill1 = nn.DataParallel(boardenc2skill1)
    decoderBoard1 = nn.DataParallel(decoderBoard1)
    boardenc2skill2 = nn.DataParallel(boardenc2skill2)
    decoderBoard2 = nn.DataParallel(decoderBoard2)
    decoderComment = nn.DataParallel(decoderComment)
    # comment2skill = nn.DataParallel(comment2skill)
if use_cuda:
#     cnn1 = cnn1.cuda()
    w_hidden2hidden = w_hidden2hidden.cuda()
    boardenc2skill1 = boardenc2skill1.cuda()
    decoderBoard1 = decoderBoard1.cuda()
    boardenc2skill2 = boardenc2skill2.cuda()
    decoderBoard2 = decoderBoard2.cuda()
#     w_hidden2board = w_hidden2board.cuda()
    decoderComment = decoderComment.cuda()
    # comment2skill = comment2skill.cuda()
    
#     w_hidden2hidden = nn.DataParallel(w_hidden2hidden)
#     boardenc2skill = nn.DataParallel(boardenc2skill)
#     decoderBoard = nn.DataParallel(decoderBoard)
#     decoderComment = nn.DataParallel(decoderComment)
#     comment2skill = nn.DataParallel(comment2skill)

# print(len(train_data.word2idx))
print("Training starts")
trainIters(cnn1, w_hidden2hidden, boardenc2skill1, boardenc2skill2, comment2skill, decoderBoard1, decoderBoard2, w_hidden2board, decoderComment, 100, dataload, dataload)
