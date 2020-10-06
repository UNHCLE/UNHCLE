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
output_len = 5
skill_max_len = 5
t_ratio = 0.5
ngpu = 3
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

import sys

torch.autograd.set_detect_anomaly(True)


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

        out_val = self.conv_val(out).view(-1, 32*8*8)
        out_val = F.relu(self.affine_val_1(out_val), inplace=True)

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

        self.predictor = nn.Linear(self.d_model, self.vocab_size)
        self.soft = nn.Softmax(dim=0)
        self.probHead = nn.Linear(self.d_model, 1)

    def forward(self, x):
        #x = torch.randn(bs, input_len, d_model).to(device)
        bs = x.size()[0]
        pos_x = x.transpose(0, 1)
#         x_enc = self.decoder_emb(x)

        encoder_output = self.encoder(pos_x).to(device)  # (input_len, bs, d_model)
        
        # initialized the input of the decoder with sos_idx (start of sentence token idx)
        output = torch.ones(self.output_len, bs).long().to(device)*sos_idx
        predictions = torch.zeros(output_len, bs, self.vocab_size).to(device)

        for t in range(self.output_len - 1):
            tgt_emb = self.decoder_emb(output[:t+1].transpose(0,1).clone()).transpose(0, 1)
            
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(device)
#             print(tgt_emb.shape)
            decoder_output = self.decoder(tgt=tgt_emb, memory=encoder_output)
            
            prediction = self.predictor(decoder_output[-1])
            # predictions - [t, bs, vocab]
            predictions[t] = prediction
            
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
        pos_x = self.pos_encoder(x.permute(1,0,2)) 
#         print(pos_x.shape)
        encoder_output = self.encoder(pos_x).to(device)  # (input_len, bs, d_model)

        # initialized the input of the decoder with sos_idx (start of sentence token idx)
        output = torch.ones(self.output_len, bs).long().to(device)*sos_idx
        predictions = torch.zeros(self.output_len, bs, self.vocab_size).to(device)

        for t in range(self.output_len - 1):
            tgt_emb = self.decoder_emb(output[:t+1].transpose(0,1).clone()).transpose(0, 1)
#             print(tgt_emb.shape)
            
            tgt_emb = self.pos_encoder(tgt_emb)
            decoder_output = self.decoder(tgt=tgt_emb, memory=encoder_output)
            
            prediction = self.predictor(decoder_output[-1])
            # predictions - [t, bs, vocab]
            predictions[t] = prediction

            one_hot_idx = prediction.argmax(1)

            # output  = [output len, batch size]
            output[t+1] = one_hot_idx            
            
        return decoder_output.transpose(0,1)
    


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

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))

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
        
        hidden = torch.transpose(hidden, 0, 1)
        bs = input.shape[0]

        output = input.view(1, bs, self.hidden_size)
        for i in range(self.n_layers):
            output = F.relu(output)

            output, hidden = self.gru(output, hidden)
            
        output = self.softmax(self.out(output[0]))
        return output, hidden.transpose(0, 1)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class Deconv(nn.Module):
    def __init__(self, in_features, out_features):
        super(Deconv, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 3840)

        self.deconv1 = nn.ConvTranspose2d(60, 40, 1)
        self.deconv2 = nn.ConvTranspose2d(40, 30, 1)
        self.deconv3 = nn.ConvTranspose2d(30, out_features, 1)
        #self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        #print(x.size())
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.size())
        x = x.view(batch_size, 60, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        return x


def train(boardSeq, comment, bert_emb, cnn1, w_hidden2hidden, boardenc2skill, comment2skill, decoderBoard, w_hidden2board, decoderComment,  
cnn1_opt, w_hidden2hidden_opt, boardenc2skill_opt, comment2skill_opt, decoderBoard_opt, w_hidden2board_opt, decoderComment_opt, 
criteriaBoard, criteriaComment,
):



#     cnn1_opt.zero_grad()
    w_hidden2hidden_opt.zero_grad()
    boardenc2skill_opt.zero_grad()
    decoderBoard_opt.zero_grad() 
#     w_hidden2board_opt.zero_grad()
    decoderComment_opt.zero_grad()
    comment2skill_opt.zero_grad()

    
    lossBoard = 0
    lossComment = 0
    loss = 0
    #move = (batch, seq_len, 5)
    ################## Board Encoding ############################################################################

    boardSeq = torch.transpose(boardSeq, 0, 1) #batch, 500, 512-> 500, batch, 512
    seq_len = boardSeq.size()[0]
    batch_size = boardSeq.size()[1]

    board_seq_encoded_enlarged = torch.zeros(seq_len, batch_size, d_model)

    for i in range(seq_len):
        board_seq_encoded_enlarged[i] = w_hidden2hidden(boardSeq[i])


    board_seq_encoded_enlarged = torch.transpose(board_seq_encoded_enlarged, 0, 1) #(input_len, bs, 768)-> (bs, input_len, 768)


    skills = boardenc2skill(board_seq_encoded_enlarged)
    skill_len = skills.size()[1]

    board_per_skill = 100 #max(1, int(10/skill_len))


    
    #################Comment Encoding ##############################

    max_length = comment.size()[1]
    
    
    #######################################################BOARD GENERATION############################\
    decoder_inputB = boardSeq.clone()[0, :, :].view(1, batch_size, 512) #game start positions as start token

    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_boards = torch.ones(board_per_skill * skill_len, batch_size, 512)

    for j in range(skills.size()[1]):
        decoder_hiddenB = skills.clone()[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)

        for i in range(board_per_skill):
            outputB, decoder_hiddenB = decoderBoard(decoder_inputB, decoder_hiddenB)

            decoder_inputB = outputB
            #print(output.shape)
            generated_boards[count] = outputB.view(batch_size, 512)
            count=count + 1
    

    criterion = SoftDTW(gamma=1.0, normalize=True)
    if torch.cuda.device_count() > 1:
        criterion = CriterionParallel(criterion)
    
    generated_boards = torch.transpose(generated_boards, 0, 1)

    boardSeq = torch.transpose(boardSeq, 0, 1)#.view(batch_size, seq_len, 8*8*15)
    tar = boardSeq.clone()[:, 1:, :]
    loss_dtw = criterion(generated_boards, tar)
#     print(loss_dtw.shape)
    

    loss = torch.sum(loss_dtw)
    
    if torch.isnan(loss):
        pass
    else:
        loss.backward()
#         cnn1_opt.step()
        w_hidden2hidden_opt.step()
        boardenc2skill_opt.step()
        decoderBoard_opt.step()
#         w_hidden2board_opt.step()
        decoderComment_opt.step()
        comment2skill_opt.step()

    return float(loss.item())

class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if not isinstance(criterion, torch.nn.Module):
            criterion = ModularizedFunction(criterion)
        self.criterion = torch.nn.DataParallel(criterion)
    def forward(self, *args, **kwargs):
        return self.criterion(*args, **kwargs).mean()


def trainIters(cnn1, w_hidden2hidden, boardenc2skill, comment2skill, decoderBoard, w_hidden2board, decoderComment, epochs, train_loader, test_loader, learning_rate=0.00001):
    start = time.time()
    cnn1_opt = None#optim.Adam(filter(lambda x: x.requires_grad, cnn1.parameters()),
                                  # lr=learning_rate)

    w_hidden2hidden_opt = optim.Adam(filter(lambda x: x.requires_grad, w_hidden2hidden.parameters()),
                                  lr=learning_rate)
    boardenc2skill_opt = optim.Adam(filter(lambda x: x.requires_grad, boardenc2skill.parameters()),
                                  lr=learning_rate)
    decoderBoard_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderBoard.parameters()),
                                  lr=learning_rate)
    w_hidden2board_opt = None#optim.Adam(filter(lambda x: x.requires_grad, w_hidden2board.parameters()),
#                                   lr=learning_rate)
    decoderComment_opt = optim.Adam(filter(lambda x: x.requires_grad, decoderComment.parameters()),
                                  lr=learning_rate)
    comment2skill_opt = optim.Adam(filter(lambda x: x.requires_grad, comment2skill.parameters()),
                                  lr=learning_rate)
    
    criteriaBoard = nn.BCEWithLogitsLoss()
    criteriaComment = nn.NLLLoss()
    if torch.cuda.device_count() > 1:
        criteriaBoard = CriterionParallel(criteriaBoard)
        criteriaComment = CriterionParallel(criteriaComment)

    # before = list(cnn.parameters())[0].clone()
    for epoch in range(epochs):
        print_loss_total = 0.
        for entry in train_loader:
            loss = train(entry[0], entry[1], entry[2], cnn1, w_hidden2hidden, boardenc2skill, comment2skill, decoderBoard, w_hidden2board, decoderComment,  
                            cnn1_opt, w_hidden2hidden_opt, boardenc2skill_opt, comment2skill_opt, decoderBoard_opt, w_hidden2board_opt, decoderComment_opt, 
                                    criteriaBoard, criteriaComment)            
            print_loss_total = print_loss_total + loss
# # # # # #         after = list(cnn.parameters())[0].clone()
# # # # # #         for i in range(len(before)):
# # # # # #             print(torch.equal(before[i].data, after[i].data))

        print('epochs: '+str(epoch))
        print('total loss: '+str(print_loss_total))
#         torch.save(cnn1, "./cnn1_siam_rev_lowLR_random2.pth")
        torch.save(w_hidden2hidden, "./models/w_hidden2hidden_siam_cook.pth")
        torch.save(boardenc2skill, "./models/boardenc2skill_siam_cook.pth")
        torch.save(decoderBoard, "./models/decoderBoard_siam_cook.pth")
#         torch.save(w_hidden2board, "./w_hidden2board_siam_rev_lowLR_random2.pth")
        torch.save(decoderComment, "./models/decoderComment_siam_cook.pth")
        torch.save(comment2skill, "./models/comment2skill_siam_cook.pth")
        print("Models saved for epoch")
    tot1 = 0
    tot2 = 0
    cnt = 0
    t_num1 = t_den1 = t_num2 = t_den2 = 0
    for entry in train_loader:
        al1, al2, iou11, iou12, iou21, iou22 = evaluate(entry[0], entry[1], entry[2], entry[3], cnn1, w_hidden2hidden, boardenc2skill, comment2skill, decoderBoard, w_hidden2board, decoderComment)
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
        # break
    print("Alignment score 1:", tot1/cnt)
    print("Alignment score 2:", tot2/cnt)
    print("IoU score 1:", t_num1/cnt)
    print("IoU score 2:", t_num2/cnt)
        

def evaluate(boardSeq, comment, bert_emb, intvals, cnn1, w_hidden2hidden, boardenc2skill, comment2skill, decoderBoard, w_hidden2board, decoderComment):
    #
    boardSeq = torch.transpose(boardSeq, 0, 1) #batch, 500, 512-> 500, batch, 512
#     boardSeq = boardSeq.to(dtype = torch.float64)
    seq_len = boardSeq.size()[0]
    batch_size = boardSeq.size()[1]

    board_seq_encoded_enlarged = torch.zeros(seq_len, batch_size, d_model)

    for i in range(seq_len):
        board_seq_encoded_enlarged[i] = w_hidden2hidden(boardSeq[i])


    board_seq_encoded_enlarged = torch.transpose(board_seq_encoded_enlarged, 0, 1) #(input_len, bs, 768)-> (bs, input_len, 768)


    skills = boardenc2skill(board_seq_encoded_enlarged)
    skill_len = skills.size()[1]
#     print('skill', skills.shape)

#     print(skill_len)
    board_per_skill = 100 #max(1, int(10/skill_len))
#     print(skills.shape)


    
    #################Comment Encoding ##############################
    #comment = torch.randn(batch_size, 12)

    max_length = comment.size()[1]
    
    
    #######################################################BOARD GENERATION############################\
    #decoder_input = torch.ones(1, bs, 84) # <SOS_index>
    decoder_inputB = boardSeq.clone()[0, :, :].view(1, batch_size, 512) #game start positions as start token

    count = 0
    use_teacher_forcing = True if random.random() < t_ratio else False
    generated_boards = torch.ones(board_per_skill * skill_len, batch_size, 512)

    for j in range(skills.size()[1]):
        decoder_hiddenB = skills.clone()[:,j,:].view(1, batch_size, d_model).to(dtype = torch.float32).repeat(1, 1, 1)
#         if use_cuda:
#             decoder_hiddenB = decoder_hiddenB.to(device)
        for i in range(board_per_skill):
            outputB, decoder_hiddenB = decoderBoard(decoder_inputB, decoder_hiddenB)

            decoder_inputB = outputB
            #print(output.shape)
            generated_boards[count] = outputB.view(batch_size, 512)
            count=count + 1

    
    generated_boards = torch.transpose(generated_boards, 0, 1)
    

    boardSeq = torch.transpose(boardSeq, 0, 1)#.view(batch_size, seq_len, 8*8*15)
    tar = boardSeq.clone()[:, 1:, :]
    # loss_dtw = criterion(generated_boards, tar)

    generated_boards_cpu = generated_boards.detach().cpu().numpy()
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
    pred = []
    prev = 0
    path, dist = dtw_path(generated_boards_cpu[0], tar_cpu[0])
    for prs in (path):
        if (prs[0]+1) % 25 == 0 and prs[0] != 0:
            pred.append((prs[1]-prev) * multi)
            prev = prs[1]
        
    pred = np.array(pred)

    
    prefix = np.unique(prefix)
    pred = np.unique(pred)

    paths1, dists1 = dtw_path(prefix, pred)

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

    return dists1, dists1, tot_num, tot_den, tot_num, tot_den

        
    





train_data = CookData(data_path="./feat_csv/feat_comment_interval_200_validation.npy")
dataload = DataLoader(train_data, batch_size=256, num_workers=64)
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
w_hidden2hidden = WeightMultInverse(512, d_model)
cnn1 = None
w_hidden2board = None
boardenc2skill = BoardEncodedtoSkills(input_len, output_len, d_model, d_model).to(device)
comment2skill = CommentEncodedtoSkills(input_len, output_len, d_model, train_data.n_words).to(device)
decoderBoard = DecoderRNNBoard(512, d_model, 1)
# decoderBoard.double()
# w_hidden2board = WeightMultInverse(128, 15*8*8)
decoderComment = DecoderRNNComment(d_model, train_data.n_words, n_layers=1)


# EVAL
# cnn1 = None
# w_hidden2board = None
# # # # cnn1 = None#torch.load('./cnn1_siam_rev_lowLR_random1.pth')
# w_hidden2hidden = torch.load('./models/w_hidden2hidden_siam_cook.pth')
# boardenc2skill = torch.load('./models/boardenc2skill_siam_cook.pth')
# decoderBoard = torch.load('./models/decoderBoard_siam_cook.pth')
# # #w_hidden2board = torch.load('./w_hidden2board_siam_rev_lowLR_random1.pth')
# decoderComment = torch.load('./models/decoderComment_siam_cook.pth')
# comment2skill = torch.load('./models/comment2skill_siam_cook.pth')
# # cnn1 = cnn1.eval()
# w_hidden2hidden = w_hidden2hidden.eval()
# boardenc2skill = boardenc2skill.eval()
# # w_hidden2board = w_hidden2board.eval()
# decoderComment = decoderComment.eval()
# decoderBoard = decoderBoard.eval()
# comment2skill = comment2skill.eval()

print("Let's use", torch.cuda.device_count(), "GPUs!")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    w_hidden2hidden = nn.DataParallel(w_hidden2hidden)
    boardenc2skill = nn.DataParallel(boardenc2skill)
    decoderBoard = nn.DataParallel(decoderBoard)
    decoderComment = nn.DataParallel(decoderComment)
    comment2skill = nn.DataParallel(comment2skill)
if use_cuda:
#     cnn1 = cnn1.cuda()
    w_hidden2hidden = w_hidden2hidden.cuda()
    boardenc2skill = boardenc2skill.cuda()
    decoderBoard = decoderBoard.cuda()
#     w_hidden2board = w_hidden2board.cuda()
    decoderComment = decoderComment.cuda()
    comment2skill = comment2skill.cuda()


# print(len(train_data.word2idx))
print("Training starts")
trainIters(cnn1, w_hidden2hidden, boardenc2skill, comment2skill, decoderBoard, w_hidden2board, decoderComment, 1000, dataload, dataload)

