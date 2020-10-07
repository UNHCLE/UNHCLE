import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import TweetTokenizer
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from chess_env import ChessEnv
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import spacy
import chess
import numpy as np

import random
import math
import time

MAX_SEQ_LEN = 10

class ChessData(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.mapping = {"p":0, "b":1, "r":2, "n":3, "k":4, "q":5, "P":6, "B":7, "R":8, "N":9, "K":10, "Q":11}
        self.word2idx1 = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.idx2word1 = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.word2idx2 = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.idx2word2 = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words1 = 3
        self.n_words2 = 3
        self.vocab1 = self.build_vocab1(self.data)  # moves data
        self.vocab2 = self.build_vocab2(self.data)  # comments data
        self.oov = 0
    
    # def get_vocab1(self):
    #     return self.vocab1

    def get_vocab(self):
        return self.vocab2

    def __len__(self):
        return len(self.data)
    
    def create_board(self, board, color, isCheck, isCastle):
        brd = str(board)
        brd = brd.split("\n")
        brd_tensor = np.zeros((15,8,8))

        for i in range(8):
            for j in range(8):
                row = brd[i].split()
                piece = row[j]
                if piece != '.':
#                     if color:
                    brd_tensor[self.mapping[piece]][i][j] = 1
#                     else:
#                         map_opponent = {"p":6, "b":7, "r":8, "n":9, "k":10, "q":11, "P":0, "B":1, "R":2, "N":3, "K":4, "Q":5}
#                         brd_tensor[map_opponent[piece]][i][j] = 1
                if color:
                    brd_tensor[12][i][j] = 1
#                 else:
#                     brd_tensor[12][i][j] = -1
                if isCastle:
                    brd_tensor[13][i][j] = 1
#                 else:
#                     brd_tensor[13][i][j] = -1
                if isCheck:
                    brd_tensor[14][i][j] = 1
#                 else:
#                     brd_tensor[14][i][j] = -1

        brd_tensor = torch.FloatTensor(brd_tensor)
        return brd_tensor


    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        move = self.data['moves'][index].split()
        comment = self.data['opening_name'][index].lower().split()
        sz = self.data['opening_ply'][index]

        if sz > MAX_SEQ_LEN:
            move = move[:MAX_SEQ_LEN]
        else:
            lmove = len(move)
            move = move[:sz]
            for i in range(MAX_SEQ_LEN - sz):
                move.append("PAD")

        if len(comment) > MAX_SEQ_LEN:
            comment = comment[:MAX_SEQ_LEN]
        else:
            lcomm = len(comment)
            for i in range(MAX_SEQ_LEN - lcomm):
                comment.append("PAD")
        comm_vector = []
        comm_vector.append(0) #<SOS>
        for tok in comment:
            if tok in self.word2idx2:
                comm_vector.append(self.word2idx2[tok])
            else:
                comm_vector.append(2)
                self.oov += 1
        comm_vector.append(1) #<EOS>

        comm_tensor = torch.LongTensor(comm_vector)

        board_tensor = np.zeros((10,15,8,8)) #10 boards at a time
        board = chess.Board()
        color = 1
        isCastle = 0
        isCheck = 0

        board_tensor[0] = self.create_board(board, color, isCheck, isCastle)
        for x in range(10 - 1):
            color = 0 if board.turn else 1
            if move[x] == "O-O" or move[x] == "O-O-O":
                isCastle = 1
            else:
                isCastle = 0
            com_sz = len(move[x])
            if move[x][com_sz-1] == '+' or move[x][com_sz-1] == '#':
                isCheck = 1
            else:
                isCheck = 0
            
            if move[x] != "PAD":
                board.push_san(move[x])
                board_tensor[x+1] = self.create_board(board, color, isCheck, isCastle)
            else:
                board_tensor[x+1] = torch.FloatTensor(np.zeros((15,8,8)))
        
        move_tensor = np.zeros((10,5))
        for index in range(10):
            entry = move[index]
            move_indexes = []
            for i in range(5):
                if i < len(entry):
                    alphabet = entry[i]
                    if alphabet in self.word2idx1:
                        move_indexes.append(self.word2idx1[alphabet])
                    else:
                        move_indexes.append(2)
                        self.oov += 1
                else:
                    move_indexes.append(2)
                    self.oov += 1
            move_tensor[index] = np.array(move_indexes)
        
        move_tensor = torch.LongTensor(move_tensor)
            

        return board_tensor, move_tensor, comm_tensor
        # return board_tensor, move_tensor
        



    def build_vocab1(self, corpus):
        word_count = {}
        tk = TweetTokenizer()
        moves = corpus['moves'].tolist()
        # print(moves[0])

        for move in moves:
            move = move.split()
            for single_move in move:
                for token in single_move:
                    if token not in word_count:
                        self.word2idx1[token] = self.n_words1
                        self.idx2word1[self.n_words1] = token
                        word_count[token] = 1
                        self.n_words1 += 1
                    else:
                        word_count[token] += 1
        # print(self.n_words1)
        return word_count

    def build_vocab2(self, corpus):
        word_count = {}
        tk = TweetTokenizer()
        corpus = corpus['opening_name'].tolist()

        for sentence in corpus:
            # tokens = tk.tokenize(sentence)
            tokens = sentence.split()
            for token in tokens:
                token = token.lower()
                if token not in word_count:
                    self.word2idx2[token.lower()] = self.n_words2
                    self.idx2word2[self.n_words2] = token.lower()
                    word_count[token.lower()] = 1
                    self.n_words2 += 1
                else:
                    word_count[token.lower()] += 1
        # print(self.n_words2)
        return word_count



# train_data = ChessData(data_path="./games.csv")

# dataload = DataLoader(train_data, batch_size=4, num_workers=4)

# for i in train_data:
#     print(i[0].shape)
#     print(i[1].shape)
#     print(i[2].shape)
#     break
# print(train_data.n_words)
# for i in dataload:
#     print(i[0].shape)
#     print(i[1].shape)
#     print(i[2].shape)
#     break