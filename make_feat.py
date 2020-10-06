import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import TweetTokenizer
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import spacy
import chess
import numpy as np
import os
import random
import math
import time
from transformers import *


MAX_SEQ_LEN = 100



class CookData(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle = True).item()
        self.data_feat = list(self.data)
        self.data_comment = list(self.data.values())
        self.word2idx = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.idx2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3
        self.vocab = self.build_vocab(self.data)
        self.oov = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.loaded_data = self.load_data(self.data_feat)#np.load("all_video_features_200_validation.npy", allow_pickle=True)#
        # self.fps_data = self.load_fps(self.data_feat)
        self.time_truth = self.load_times(self.data_comment)
    
    def load_times(self, inp):
        time_list = []
        for indexes in range(len(inp)):
            resnet_feat_path = inp[indexes]
            # print(len(resnet_feat_path))
            times = resnet_feat_path[2]
            if len(times) < 20:
                for x in range(20-len(times)):
                    times.append(0)
            else:
                times = times[:20]
            time_list.append(times)
        time_list = torch.FloatTensor(time_list)
        print("Times loaded!")
        # print(fps_list[:10])
        return time_list

    def load_fps(self, inp):
        fps_list = []
        for indexes in range(len(inp)):
            resnet_feat_path = inp[indexes]
            interval = resnet_feat_path[1]
            fps = resnet_feat_path[2]
            fps_list.append(fps)
        fps_list = torch.FloatTensor(fps_list)
        print("FPS loaded!")
        # print(fps_list[:10])
        return fps_list

    def load_data(self, inp):
        combine_tensor = np.zeros((len(inp), 200, 512))
        downsample_frames = np.zeros((200, 512))
        fps_list = []
        print(len(inp))
        for indexes in range(len(inp)):
            # print(indexes)
            resnet_feat_path = inp[indexes]
            feat_tensor = np.zeros((500, 512)) # 500 frames * 512
            path_prefix = "./feat_csv/feat_csv/"
            
            feat_path = path_prefix + resnet_feat_path[0]
            intervals = self.data[resnet_feat_path][1]
            fps = resnet_feat_path[1]
            fps_list.append(fps)
            # fps = torch.FloatTensor([fps])
            
            i = 0
    #         print(feat_path)
            for folder in os.listdir(feat_path):
                new_path = feat_path + "/" + "0001" + "/" + "resnet_34_feat_mscoco.csv"
                # print(new_path)
                # print(start)
            
                df = pd.read_csv(new_path)
                error_column = list(df.columns)
                new_column = []
                for num in error_column:
                    try:
                        x = np.float32(num)
                    except ValueError as e:
                        x = np.float32(num[:5])
                    new_column.append(x)
                    
                col = np.array(new_column)   
                rest = np.float32(np.array(df.values))
                curr = np.vstack((col, rest)) # (500,512)
                feat_tensor[i:i+500, :] = curr
                i += 500
                break
            
            flag = 0
            for interval in intervals:
                start = int(interval[0] - 10.0) # Extra 10 frames on both sides of interval given
                end = int(interval[1] + 10.0)
                start = max(0.0, start)
                end = min(500.0, end)
                start = int(start)
                end = int(end)
                if not flag:
                    relevant_frames = feat_tensor[start:end]
                    flag = 1
                else:
                    relevant_frames = np.vstack((relevant_frames, feat_tensor[start:end]))
            
            sz = int(relevant_frames.shape[0])
            ctr = 0
            if sz >= 200:
                for tmp in range(0,sz,max(1, (sz//200))):
                    downsample_frames[ctr, :] = relevant_frames[tmp, :]
                    ctr += 1
                    if ctr >= 200:
                        break
                relevant_frames = downsample_frames
                # relevant_frames = relevant_frames[:200, :]
            else:
                relevant_frames = np.vstack((relevant_frames, np.zeros((200-sz, 512))))
            relevant_tensor = torch.FloatTensor(relevant_frames)

            combine_tensor[indexes] = relevant_tensor
        
        # save_arr = np.array(combine_tensor)
        np.save("all_video_features_200_validation.npy", combine_tensor)
        # combine_tensor = torch.FloatTensor(combine_tensor)
        return combine_tensor#, fps_list
    

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.loaded_data)

    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
#         print(index)
        comment = self.data_comment[index][0]
        comment = comment.split()

        if len(comment) > MAX_SEQ_LEN:
            comment = comment[:MAX_SEQ_LEN]
        else:
            lcomm = len(comment)
            for i in range(MAX_SEQ_LEN - lcomm):
                comment.append("PAD")
        input_ids = torch.tensor([self.tokenizer.encode(comment, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0] # BERT-base embeddings
        ##########################
        comm_vector = []
        comm_vector.append(0) #<SOS>
        for tok in comment:
            if tok in self.word2idx:
                comm_vector.append(self.word2idx[tok])
            else:
                comm_vector.append(2)
                self.oov += 1
        comm_vector.append(1) #<EOS>

        comm_tensor = torch.LongTensor(comm_vector)

        res_tensor = self.loaded_data[index]
        res_tensor = torch.FloatTensor(res_tensor)
        # fps = torch.FloatTensor([self.fps_data[index]])
        # return board_tensor, comm_tensor, move_tensor
        return res_tensor, comm_tensor, last_hidden_states, self.time_truth[index]
        



    def build_vocab(self, corpus):
        word_count = {}

        for sentences in corpus:
            sent = corpus[sentences][0]
            sent = sent.split()
            for token in sent:
                token = token.lower()
                if token not in word_count:
                    self.word2idx[token] = self.n_words
                    self.idx2word[self.n_words] = token
                    word_count[token] = 1
                    self.n_words += 1
                else:
                    word_count[token] += 1
        
#         print(self.n_words)
        return word_count


train_data = CookData(data_path="./feat_csv/feat_comment_interval_200_validation.npy")
# dataload = DataLoader(train_data, batch_size=1, num_workers=1)

# print("Vocab:", train_data.n_words)
# for ind, i in enumerate(train_data):
#     print(i[0].shape)
#     print(i[1].shape)
#     print(i[2].shape)
#     # print(i[3][0])
#     break
