import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.2):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
        self.hidden = None
        
    
    def forward(self, features, captions):
        embedded_captions = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embedded_captions), 1)
        x, self.hidden = self.lstm(inputs, self.hidden)
        x = self.fc1(x)
        
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        self.hidden = None
        output = []
        
        for i in range(max_len):
            lstm_out, self.hidden = self.lstm(inputs, self.hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.fc1(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            output.append(max_indice.cpu().numpy()[0].item())
            if max_indice == 1: break
            inputs = self.embed(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) 
            
        return output

    

# class DecoderAttentionRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.2):
#         super(DecoderAttentionRNN, self).__init__()
        
#         self.hidden_size = hidden_size
        
#         self.attention = 
        
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, vocab_size)
        
#         self.hidden = self.initHidden()
        
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, 
#                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    
#     def forward(self, features, captions):
#         embedded_captions = self.embed(captions[:,:-1])
        
#         attn_weights = F.softmax(self.attn(torch.cat((embedded_captions[0], self.hidden[0]), 1)), dim=1)
        
#         attn_applied = tourch.bmm(attn_weights.unsqueeze(0),
#                                  features.unsqueeze(0))
        
#         inputs = torch.cat((embedded_captions[0], attn_applied[0]), 1)
#         inputs = F.relu(self.attn_combine(inputs).unsqueeze(0))
        
# #         inputs = torch.cat((features.unsqueeze(1), embedded_captions), 1)
#         x, self.hidden = self.lstm(inputs, self.hidden)
#         x = self.fc1(x)
        
#         return x

#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         hidden = None
#         output = []
        
#         for i in range(max_len):
#             lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
#             outputs = self.fc1(lstm_out)  # outputs shape : (1, 1, vocab_size)
#             outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
#             _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
#             output.append(max_indice.cpu().numpy()[0].item())
#             if max_indice == 1: break
#             inputs = self.embed(max_indice) # inputs shape : (1, embed_size)
#             inputs = inputs.unsqueeze(1) 
            
#         return output