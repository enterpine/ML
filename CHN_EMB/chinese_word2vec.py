#import word2vec
#word2vec.word2vec('lvke20200305_02_splite_done.txt', 'lvke20200305_02_emb.bin', size=300,verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

import word2vec
model = word2vec.load('lvke20200305_02_emb.bin')
print (model.vectors.shape)



context_len = torch.sum(text_raw_indices != 0, dim=-1)

self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

context = self.embed(text_raw_indices)