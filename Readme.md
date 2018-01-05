# Attention Sum Model
This is an implementation of the Attention Sum Model which was introduced by Kaldec et al. in the paper titeled [Text Understanding with the Attention Sum Reader Network](https://arxiv.org/pdf/1603.01547.pdf). The paper aims at solving cloze-style question answering problem. 

## Dataset
We train the model on the Children's Book Test (CBT) dataset. Each datapoint consists of twenty-one consecutive lines extracted from a book. The first twenty lines form the context. From the twenty first line, one named-entity is removed. The purpose of the model is to predict this missing named-entity out a list of possible candidates. 

