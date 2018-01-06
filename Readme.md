# Attention Sum Reader Model
This is an implementation of the Attention Sum Reader Model which was introduced by Kadlec et al. in the paper titled [Text Understanding with the Attention Sum Reader Network](https://arxiv.org/pdf/1603.01547.pdf). The paper aims at addressing the machine reading problem and specifically targets to solve cloze-style question answering challenge. 

## Dataset
We train the model on the Children's Book Test (CBT) dataset. Each datapoint consists of twenty-one consecutive lines extracted from a book. The first twenty lines form the context. From the twenty first line, one named-entity is removed. The purpose of the model is to predict this missing named-entity out of a list of possible candidates. 

An example datapoint (Context in the first twenty lines. Sentence with missing named-entity, the answer and the candidates on the twenty-first line):
```
1 Then he remembered the awful curse of the oldest fairy , and was sorry for the rudeness of the queen .
2 And when the prince , after having his ears boxed , said that `` force was no argument , '' the king went away in a rage .
3 -LCB- Prigio reading a book : p11.jpg -RCB- Indeed , I can not tell you how the prince was hated by all !
4 He would go down into the kitchen , and show the cook how to make soup .
5 He would visit the poor people 's cottage , and teach them how to make the beds , and how to make plum-pudding out of turnip-tops , and venison cutlets out of rusty bacon .
6 He showed the fencing-master how to fence , and the professional cricketer how to bowl , and instructed the rat-catcher in breeding terriers .
7 He set sums to the Chancellor of the Exchequer , and assured the Astronomer Royal that the sun does not go round the earth -- which , for my part , I believe it does .
8 The young ladies of the Court disliked dancing with him , in spite of his good looks , because he was always asking , `` Have you read this ? ''
9 and `` Have you read that ? ''
10 -- and when they said they had n't , he sneered ; and when they said they had , he found them out .
11 He found out all his tutors and masters in the same horrid way ; correcting the accent of his French teacher , and trying to get his German tutor not to eat peas with his knife .
12 He also endeavoured to teach the queen-dowager , his grandmother , an art with which she had long been perfectly familiar !
13 In fact , he knew everything better than anybody else ; and the worst of it was that he did : and he was never in the wrong , and he always said , `` Did n't I tell you so ? ''
14 And , what was more , he had !
15 As time went on , Prince Prigio had two younger brothers , whom everybody liked .
16 They were not a bit clever , but jolly .
17 Prince Alphonso , the third son , was round , fat , good-humoured , and as brave as a lion .
18 Prince Enrico , the second , was tall , thin , and a little sad , but never too clever .
19 Both were in love with two of their own cousins -LRB- with the approval of their dear parents -RRB- ; and all the world said , `` What nice , unaffected princes they are ! ''
20 But Prigio nearly got the country into several wars by being too clever for the foreign ambassadors .
21 Now , as Pantouflia was a rich , lazy country , which hated fighting , this was very unpleasant , and did not make people love Prince XXXXX any better .     Prigio          Court|Enrico|Exchequer|German|Prigio|Royal|cricketer|p11.jpg|second|turnip-tops
```

Note that the data is alread tokenized and very little pre-processing is necessary. There were 107944 training samples, 1991 validation samples and 2491 test samples. After keeping top 90 percent most frequent words from training data, the vocabulary size was 54235. 

## Attention Sum Model
Two embeddings are constructed: context (document) embedding and question embedding. These embedding are calculated using bi-directional GRUs. A dot product between the contextual embedding of each word in the context is then calculated with the question embedding (For question embedding, only the output of the last RNN unit is used). This dot product gives the score for each word to be the correct answer of the question. Using a softmax function, the scores are converted to probabilities. This gives the probability of each word at a given position in the document to be the answer. The probabilities for a word are then summed over all positions in the document, given the probability of the word (irrespective of where it occurs in the ducument) to be the answer. In doing this, we also maintain a lookup table for each word in the vocabulary. This lookup table provides the embedding necessary for the input to the GRUs. The question and the context embeddings share the same lookup table. 

For the detail of the model, please refer to the original paper. 

## Dynet
Dynet, which was introduced by [Neubig et al.](https://arxiv.org/abs/1701.03980), is used to implement the attention sum model here. From a personal view, the syntax of Dynet allows for simpler, easily refactorable and writing reusable code, allowing for good software development practices. Using Dynet's auto-batching functionality, the complexity of NLP models, where the input sentences/chars are inevitably of differnet size, greatly reduces code complexity. Also it allows to write a more Pythonic code, for example using list comprehensions instead of scan functions as in Theano/Tensorflow. 

## Further improvement
* I have not yet been able to figure out saving/loading in Dynet (pull requests/suggestions are very welcome). 
* One thing to try is to feed the question embedding to the initial state of the contextual embedding. This mimics the intuition that it is easier to find the answer if one reads the question first and then reads the passage with the aim to only answer the question and not attempt to understand all the un-necessary contexts which might be present in the passage.
* Although Dynet's auto-batching is good, as Dynet's document suggests, it would help to explicitly use batching to further speed up training.  

## Result
As each sample has 10 candidates to choose from, a model which chooses the answer at random will acheive atleast 10% accuracy. We achieved **59.29%** test accuracy on the CBT-NE dataset, indicating the model has learnt something as it is definitely doing better than the base line of 10% (yay)! A reason for the accuracy to be lesser than **68.6%**, as reported in the original paper, can be because we used a smaller model having lesser model capacity. In the current implementation, the GRU hidden layer and the embedding dimention each had a size of 128, whereas in the original paper, the best accuracy is reported with a size of 384. We used a smaller model to help speedup training.

## Parameters
The above accuracy was acheived using the following parameters:
```
emb_dim = 128
gru_layers = 1
gru_input_dim = 128
gru_hidden_dim = 128
num_of_unk = 10

adam_alpha = 0.001
minibatch_size = 16
n_epochs = 2
gradient_clipping_thresh = 10.0

keep_top_vocab_percentage = 90.0
```

## Code structure
Usage:
```
python main.py --dynet-mem 5000 --dynet-autobatch 1 --dynet-seed 42
```

The bulk of the work is done by the following files:
```
main.py: Execution starts here
data_utils.py: Class for preparing data in a form needed while training
ASReaderTrainer.py: Class for training attention sum reader
ASReaderModel.py: Class creating the model-parameters which are optimized while training
```
The other files: configWriter.py, ASReaderConfig.py and ASReader.cfg are only for documenting, reading and keeping track of hard-coded values like file locations, training-parameters and model hyper-parameters. 

