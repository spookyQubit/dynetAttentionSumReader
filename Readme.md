# Attention Sum Model
This is an implementation of the Attention Sum Model which was introduced by Kaldec et al. in the paper titeled [Text Understanding with the Attention Sum Reader Network](https://arxiv.org/pdf/1603.01547.pdf). The paper aims at solving cloze-style question answering problem. 

## Dataset
We train the model on the Children's Book Test (CBT) dataset. Each datapoint consists of twenty-one consecutive lines extracted from a book. The first twenty lines form the context. From the twenty first line, one named-entity is removed. The purpose of the model is to predict this missing named-entity out a list of possible candidates. 

An example datapoint:
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
