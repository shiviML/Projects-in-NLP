# Word2Vec and Web scrapping Demo

Reference for web scrapping:
https://www.crummy.com/software/BeautifulSoup/bs3/documentation.html

Ali Baba and 40 thieves story: 
https://www.pitt.edu/~dash/alibaba.html

###Results
After feeding Ali Baba and 40 thieves story to trained model (word2vec algorithm) below are the observations:
model1.most_similar("ali", topn=3)
>> [('baba', 0.9806886911392212),
 ('son', 0.9230267405509949),
 ('sup', 0.907829761505127)]
 
model1.most_similar("morgiana", topn=3) 
 >>[('guest', 0.8913780450820923),
 ('dance', 0.8810752630233765),
 ('answered', 0.8704050779342651)]
 
model1.most_similar(positive=['ali','brother'])
[('buried', 0.946819543838501),
 ('told', 0.946718156337738),
 ('husband', 0.9416701793670654),
 ('thus', 0.9339929223060608),
 ('ran', 0.9168856739997864)]