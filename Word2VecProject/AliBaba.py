# Importing the libraries
import bs4 as bs
import urllib.request
import re
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import gensim.models.word2vec as w2v
import sklearn.manifold
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import pandas as pd
# Gettings the data source
source = urllib.request.urlopen('https://www.pitt.edu/~dash/alibaba.html').read()

# Parsing the data/ creating BeautifulSoup object
soup = bs.BeautifulSoup(source,'lxml')

# Fetching the data
text = ""

start_point = soup.find('h3')
    
for elem in start_point.next_siblings:
    if elem.name and elem.name.startswith('h'):
                # stop at next header
            break
    if elem.name == 'p':
        text += elem.get_text()

#Preprocessing the data
text = text.lower()
text = re.sub(r"[!\"--,;]"," ",text)

sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(x) for x in sentences]

for i in range(len(sentences)):
        sentences[i] = [x for x in sentences[i] if x not in stopwords.words('english')]
    
model1 = w2v.Word2Vec(
    sg=1,
    size=200,
    min_count=2)
model1.build_vocab(sentences)
#print(model1.wv.vocab)

model1.train(sentences,total_examples=model1.corpus_count, epochs=100)

#squash dimensionality to 2
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = model1.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#plot point in 2d space
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model1.wv.vocab[word].index])
            for word in model1.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

#plot
sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
        
plot_region(x_bounds=(-4.0,1.0), y_bounds=(-8.0,-5.0 ))
model1.most_similar("ali")
model1.most_similar("morgiana")
model1.most_similar(positive=['ali','brother'])

