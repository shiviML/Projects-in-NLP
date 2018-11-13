import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/September_11_attacks').read()

soup = bs.BeautifulSoup(source,'lxml')

# Fetching the data
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\[[a-z]*\]',' ',text)
text = re.sub(r'\s+',' ',text)

#Now find important words 
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)

stop_words = nltk.corpus.stopwords.words('english')

word2count = {}

for word in nltk.word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] +=1

#Print top 10 words 
sorted_by_value = sorted(word2count, key=word2count.get, reverse=True)
print("Below are top 10 words in article :")

for i in range(10):
    print(sorted_by_value[i])


sentences = nltk.sent_tokenize(text)

#normalise word2count values
word2count_normal = { k : word2count[k]/max(word2count.values()) for k,v in word2count.items()}

sentences_score ={}

for sentence in sentences:
    score_ = 0
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count_normal.keys():
            if len(sentence.split(' ')) < 20:
                score_ += word2count_normal[word]
    sentences_score[sentence] = score_

#Print top 10 sentences
sorted_by_value_sentences = sorted(sentences_score, key=sentences_score.get, reverse=True)
print("Below are top 10 words in article :")
for i in range(10):
    print(sorted_by_value_sentences[i])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
# Creating short summary based on first 10 sentences:

summary = str(sorted_by_value_sentences[:10])


        