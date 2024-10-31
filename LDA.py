import random
random.seed(0)  
from collections import Counter
import nltk
import matplotlib.pyplot as plt
import re
from re import RegexFlag
from wordcloud import WordCloud 

class LDA(object):
   
    def __init__(self, K, max_iteration):
        self.K = K
        self.max_iteration = max_iteration  
        
    def sample_from_weights(self, weights):
        total = sum(weights)
        rnd = total * random.random()
        for i, w in enumerate(weights):
            rnd -= w  
            if rnd <= 0: return i  

    def p_topic_given_document(self, topic, d, alpha=0.1):   
        return ((self.document_topic_counts[d][topic] + alpha) / 
                (self.document_lengths[d] + self.K * alpha))
    
    def p_word_given_topic(self, word, topic, beta=0.1):
        return ((self.topic_word_counts[topic][word] + beta) / 
                (self.topic_counts[topic] + self.W * beta))
    
    def topic_weight(self, d, word, topic):    
        return self.p_word_given_topic(word, topic) * self.p_topic_given_document(topic, d)
    
    def choose_new_topic(self, d, word):
        return self.sample_from_weights([self.topic_weight(d, word, k)
                            for k in range(self.K)])
    
    def gibbs_sample(self, document_topics):
        for _ in range(self.max_iteration):
            for d in range(self.D):
                for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):        
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[d] -= 1
        
                    new_topic = self.choose_new_topic(d, word)
                    document_topics[d][i] = new_topic
        
                    self.document_topic_counts[d][new_topic] += 1
                    self.topic_word_counts[new_topic][word] += 1
                    self.topic_counts[new_topic] += 1
                    self.document_lengths[d] += 1
        

    def run(self, documents):  
        self.document_topic_counts = [Counter() for _ in documents]
        self.topic_word_counts = [Counter() for _ in range(self.K)]
        self.topic_counts = [0 for _ in range(self.K)]
        self.document_lengths = [len(d) for d in documents]        
        self.distinct_words = set(word for document in documents for word in document)
        self.W = len(self.distinct_words)
        self.D = len(documents)      
        document_topics = [[random.randrange(self.K) for word in document]
                           for document in documents]
        
        for d in range(self.D):
            for word, topic in zip(documents[d], document_topics[d]):
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.topic_counts[topic] += 1
        
        self.gibbs_sample(document_topics)
        return(self.topic_word_counts, self.document_topic_counts)        
             
    def plot_words_clouds_topic(self, plt):
        for topic in range(self.K):
            data = []   
            text = ""
            for word, count in self.topic_word_counts[topic].most_common():
                if count > 1: 
                    data.append(word) 
            text = ' '.join(data)
            wordcloud = WordCloud().generate(text)  
            plt.figure()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("Topic")
            plt.show()

documents = [
    ['The sky is blue and beautiful.'],
    ['Love this blue and beautiful sky!'],        
    ['The quick brown fox jumps over the lazy dog.'],        
    ["A king's breakfast has sausages, ham, bacon, eggs, toast and beans"],        
    ['I love green eggs, ham, sausages and bacon!'],        
    ['The brown fox is quick and the blue dog is lazy!'],        
    ['The sky is very blue and the sky is very beautiful today'],        
    ['The dog is lazy but the brown fox is quick!']        
]

def pre_process_documents(doc):
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    for i in range(len(doc)):            
        doc[i][0] = re.sub(r'[^a-zA-Z\s]', '', doc[i][0], flags=RegexFlag.IGNORECASE | RegexFlag.A)
        doc[i][0] = doc[i][0].lower()
        doc[i][0] = doc[i][0].strip()
        tokens = wpt.tokenize(doc[i][0])
        filtered_tokens = [token for token in tokens if token not in stop_words]       
        doc[i] = filtered_tokens
    return filtered_tokens

pre_processed_documents = pre_process_documents(documents)

K = 3 
max_iteration = 1000
print(documents)

lda = LDA(K, max_iteration)
lda.run(documents) 
lda.plot_words_clouds_topic(plt)  

