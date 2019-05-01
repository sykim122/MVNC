import gensim, logging
import sys
import os.path
 
logging.basicconfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.warning)

# load model from my corpus

#class MySentences(object):
#    def __init__(self, fname):
#        self.fname = fname

#    def __iter__(self):
#        for line in open(os.path.join(self.fname)):
#                yield line.split()
 
#sentences = MySentences('wiki.txt') 
#model = gensim.models.Word2Vec(sentences, min_count=3, window=3)
#model.save('wiki_25loc_model')
#model = gensim.models.Word2Vec.load('wiki_25loc_model')
#print model



# load model from pretrained corpus

#model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#model.save('GoogleNews_model')

#model = gensim.models.Word2Vec.load('freebase_model')

model = gensim.models.Word2Vec.load("enwiki.w2v.model")

print 'enwiki word2vec model load complete..'



#print model['saint_gall']


