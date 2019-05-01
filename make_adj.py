from elementtree.ElementTree import Element, ElementTree, SubElement, dump, parse, tostring  
import collections
import math
from nltk import cluster
import numpy as np
import random
import matplotlib.axes as ax
import networkx as nx
import scipy as sp
import statistics as st
import matplotlib.pyplot as plt
import csv
import re
from itertools import groupby
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering, KMeans
from sklearn import metrics
from sklearn import preprocessing

import time
import SNF
import displayClusters
import affinityMatrix
import gc
from matplotlib.pyplot import imshow
from numpy import linalg as la
import sys
from functions import *

import gensim, logging

class TxtDict(object):
	def __init__(self, topics, tagdict, txtdict, vit_tagdict, w2v_model):
		self.topics = topics
		self.tagdict = tagdict
		self.txtdict = txtdict
		self.viterbi_tagdict = vit_tagdict
		self.word2vec_model = w2v_model


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec.load("enwiki.w2v.model")
print 'enwiki word2vec model load complete..'

path = "/home/soyeon1771/SNFtools/Div400/devset/devsetkeywords/"

topics = read_topic(path)

# tagdict['photo id'] = ['tag1', 'tag2', 'tag3', ...]
# nodes = ['photo_id1', 'photo_id2', ...]
tagdict, nodes = get_tags_all(path, topics)
#print 'tag sets ********************'
#print tagdict
txtdict = get_textdescs_all(path, topics)
#print 'social-tfidf descriptors ******************'
#print txtdict

viterbi_segmented_tagdict = get_viterbi_tagdict(tagdict, nodes)
#print 'viterbi segmented tag sets **********************'
#print viterbi_segmented_tagdict

Dicts = TxtDict(topics, tagdict, txtdict, viterbi_segmented_tagdict, model)
print Dicts

inter_nodes_sub20 = np.loadtxt('inters_sub20.txt', dtype=np.dtype(str))
#print 'photo ids ***********************'
#print inter_nodes_sub20

trues_sub20 = np.loadtxt('trues_sub20.txt')
#print 'true labels ******************'
#print trues_sub20

adj = make_graph(Dicts, inter_nodes_sub20, 'wjaccard_vit', '0')
adj = make_graph(Dicts, inter_nodes_sub20, 'word2vec', '0')
#adj = make_graph(tagdict, viterbi_segmented_tagdict, txtdict, inter_nodes_sub20, 'word2vec', '0')
#np.savetxt('Graphs/adj_word2vec_sub20.txt', np.nan_to_num(adj))

print 'make word2vec text graph done'



