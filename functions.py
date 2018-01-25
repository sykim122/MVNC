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
import itertools
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
import os, fnmatch
from IPython.display import Image, display
import glob

import igraph
from igraph import *
import gensim, logging

def read_data():
    # load word2vec model 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    flickr_model = gensim.models.Word2Vec.load("flickr4m_skipgram.model")
    print 'flickr word2vec model load complete..'

    enwiki_model = gensim.models.Word2Vec.load("enwiki.w2v.model")
    print 'enwiki word2vec model load complete..'

    print flickr_model, enwiki_model

    # Whole path set
    paths = ["/home/soyeon1771/SNFtools/Div400/devset/devsetkeywords/",
             "/home/soyeon1771/SNFtools/Div400/devset/devsetkeywordsGPS/",
             "/home/soyeon1771/SNFtools/Div400/testset/testset_keywords/",
             "/home/soyeon1771/SNFtools/Div400/testset/testset_keywordsGPS/"]

    # load dictionaries 

    topics=[]
    tagdict={}
    nodes=[]
    txtdict={}
    imdict_HOG={}
    imdict_VGG={}
    imdict_Res152={}
    imdict_LeNet={}

    true_name = "trues.txt"
    inters_name = "inters.txt"

    try:
        os.remove("big.txt")
        os.remove(true_name)
        os.remove(inters_name)
    except OSError:
        pass

    for i in range(0, len(paths)):


        topics_ = read_topic(paths[i])
        topics.extend(topics_)
        print len(topics_)
        # tagdict['photo id'] = ['tag1', 'tag2', 'tag3', ...]
        tagdict_, nodes_ = get_tags_all(paths[i], topics_)
        tagdict.update(tagdict_)
        nodes.extend(nodes_)
        print len(tagdict_), len(nodes_)

        txtdict_ = get_textdescs_all(paths[i], topics_)
        txtdict.update(txtdict_)
        print len(txtdict_)


        get_bigtxt(paths[i], topics_)

        get_gt_inters(paths[i], topics_, nodes_, true_name, inters_name)

        imdict_HOG.update(get_imgdesc(paths[i], topics_, "HOG"))
        imdict_VGG.update(get_imgdesc(paths[i], topics_, "VGGnet"))
        imdict_Res152.update(get_imgdesc(paths[i], topics_, "Res152net-Caffe"))
        imdict_LeNet.update(get_imgdesc(paths[i], topics_, "LeNet"))

    viterbi_segmented_tagdict = get_viterbi_tagdict(tagdict, nodes, 'big.txt')

    return flickr_model, enwiki_model, paths, topics, tagdict, viterbi_segmented_tagdict, nodes, txtdict, imdict_HOG, imdict_VGG, imdict_Res152, imdict_LeNet

    # 392 topics, 42479 tag sets(42945 nodes), 41001 tags

#    with open('data.pickle', 'w') as f:  
#        pickle.dump([flickr_model, enwiki_model, paths, topics, tagdict, viterbi_segmented_tagdict, nodes, txtdict, imdict_HOG, imdict_VGG, imdict_Res152, imdict_LeNet], f)

    
    
def display_image(path, pid):              
    for dir in os.listdir(path+'img/'):
        for file in os.listdir(path+'img/'+dir):
            if fnmatch.fnmatch(file, pid+'.*'):
                print dir
                display(Image(filename=path+'img/'+dir+'/'+file))
    
        
# read topic file
def read_topic(path):

    topics = []
    
    topicpath = glob.glob(path+"*_topics.xml")
    topicXML = open(topicpath[0], 'r')
    tree = parse(topicXML)
    root = tree.getroot()

    for element in root.findall('topic'):
        topics.append(element.findtext('title'))

        #print '******* Topics *********'
        #print topics
        #print len(topics)
        #print '************************'

    return topics

def get_textdescs_all(path, topics):
    desc = collections.defaultdict(list)
    txtdesc_path = glob.glob(path+"desctxt/*social-tfidf.txt") # social tf-idf weighting
    with open(txtdesc_path[0], 'r') as text:
        for line in text:
            for topic in topics:
                if line.startswith(topic):
                    descs = line.lstrip(topic).split()
                    desc.update(zip(descs, descs[1:])[::2])
                    break
    #print topics[0]
    #print desc
    
    return desc

def get_textdescs_query(path, topic):
    desc = collections.defaultdict(list)
    txtdesc_path = path + "desctxt/devsetkeywords-social-tfidf.txt" # social tf-idf weighting
    with open(txtdesc_path, 'r') as text:
        for line in text:
            if line.startswith(topic):
                descs = line.lstrip(topic).split()
                desc.update(zip(descs, descs[1:])[::2])
                break
    #print topics[0]
    #print desc
    
    return desc

def get_imgdesc(path, topics, desc):
    
    pids = []
    plist = []
    for topic in topics:
        targetpath = path+"descvis/img/"+topic+" "+desc+".csv"
        csvfile = open(targetpath, 'rb')
        targetCSV = csv.reader(csvfile, delimiter=',')
        for row in targetCSV:
            pids.append(row[0])
            plist.append(map(float, row[1:]))
            #print len(plist)

    imdict = dict(zip(pids, plist))

    print len(plist)
    print len(imdict)
    
    return imdict        
            
def get_tags_all(path, topics):
    #vec = []
    nodes = []
    tagdict = {}
    for topic in topics:
        targetpath = path+"xml/"+topic+".xml"
        targetXML = open(targetpath, 'r')
        tree = parse(targetXML)
        root = tree.getroot()

        for neighbor in root.getiterator('photo'):
            #rank = neighbor.attrib.get('rank')
            photoid = neighbor.attrib.get('id') ## -------> add node with phto id
            tags = neighbor.attrib.get('tags')
            tagvec = tags.split()
            #if not len(tagvec) == 0:
            #vec.append(tagvec)
            tagdict[photoid] = tagvec
            nodes.append(photoid)
            
    #tagdict = dict(zip(nodes, vec))
    #print len(tagdict), len(nodes)
    #return vec, nodes
    
    return tagdict, nodes

def get_tags_query(path, topic):
    vec = []
    nodes = []
    targetpath = path+"xml/"+topic+".xml"
    targetXML = open(targetpath, 'r')
    tree = parse(targetXML)
    root = tree.getroot()

    #print topic
    for neighbor in root.getiterator('photo'):
        #rank = neighbor.attrib.get('rank')
        photoid = neighbor.attrib.get('id') ## -------> add node with phto id
        tags = neighbor.attrib.get('tags')
        tagvec = tags.split()
        #if not len(tagvec) == 0:
        vec.append(tagvec)
        nodes.append(photoid)
    
    tagdict = dict(zip(nodes, vec))
    #print vec
    #return vec, nodes
    return tagdict, nodes

def viterbi_segment(text, dictionary, max_word_length, total):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i], dictionary, total), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    #return words, probs[-1]
    return words

def word_prob(word, dictionary, total): return dictionary.get(word, 0) / total
def words(text): return re.findall('[a-z]+', text.lower()) 

def get_dict(dictfile):
    dictionary = {}
    with open(dictfile) as myfile:
        for line in myfile:
            for w, ws in itertools.groupby(sorted(words(line))):
                dictionary[w] = len(list(ws))
    
    #txt = open(dictfile).read() 
    #dictionary = dict((w, len(list(ws))) for w, ws in itertools.groupby(sorted(words(txt))))
    max_word_length = max(map(len, dictionary))
    total = float(sum(dictionary.values()))
    
    return dictionary, max_word_length, total

def get_bigtxt(path, topics):
    
    f = open("big.txt", 'a')
    vec = []
    for topic in topics:
        targetpath = path+"xml/"+topic+".xml"
        targetXML = open(targetpath, 'r')
        tree = parse(targetXML)
        root = tree.getroot()

        #print topic
        for neighbor in root.getiterator('photo'):
            desc = neighbor.attrib.get('description')
            f.write(desc.encode('utf8') + "\n")

    f.close()

def get_viterbi_tagdict(tagdict, nodes, dictfile):
    viterbi_segmented_tagdict = {}
    dictionary, max_word_length, total = get_dict(dictfile)

    for i in np.arange(len(nodes)):
        N = nodes[i]
        viterbi_segmented_tagdict[N]=[]
        for tag in tagdict[N]: 
            viterbi_segmented_tagdict[N].append(viterbi_segment(tag, dictionary, max_word_length, total))
    return viterbi_segmented_tagdict

def normalize(v):
    vmean = st.mean(v)
    return [v[i]-vmean for i in range(len(v))]

def tags2descs(tagset, txtdict):
    vec = []
    for tag in tagset:
        if tag in txtdict:
            vec.append(np.float64(txtdict[tag]))
        else:
            vec.append(0)
    
    return vec

def viterbi_intersection(A,B,vA,vB):
    inter = []
    for i in np.arange(len(vA)):
        for j in np.arange(len(vB)):
            if len(list(set(vA[i]) & set(vB[j]))) > 0:
                inter.append(A[i])
                inter.append(B[j])

    return list(set(inter)) 

def tag_similarity(pid1, pid2, Dicts, distance):
#def tag_similarity(pid1, pid2, tagdict, viterbi_segmented_tagdict, txtdict, distance, model=None):
    
    if distance != 'img':
        set1 = Dicts.tagdict[pid1]
        set2 = Dicts.tagdict[pid2]

        vset1 = Dicts.viterbi_tagdict[pid1]
        vset2 = Dicts.viterbi_tagdict[pid2]

        # convert tags to descriptor values 
        vec1 = tags2descs(set1, Dicts.txtdict)
        vec2 = tags2descs(set2, Dicts.txtdict)

    #print vec1, vec2
    if distance == 'wpearson':
        
        # normalize tag vector for each image
        vec1 = normalize(vec1)
        vec2 = normalize(vec2)
        #print vec1, vec2
        
        # intersection of tag vector of two images
        dict1 = dict(zip(set1, vec1))
        dict2 = dict(zip(set2, vec2))
        
        v1 = [dict1[k] for k in list(set(set1) & set(set2))]
        v2 = [dict2[k] for k in list(set(set1) & set(set2))]
        
        #v1 = [dict1[k] for k in inter_list]
        #v2 = [dict2[k] for k in inter_list]
        
        if not v1 and not v2:
            return 0
        else:
            return 1-cluster.util.cosine_distance(v1, v2)
        
        #return v1, v2
        
    elif distance == 'wjaccard':
        
        intersect = tags2descs( list(set(set1) & set(set2)), Dicts.txtdict )
        union = tags2descs( list(set(set1) | set(set2)), Dicts.txtdict )
        #print list(set(set1) | set(set2))
        #print intersect, union
        
        if sum(union)==0:
            return 1
        else:
            return sum(intersect)/sum(union)
        
    elif distance == 'wjaccard_vit':
        inter_list = viterbi_intersection(set1, set2, vset1, vset2)
        intersect = tags2descs( inter_list, Dicts.txtdict )
        union = tags2descs( list(set(set1) | set(set2)), Dicts.txtdict )
        
        if sum(union)==0:
            return 1
        else:
            return sum(intersect)/sum(union)

    elif distance == 'word2vec':
        vset1 = list(itertools.chain.from_iterable(vset1))
        vset2 = list(itertools.chain.from_iterable(vset2))

        vset1 = filter(lambda x: x in Dicts.word2vec_model.vocab, vset1)
        vset2 = filter(lambda x: x in Dicts.word2vec_model.vocab, vset2)

        if (not vset1) or (not vset2):
            return 0
        else:
            return Dicts.word2vec_model.n_similarity(vset1, vset2)
        
    elif distance == 'img':
        return metrics.pairwise.cosine_similarity(Dicts.imdict[pid1], Dicts.imdict[pid2])

def make_graph(Dicts, nodes, distm, thres):
#def make_graph(tagdict, viterbi_segmented_tagdict, txtdict, nodes, distm, thres):
    
    #path = 'tag_network_'+distm+'_'+thres+'.graphml'
    
    #G=nx.Graph()
    
    #random.seed(1)
    nodelen = len(nodes)
    adj = np.zeros(shape=(nodelen, nodelen), dtype=np.float64)
    
    print distm

    #G.add_nodes_from(nodes)
    start = time.time()
    for r in xrange(0, nodelen):
        for c in xrange(0, nodelen):
            sim = tag_similarity(nodes[r], nodes[c], Dicts, distm)
            #sim = tag_similarity(nodes[r], nodes[c], tagdict, viterbi_segmented_tagdict, txtdict, distm, model)
            adj[r][c] = sim
            
            #f.write("%s %s %f\n" % (nodes[r], nodes[c], sim))
            
            #if sim > float(thres) and math.isnan(sim) == False:
            #G.add_edge(nodes[r], nodes[c], weight="%.4f"%sim)
            
    print time.time()-start
    #adj = preprocessing.normalize(adj, norm='l2')
    #adj = preprocessing.MinMaxScaler().fit_transform(adj)
    
    #f.close()
    #return G
    return adj

def make_graph_GPU(Dicts, nodes, distm, thres):

    

    nodelen = len(nodes)
    adj = np.zeros(shape=(nodelen, nodelen), dtype=np.float64)
    
    print distm

    #G.add_nodes_from(nodes)
    start = time.time()
    for r in xrange(0, nodelen):
        for c in xrange(0, nodelen):
            sim = tag_similarity(nodes[r], nodes[c], Dicts, distm)
            #sim = tag_similarity(nodes[r], nodes[c], tagdict, viterbi_segmented_tagdict, txtdict, distm, model)
            adj[r][c] = sim
            
            #f.write("%s %s %f\n" % (nodes[r], nodes[c], sim))
            
            #if sim > float(thres) and math.isnan(sim) == False:
            #G.add_edge(nodes[r], nodes[c], weight="%.4f"%sim)
            
    print time.time()-start
    #adj = preprocessing.normalize(adj, norm='l2')
    #adj = preprocessing.MinMaxScaler().fit_transform(adj)
    
    #f.close()
    #return G
    return adj

def get_gt_inters(path, topics, nodes, true_name, inters_name):
    ind = 0
    photoid = []
    #photoid_sub20 = []

    clusterid = []

    gt = {}

    for topic in topics:
        ind = ind+1
        truepath = path+"gt/dGT/"+topic+" dGT.txt"
        #truepath = path+"gt/dGT/"+topics[topi]+" dGT.txt"

        f = open(truepath)
        line = f.readline().strip()

        tmp = []
        while line: 
            columns = line.split(',')
            photoid.append(columns[0])
            tmp.append(columns[0])

            #clusterid.append(columns[1])
            clusterid.append(ind)

            #gt[columns[0]] = int(columns[1])
            gt[columns[0]] = ind
            line = f.readline().strip()

        #print len(tmp)
        #if len(tmp) >= 20: 
        #    sub20 = np.random.choice(tmp, 20, replace=False)
        #    photoid_sub20.extend(sub20)  

        #print gt
        #print topics[0]

    #print len(photoid_sub20)

    inters = list(set(nodes) & set(photoid))
    #inters_sub20 = list(set(nodes) & set(photoid_sub20))

    trues = np.empty(len(inters))
    for i in range(len(inters)):
        trues[i] = gt[inters[i]]    
    
    #return trues, inters
    f_trues=open(true_name,'a')
    f_inters=open(inters_name,'a')
    np.savetxt(f_trues, trues, fmt='%d', newline=' ')
    np.savetxt(f_inters, inters, fmt='%s', newline=' ')
    
    #rues = np.empty(len(inters_sub20))
    #for i in range(len(inters_sub20)):
    #    trues[i] = gt[inters_sub20[i]]    
        
    #f_trues=open('trues_sub20_whole.txt','a')
    #f_inters=open('inters_sub20_whole.txt','a')
    #np.savetxt(f_trues, trues, fmt='%d', newline=' ')
    #np.savetxt(f_inters, inters_sub20, fmt='%s', newline=' ')