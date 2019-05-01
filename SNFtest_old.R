
# read text graph
# The edges should be distances (diag = 0)
#adj_socialtfidf_jaccard = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_socialtfidf_jaccard_sub20.txt"))
adj_socialtfidf_jaccardvit = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_socialtfidf_jaccardvit_sub20.txt"))
adj_socialtfidf_jaccardvit_flickr = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_socialtfidf_jaccardvit_sub20_flickrspacing.txt"))
adj_socialtfidf_jaccardvit_enwiki = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_socialtfidf_jaccardvit_sub20_enwikispacing.txt"))

#adj_socialtfidf_pearson = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_socialtfidf_pearson_sub20.txt"))
#adj_word2vec_flickr = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_word2vec_flickr_sub20.txt"))
adj_word2vec_vflickr = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_word2vec_vflickr_sub20_flickrspacing.txt"))
adj_word2vec_venwiki = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_word2vec_venwiki_sub20_enwikispacing.txt"))

# read image graph
#adj_HOG_euclidean = data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_HOG_euclidean_sub20.txt"))
adj_HOG_cosine = 1-data.matrix(read.table("/home/soyeon1771/SNFtools/SNFtools/Graphs/adj_HOG_cosine_sub20.txt"))

# read nodes and ground truth labels
nodes = read.delim('/home/soyeon1771/SNFtools/SNFtools/inters_sub20.txt', sep=" ", header=F)
trues = as.numeric(read.delim('/home/soyeon1771/SNFtools/SNFtools/trues_sub20.txt', sep=" ", header=F))


#colnames(adj_socialtfidf_jaccard) <- nodes
#rownames(adj_socialtfidf_jaccard) <- nodes

colnames(adj_socialtfidf_jaccardvit) <- nodes
rownames(adj_socialtfidf_jaccardvit) <- nodes

colnames(adj_socialtfidf_jaccardvit_flickr) <- nodes
rownames(adj_socialtfidf_jaccardvit_flickr) <- nodes

colnames(adj_socialtfidf_jaccardvit_enwiki) <- nodes
rownames(adj_socialtfidf_jaccardvit_enwiki) <- nodes

#colnames(adj_socialtfidf_pearson) <- nodes
#rownames(adj_socialtfidf_pearson) <- nodes

colnames(adj_HOG_cosine) <- nodes
rownames(adj_HOG_cosine) <- nodes

#colnames(adj_word2vec_flickr) <- nodes
#rownames(adj_word2vec_flickr) <- nodes

colnames(adj_word2vec_vflickr) <- nodes
rownames(adj_word2vec_vflickr) <- nodes

colnames(adj_word2vec_venwiki) <- nodes
rownames(adj_word2vec_venwiki) <- nodes


## set all the parameters
K = 20; # number of neighbors, usually (10~30)
alpha = 0.5; # hyperparameter, usually (0.3~0.8)
T = 10; # Number of Iterations, usually (10~20)

C = 22 # number of clusters

NMI = c()
ARI = c()

alphas = seq(0.4,2,0.1)
alpha = 0.8
for (alpha in alphas) {
  ## next, construct similarity graphs
  #W_jaccard = affinityMatrix(as.matrix(adj_socialtfidf_jaccard), K, alpha)
  W_vjaccard = affinityMatrix(as.matrix(adj_socialtfidf_jaccardvit), K, alpha)
  W_vjaccardflickr = affinityMatrix(as.matrix(adj_socialtfidf_jaccardvit_flickr), K, alpha)
  W_vjaccardenwiki = affinityMatrix(as.matrix(adj_socialtfidf_jaccardvit_enwiki), K, alpha)

  #W3 = affinityMatrix(as.matrix(adj_socialtfidf_pearson), K, alpha)
  #W_flickr = affinityMatrix(as.matrix(adj_word2vec_flickr), K, alpha)
  W_vflickr = affinityMatrix(as.matrix(adj_word2vec_vflickr), K, alpha)
  W_venwiki = affinityMatrix(as.matrix(adj_word2vec_venwiki), K, alpha)
  W_HOG = affinityMatrix(as.matrix(adj_HOG_cosine), K, alpha)

  ## These similarity graphs have complementary information about clusters.
  #par(mfrow=c(1,3))
  #displayClusters(W1,trues);
  #displayClusters(W2,trues);
  #displayClusters(W3,trues);
  #displayClusters(W_im,trues);

  ## next, we fuse all the graphs
  ## then the overall matrix can be computed by similarity network fusion(SNF):
  #W12 = SNF(list(W1,W2), K, T)
  #W13 = SNF(list(W1,W3), K, T)
  #W123 = SNF(list(W1,W2,W3), K, T)

  W_vjaccard_HOG = SNF(list(W_vjaccard,W_HOG), K, T)
  W_vjaccardflickr_HOG = SNF(list(W_vjaccardflickr,W_HOG), K, T)
  W_vjaccardenwiki_HOG = SNF(list(W_vjaccardenwiki,W_HOG), K, T)

  #W_123im = SNF(list(W1,W2,W3,W_im), K, T)
  #W_12im = SNF(list(W1,W2,W_im), K, T)
  W_vflickr_HOG= SNF(list(W_vflickr,W_HOG), K, T)
  W_venwiki_HOG= SNF(list(W_venwiki,W_HOG), K, T)

  ## You can get cluster labels for each data point by spectral clustering
  #labels_jaccard = spectralClustering(W_jaccard, C)
  labels_vjaccard = spectralClustering(W_vjaccard, C)
  labels_vjaccardflickr = spectralClustering(W_vjaccardflickr, C)
  labels_vjaccardenwiki = spectralClustering(W_vjaccardenwiki, C)

  #labels3 = spectralClustering(W3, C)
  #labels_flickr = spectralClustering(W_flickr, C)
  labels_vflickr = spectralClustering(W_vflickr, C)
  labels_venwiki = spectralClustering(W_venwiki, C)
  labels_HOG = spectralClustering(W_HOG, C)

  #labels12 = spectralClustering(W12, C)
  #labels13 = spectralClustering(W13, C)
  #labels123 = spectralClustering(W123, C)

  labels_vjaccard_HOG = spectralClustering(W_vjaccard_HOG, C)
  labels_vjaccardflickr_HOG = spectralClustering(W_vjaccardflickr_HOG, C)
  labels_vjaccardenwiki_HOG = spectralClustering(W_vjaccardenwiki_HOG, C)

  #labels_123im = spectralClustering(W_123im, C)
  #labels_12im = spectralClustering(W_12im, C)
  labels_vflickr_HOG = spectralClustering(W_vflickr_HOG, C)
  labels_venwiki_HOG = spectralClustering(W_venwiki_HOG, C)

  #plot.new()

  #displayClusters(W1,labels1);
  #displayClusters(W2,labels2);
  #displayClusters(W3,labels3);
  #displayClusters(W_im, labels_im);

  #displayClusters(W12,labels12);
  #displayClusters(W13,labels13);
  #displayClusters(W_2im,labels_2im);



  #plot(adj_socialtfidf_jaccard, col=labels, main='Data type 1')
  #plot(adj_socialtfidf_jaccardvit, col=labels, main='Data type 2')

  # Use NMI to measure the goodness of the obtained labels.
  # single view
  #cal = c(calNMI(labels1,trues), calNMI(labels2,trues), calNMI(labels3,trues), calNMI(labels12,trues), calNMI(labels13,trues), calNMI(labels123,trues))
  #cal_r = c(adjustedRandIndex(labels1, trues),adjustedRandIndex(labels2, trues),adjustedRandIndex(labels3, trues),adjustedRandIndex(labels12, trues),adjustedRandIndex(labels13, trues),adjustedRandIndex(labels123, trues))

  # single view + multi-view
  #cal = c(calNMI(labels1,trues), calNMI(labels2,trues), calNMI(labels3,trues), calNMI(labels12,trues), calNMI(labels13,trues), calNMI(labels123,trues), calNMI(labels_2im,trues), calNMI(labels_123im,trues))
  #cal_r = c(adjustedRandIndex(labels1, trues),adjustedRandIndex(labels2, trues),adjustedRandIndex(labels3, trues),adjustedRandIndex(labels12, trues),adjustedRandIndex(labels13, trues),adjustedRandIndex(labels123, trues),adjustedRandIndex(labels_2im,trues), adjustedRandIndex(labels_123im,trues))

  # multi-view
  #cal = c(calNMI(labels2,trues), calNMI(labels12,trues), calNMI(labels123,trues), calNMI(labels_im, trues), calNMI(labels_2im,trues), calNMI(labels_12im, trues), calNMI(labels_123im,trues))
  #cal_r = c(adjustedRandIndex(labels2, trues),adjustedRandIndex(labels12, trues),adjustedRandIndex(labels123, trues),adjustedRandIndex(labels_im, trues),adjustedRandIndex(labels_2im,trues), adjustedRandIndex(labels_12im,trues),adjustedRandIndex(labels_123im,trues))

  cal = c(calNMI(labels_vjaccard,trues),
          calNMI(labels_vjaccardflickr,trues),
          calNMI(labels_vjaccardenwiki,trues),
#          calNMI(labels_flickr,trues),
          calNMI(labels_vflickr, trues),
          calNMI(labels_venwiki,trues),
          calNMI(labels_HOG,trues),
          calNMI(labels_vjaccard_HOG,trues),
          calNMI(labels_vjaccardflickr_HOG,trues),
          calNMI(labels_vjaccardenwiki_HOG,trues),
          calNMI(labels_vflickr_HOG,trues),
          calNMI(labels_venwiki_HOG,trues))

  cal_r = c(adjustedRandIndex(labels_vjaccard, trues),
            adjustedRandIndex(labels_vjaccardflickr, trues),
            adjustedRandIndex(labels_vjaccardenwiki, trues),
#            adjustedRandIndex(labels_flickr, trues),
            adjustedRandIndex(labels_vflickr, trues),
            adjustedRandIndex(labels_venwiki,trues),
            adjustedRandIndex(labels_HOG,trues),
            adjustedRandIndex(labels_vjaccard_HOG,trues),
            adjustedRandIndex(labels_vjaccardflickr_HOG,trues),
            adjustedRandIndex(labels_vjaccardenwiki_HOG,trues),
            adjustedRandIndex(labels_vflickr_HOG,trues),
            adjustedRandIndex(labels_venwiki_HOG,trues))

  NMI = c(NMI, cal)
  ARI = c(ARI, cal_r)

}

# single view
#names = c('jaccard', 'jaccard(viterbi)', 'pearson', 'jaccard+jaccard(viterbi)', 'jaccard+pearson', 'All')
# single view + multi-view
#names = c('jaccard', 'jaccard(viterbi)', 'pearson', 'jaccard+jaccard(viterbi)', 'jaccard+pearson', 'All', 'jaccard(viterbi)+HOG', 'All+HOG')
# multi-view
#names = c('text(single best)', 'text(fused best)', 'text(all)', 'image', 'text(single best)+image', 'text(fused best)+image', 'text(all)+image')
names = c('text(vjaccard)',
          'text(vjaccardflickr)',
          'text(vjaccardenwiki)',
#          'text(word2vec_flickr)',
          'text(word2vec_vflickr)',
          'text(word2vec_venwiki)',
          'image(HOG)',
          'text(vjaccard)+image(HOG)',
          'text(vjaccardflickr)+image(HOG)',
          'text(vjaccardenwiki)+image(HOG)',
          'text(word2vec_flickr)+image(HOG)',
          'text(word2vec_enwiki)+image(HOG)')

type = rep(names, length(alphas))
alpha = rep(alphas, each=length(names))

#type = names
#alpha = rep(0.8, each=length(names))

data_NMI = data.frame(type, alpha, NMI)
data_ARI = data.frame(type, alpha, ARI)

ltypes = c("longdash","longdash","longdash","dotdash", "dotdash",
           "dashed", "solid","solid","solid", "twodash", "twodash")

cpalette <- brewer.pal(11, "Spectral")
#cpalette = c("#ff9999", "#0099cc", "#669900", "#ff0000", "#cc6600")

# line plot (w.r.t. alpha)
ggplot(data=data_NMI, aes(x=alpha, y=NMI, group=type, colour=type)) +
  geom_line(aes(linetype=type)) +
  geom_point() +
  scale_linetype_manual(values=ltypes) +
  scale_color_manual(values=cpalette) +
  ylim(0,1)

ggplot(data=data_ARI, aes(x=alpha, y=ARI, group=type, colour=type)) +
  geom_line(aes(linetype=type)) +
  geom_point() +
  scale_linetype_manual(values=ltypes) +
  scale_color_manual(values=cpalette) +
  ylim(0,1)

# bar plot

ggplot(data=data_NMI[data_NMI$alpha==0.8,], aes(x=reorder(type, NMI), y=NMI, fill=type)) +
  geom_bar(stat='identity') +
  scale_fill_manual(values=cpalette) +
  xlab('Network') +
  ylim(0,1) +
  geom_text(aes(label=round(NMI,2)), vjust=0.3, size=5)+
  theme(axis.text.x=element_text(angle=45, hjust=1, size=10))


ggplot(data=data_ARI[data_ARI$alpha==0.8,], aes(x=reorder(type, ARI), y=ARI, fill=type)) +
  geom_bar(stat='identity') +
  scale_fill_manual(values=cpalette) +
  xlab('Network') +
  ylim(0,1) +
  geom_text(aes(label=round(ARI,2)), vjust=0.3, size=5)+
  theme(axis.text.x=element_text(angle=45, hjust=1, size=10))


# wrtie networks to file
# nodes
jaccard_net_nodes = data.frame(node=unlist(nodes), label=labels1)
jaccardvit_net_nodes = data.frame(node=unlist(nodes), label=labels2)
jaccard_jaccardvit_net_nodes = data.frame(node=unlist(nodes), label=labels12)
HOG_net_nodes = data.frame(node=unlist(nodes), label=labels_im)
jaccardvit_HOG_net_nodes = data.frame(node=unlist(nodes), label=labels_2im)
word2vec_net_nodes = data.frame(node=unlist(nodes), label=labels_w2v)

write.table(jaccard_net_nodes, file='text_jaccard_nodes.csv',row.names = F,col.names = F)
write.table(jaccardvit_net_nodes, file='text_jaccardvit_nodes.csv',row.names = F,col.names = F)
write.table(jaccard_jaccardvit_net_nodes, file='text_jaccard_jaccardvit_nodes.csv',row.names = F,col.names = F)
write.table(HOG_net_nodes, file='image_HOG_nodes.csv',row.names = F,col.names = F)
write.table(jaccardvit_HOG_net_nodes, file='jaccardvit_HOG_nodes.csv',row.names = F,col.names = F)
write.table(word2vec_net_nodes, file='word2vec_nodes.csv',row.names = F,col.names = F)

# edges
colnames(W12) <- nodes
rownames(W12) <- nodes

colnames(W_2im) <- nodes
rownames(W_2im) <- nodes

jaccard_net_edges = melt(W1)
jaccardvit_net_edges = melt(W2)
jaccard_jaccardvit_net_edges = melt(W12)
HOG_net_edges = melt(W_im)
jaccardvit_HOG_net_edges = melt(W_2im)
word2vec_net_edges = melt(adj_word2vec)

write.table(jaccard_net_edges, file='text_jaccard_edges.csv',row.names = F,col.names = F)
write.table(jaccardvit_net_edges, file='text_jaccardvit_edges.csv',row.names = F,col.names = F)
write.table(jaccard_jaccardvit_net_edges, file='text_jaccard_jaccardvit_edges.csv',row.names = F,col.names = F)
write.table(HOG_net_edges, file='image_HOG_edges.csv',row.names = F,col.names = F)
write.table(jaccardvit_HOG_net_edges, file='jaccardvit_HOG_edges.csv',row.names = F,col.names = F)
write.table(word2vec_net_edges, file='word2vec_edges.csv',row.names = F,col.names = F)

plot.new()

displayClusters(W1,labels1);
displayClusters(W2,labels2);
#displayClusters(W3,labels3);
displayClusters(W_im, labels_im);

displayClusters(W12,labels12);
#displayClusters(W13,labels13);
displayClusters(W_2im,labels_2im);
