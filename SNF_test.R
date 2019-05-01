library(SNFtool)
library(ggplot2)
library(cluster)
library(mclust)
library(RColorBrewer)

read_graphs <- function(fname_list, gname_list, nodes) {
  
  gpath = "/home/soyeon1771/SNFtools/SNFtools/Graphs"
  
  graph_list = list()
  for (i in 1:length(fname_list)) {
    path = paste(gpath, fname_list[i], sep="/")
    adj = 1-data.matrix(read.table(path))
    colnames(adj) <- nodes
    rownames(adj) <- nodes
    
    graph_list[[gname_list[i]]] = adj
  }
  
  return(graph_list)
}

## set all the parameters
K = 40; # number of neighbors, usually (10~30)
alpha = 0.5; # hyperparameter, usually (0.3~0.8)
T = 20; # Number of Iterations, usually (10~20)
C = 50 # number of clusters


# read nodes and ground truth labels
nodes = read.delim('/home/soyeon1771/SNFtools/SNFtools/inters_3789.txt', sep=" ", header=F)
trues = as.numeric(read.delim('/home/soyeon1771/SNFtools/SNFtools/trues_3789.txt', sep=" ", header=F))

# read text graph
# The edges should be distances (diag = 0)
text_fname_list = c("adj_socialtfidf_jaccardvit_3789.txt", 
                    "adj_word2vec_vflickr_3789.txt", 
                    "adj_word2vec_venwiki_3789.txt")

img_fname_list = c("adj_HOG_cosine_3789.txt",
                   "adj_VGG_cosine_3789.txt",
                   "adj_Res152-Caffe_cosine_3789.txt",
                   "adj_LeNet_cosine_3789.txt")

text_gname_list = c("socialTF-IDF",
                    "Word2Vec-Flickr",
                    "Word2Vec-Wiki")

img_gname_list = c("HOG",
                   "VGGnet",
                   "ResNet",
                   "LeNet")

text_graph_list = read_graphs(text_fname_list, text_gname_list, nodes)
img_graph_list = read_graphs(img_fname_list, img_gname_list, nodes)

gaussian_kernel <- function(graph, alpha) {

  g = as.matrix(graph)
  W = exp(-g/(2*alpha^2))
  return(W)
}


eval <- function(W, C, measure, trues) {
  labels = spectralClustering(W, C)
  if(measure == "NMI") return(list(labels, calNMI(labels, trues)))
  else if(measure == "ARI") return(list(labels, adjustedRandIndex(labels, trues)))
}


# alpha_text = 0.2
#alpha_img = 0.2
sink('output-fusednetwork.txt')
for (alpha_text in seq(0.2,0.6,0.1)) 
{
    for (alpha_img in seq(0.1,0.6,0.1))
  # for (alpha_img in seq(0.1,1,0.1)) {
    
    #W_text = affinityMatrix(text_graph_list[[1]], K, alpha_text)
    #W_img = affinityMatrix(img_graph_list[[3]], K, alpha_img)
    
    W_text = gaussian_kernel(text_graph_list[[1]], alpha_text)
    W_img = gaussian_kernel(img_graph_list[[3]], alpha_img)
    
    W_fused = SNF(list(W_text, W_img), K, T)
    
    res_fused_ARI=eval(W_fused, C, "ARI", trues)
    #displayClusters(W_fused, res_fused[[1]])
    max_SNF_ARI = res_fused_ARI[[2]]
    
    res_fused_NMI=eval(W_fused, C, "NMI", trues)
    #displayClusters(W_fused, res_fused[[1]])
    max_SNF_NMI = res_fused_NMI[[2]]
    
    print(list(alpha_text,alpha_img,max_SNF_ARI,max_SNF_NMI))
  # }
  
}
