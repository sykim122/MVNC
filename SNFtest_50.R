library(SNFtool)
library(ggplot2)
library(cluster)
library(mclust)
library(RColorBrewer)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

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
K = 30; # number of neighbors, usually (10~30)
alpha = 0.5; # hyperparameter, usually (0.3~0.8)
T = 15; # Number of Iterations, usually (10~20)
C = 50 # number of clusters

# read nodes and ground truth labels
nodes = read.delim('/home/soyeon1771/SNFtools/SNFtools/inters_3789.txt', sep=" ", header=F)
trues = as.numeric(read.delim('/home/soyeon1771/SNFtools/SNFtools/trues_3789.txt', sep=" ", header=F))

# read text graph
# The edges should be distances (diag = 0)
text_fname_list = c("adj_socialtfidf_jaccardvit_3789.txt")

img_fname_list = c("adj_Res152-Caffe_cosine_3789.txt")
                   
text_gname_list = c("socialTF-IDF")

img_gname_list = c("ResNet")

text_graph_list = read_graphs(text_fname_list, text_gname_list, nodes)
img_graph_list = read_graphs(img_fname_list, img_gname_list, nodes)


affinity <- function(graph, K, alpha) {
  
  W = affinityMatrix(as.matrix(graph), K, alpha)
  return(W)
}

gaussian_kernel <- function(graph, alpha) {
  
  g = as.matrix(graph)
  W = exp(-g/(2*alpha^2))
  
  return(W)
}

write_affinity <- function(W, gname) {
  write.table(W, file=paste(gname, "_kernel.txt", sep=""), row.names = F,col.names = F)
}

eval <- function(W, C, measure, trues) {
  labels = spectralClustering(W, C)
  if(measure == "NMI") return(list(labels, calNMI(labels, trues)))
  else if(measure == "ARI") return(list(labels, adjustedRandIndex(labels, trues)))
}

single <- function(W, W_names, C, measure, trues) {
  score = c()
  for (name in W_names) {
    score = c(score, eval(W[[name]], C, measure, trues)[[2]])
  }
  return(score)
}

fused <- function(W1, W1_names, W2, W2_names, K, T, C, measure, trues) {
  score = c()
  for (txt in W1_names) {
    for (img in W2_names) {
      start=Sys.time()
      W12 = SNF(list(W1[[txt]], W2[[img]]), K, T)
      print(Sys.time()-start)
      score = c(score, eval(W12, C, measure, trues)[[2]])
    }
  }
  return(score)
}


#scores = c()
# tscores = c()
# iscores = c()
# 
# alphas = seq(0.2,2,0.1)
# for (alpha in alphas) {
#   
#   #W_text = affinity(text_graph_list, text_gname_list, K, alpha)
#   #W_img = affinity(img_graph_list, img_gname_list, K, alpha)
#   W_text = gaussian_kernel(text_graph_list, text_gname_list, alpha)
#   W_img = gaussian_kernel(img_graph_list, img_gname_list, alpha)
#     
#   scores_text = single(W_text, text_gname_list, C, "NMI", trues)
#   tscores = c(tscores, scores_text)
#   
#   try({scores_img = single(W_img, img_gname_list, C, "NMI", trues); iscores = c(iscores, scores_img)}, T)
#   
#   
#   #scores_fused = fused(W_text, text_gname_list, W_img, img_gname_list, K, T, C, "NMI", trues)
#   
#   #scores = c(scores, c(scores_text, scores_img, scores_fused))
#   
# }

######################################
# Plot graphs (single-view)

text_scores = read.delim('/home/soyeon1771/SNFtools/SNFtools/baseSC/result_txt_singlebaseSC_gkernel_3789.txt', sep="\t", header=F, 
                         col.names=c('type','alpha','ARI','NMI'))

image_scores = read.delim('/home/soyeon1771/SNFtools/SNFtools/baseSC/result_img_singlebaseSC_gkernel_3789.txt', sep="\t", header=F, 
                          col.names=c('type','alpha','ARI','NMI'))


# text_scores = data.frame(type=rep(text_gname_list, length(alphas)),
#                          alpha=rep(alphas, each=length(text_gname_list)),
#                          scores=tscores)
# 
# image_scores = data.frame(type=rep(img_gname_list, length(alphas)),
#                          alpha=rep(alphas, each=length(img_gname_list)),
#                          scores=iscores)

# single best(text) line plot w.r.t. alpha(scaling parameter of affinity matrix)
g1 <- ggplot(data=text_scores, aes(x=alpha, y=NMI, group=type, colour=type)) + 
  theme_bw() +
  geom_line(aes(linetype=type, color=type), size=1.2) + 
  geom_point(size=1.5) + 
  scale_linetype_manual(values=c('twodash','longdash','solid')) +
  scale_color_manual(values=c('darkgreen', 'darkblue','red')) +
  ylab("NMI score")+
  theme(legend.position = "bottom", 
        plot.title = element_text(size = rel(3), face = "bold"),
        legend.text=element_text(size=15),
        axis.title.x = element_text(size = rel(2)),
        axis.title.y = element_text(size = rel(2))) +
  ylim(0,1)

# single best(image) line plot w.r.t. alpha(scaling parameter of affinity matrix)
g2 <- ggplot(data=image_scores, aes(x=alpha, y=NMI, group=type, colour=type)) + 
  theme_bw() +
  geom_line(aes(linetype=type), size=1.2) + 
  geom_point(size=1.5) + 
  scale_linetype_manual(values=c('twodash','longdash','solid','dashed')) +
  scale_color_manual(values=c('darkorange','purple','red','darkgreen')) +
  ylab("NMI score")+
  theme(legend.position = "bottom", 
        plot.title = element_text(size = rel(3), face = "bold"),
        legend.text=element_text(size=15),
        axis.title.x = element_text(size = rel(2)),
        axis.title.y = element_text(size = rel(2))) +
  ylim(0,1)

multiplot(g1, g2, layout=matrix(c(1,2), nrow=1))


###############################
# Plot graph (all combinations)

fused_names = c()
for (txt in text_gname_list) {
  for (img in img_gname_list) {
    fused_names = c(fused_names, paste(txt, img, sep="+"))
  }
}

names = c(text_gname_list, img_gname_list, fused_names)
type = rep(names, length(alphas))
alpha = rep(alphas, each=length(names))

data_scores = data.frame(type, alpha, scores)

ltypes = rep(c("longdash","twodash","solid"), c(length(text_gname_list), length(img_gname_list), length(fused_names)))

cpalette <- c(brewer.pal(length(text_gname_list), "Set2"), 
              brewer.pal(length(img_gname_list), "Dark2"), 
              brewer.pal(length(fused_names), "Set3"))

# line plot w.r.t. alpha(scaling parameter of affinity matrix)
ggplot(data=data_scores, aes(x=alpha, y=scores, group=type, colour=type)) + 
  geom_line(aes(linetype=type)) + 
  geom_point() + 
  scale_linetype_manual(values=ltypes) +
  scale_color_manual(values=cpalette) +
  ylim(0.25,1)


# bar plot at alpha=0.8
ggplot(data=data_scores[data_scores$alpha==0.5,], aes(x=reorder(type, scores), y=scores, fill=type)) + 
  geom_bar(stat='identity') +
  scale_fill_manual(values=cpalette) +
  xlab('Network') +
  ylim(0,1) +
  geom_text(aes(label=round(scores,3)), vjust=0.3, size=5)+
  theme(axis.text.x=element_text(angle=45, hjust=1, size=10))


######################################
# NMI best score
#head(data_scores[order(data_scores$alpha==0.2, -scores),])

head(text_scores[order(-text_scores$NMI),]) # 0.2
head(image_scores[order(-image_scores$NMI),]) # 0.5

#head(data_scores[order(-data_scores$scores),])

# NMI/ARI score at best alpha
# W_text = gaussian_kernel(text_graph_list, text_gname_list, 0.2)
# res_text_ARI=eval(W_text[[text_gname_list[1]]], C, "ARI", trues)
# single_text_ARI = res_text_ARI[[2]]
# 
# res_text_NMI=eval(W_text[[text_gname_list[1]]], C, "NMI", trues)
# #displayClusters(W_text[[text_gname_list[1]]], res[[1]])
# single_text_NMI = res_text_NMI[[2]]

single_text_ARI = text_scores[text_scores$alpha==0.4, 'ARI'][1]
single_text_NMI = text_scores[text_scores$alpha==0.4, 'NMI'][1]
print(list(single_text_ARI,single_text_NMI))


# W_img = gaussian_kernel(img_graph_list, img_gname_list, 0)
# res_img_ARI=eval(W_img[[img_gname_list[3]]], C, "ARI", trues)
# single_img_ARI = res_img_ARI[[2]]
# 
# res_img_NMI=eval(W_img[[img_gname_list[3]]], C, "NMI", trues)
# #displayClusters(W_img[[img_gname_list[3]]], res[[1]])
# single_img_NMI = res_img_NMI[[2]]

single_img_ARI = image_scores[image_scores$alpha==0.2, 'ARI'][3]
single_img_NMI = image_scores[image_scores$alpha==0.2, 'NMI'][3]
print(list(single_img_ARI,single_img_NMI))

alpha_text = 0.4
alpha_img = 0.2

W_text = gaussian_kernel(text_graph_list[[1]],alpha_text)
W_img = gaussian_kernel(img_graph_list[[1]],alpha_img)
W_fused = SNF(list(W_text, W_img), K, T)

res_fused_ARI=eval(W_fused, C, "ARI", trues)

max_SNF_ARI = res_fused_ARI[[2]]

res_fused_NMI=eval(W_fused, C, "NMI", trues)

max_SNF_NMI = res_fused_NMI[[2]]

print(list(max_SNF_ARI,max_SNF_NMI))

#write.table(W_fused, file='test_fused.txt', sep = "\t", row.names=F,col.names=F)
#write.table(res_fused_NMI[[1]], file='test_fused_label.txt', sep = "\t", row.names=F,col.names=F)

require(grDevices)

Wt = as.matrix(read.table('~/SNFtools/SNFtools/baseSC/test_text.txt'))
Wt_label = read.table('~/SNFtools/SNFtools/baseSC/test_text_label.txt')
Wt_label = t(Wt_label)

Wi = as.matrix(read.table('~/SNFtools/SNFtools/baseSC/test_image.txt'))
Wi_label = read.table('~/SNFtools/SNFtools/baseSC/test_image_label.txt')
Wi_label = t(Wi_label)

display_clusters <- function(W, group) {
  normalize <- function(X) X / rowSums(X)
  ind = sort(as.vector(group),index.return=TRUE)
  ind = ind$ix
  diag(W) = 0
  W = normalize(W);
  W = W + t(W);
  image(1:ncol(W),
        1:nrow(W),
        W[ind,ind],
        col = gray.colors(200, start=0.1, end=1, gamma = 15, alpha = NULL),
        xlab = 'Images',ylab='Images');
}

display_clusters(Wt, as.integer(Wt_label))

display_clusters(Wi, as.integer(Wi_label))

display_clusters(W_fused, res_fused_NMI[[1]])

#display_clusters(W_text[[text_gname_list[1]]], )
#display_clusters(W_img[[img_gname_list[3]]], )

####################################################
# comparison with base spectral methods (with affinity matrix at alpha=0.8)
baseSC = read.delim('/home/soyeon1771/SNFtools/SNFtools/baseSC/result_multibaseSC_gkernel_3789.txt', sep="\t", header=F, 
                    col.names=c('type','lambda','ARI','NMI'))

#max_SNF_NMI = max(data_scores$scores, na.rm=TRUE)
#single_txt_NMI = max(text_scores$scores, na.rm=TRUE)
#single_img_NMI = max(image_scores$scores, na.rm=TRUE)

lambdas = c(seq(0,1,0.1))
for (lambda in lambdas) {
  baseSC = rbind(baseSC, data.frame(type='SNF+SC', lambda=lambda, ARI=max_SNF_ARI, NMI=max_SNF_NMI))
  baseSC = rbind(baseSC, data.frame(type='single view(text)', lambda=lambda, ARI=single_text_ARI, NMI=single_text_NMI))
  baseSC = rbind(baseSC, data.frame(type='single view(img)', lambda=lambda, ARI=single_img_ARI, NMI=single_img_NMI))
}

# line plot w.r.t. lambda(hyper parameter of co-regularized SC)
# g1<-ggplot(data=baseSC[baseSC$lambda %in% seq(0,1,0.1),], aes(x=lambda, y=ARI, group=type, colour=type)) + 
#   geom_line(aes(linetype=type),size=1) + 
#   scale_color_manual(values=c(brewer.pal(6, "Dark2"))) +
#   theme_bw() +
#   geom_point(size=1.3) + 
#   xlim(0,1)+
#   ylim(0.6,0.85)
# 
# g2<-ggplot(data=baseSC[baseSC$lambda %in% seq(0,1,0.1),], aes(x=lambda, y=NMI, group=type, colour=type)) + 
#   geom_line(aes(linetype=type),size=1) + 
#   scale_color_manual(values=c(brewer.pal(6, "Dark2"))) +
#   theme_bw() +
#   geom_point(size=1.3) + 
#   xlim(0,1)+
#   ylim(0.7,0.95)
# 
# multiplot(g1, g2, layout=matrix(c(1,2), nrow=1))


maxdf = data.frame(type=c('single view(text)', 
                          'single view(img)', 
                          'mindisSC', 
                          'pairwiseSC', 
                          'centroidSC', 
                          'proposed method'), 
                   
                   ARI=c(single_text_ARI,
                         single_img_ARI,
                         max(baseSC[baseSC$type=='mindisSC',]$ARI),
                         baseSC[baseSC$NMI==max(baseSC[baseSC$type=='pairwiseSC',]$NMI),'ARI'],
                         baseSC[baseSC$NMI==max(baseSC[baseSC$type=='centroidSC',]$NMI),'ARI'],
                         max_SNF_ARI),
                   
                   NMI=c(single_text_NMI,
                         single_img_NMI,
                         max(baseSC[baseSC$type=='mindisSC',]$NMI),
                         max(baseSC[baseSC$type=='pairwiseSC',]$NMI),
                         max(baseSC[baseSC$type=='centroidSC',]$NMI),
                         max_SNF_NMI))

# bar plot at max ARI score
bar1 <- ggplot(data=maxdf, aes(x=reorder(type, ARI), y=ARI, fill=type)) + 
  theme_bw() +
  geom_bar(stat='identity') +
  xlab('Methods') +
  ylim(0,1) +
  geom_text(aes(label=round(ARI,3)), vjust=-0.2, size=6)+
  scale_fill_manual(values=c(brewer.pal(6, "Dark2"))) +
  theme(axis.text.x=element_text(angle=45, hjust=1, size=rel(1.5)),
        legend.position = "none",
        axis.title.x = element_text(size = rel(2)),
        axis.title.y = element_text(size = rel(2)))

# bar plot at max NMI score
bar2 <- ggplot(data=maxdf, aes(x=reorder(type, NMI), y=NMI, fill=type)) + 
  theme_bw() +
  geom_bar(stat='identity') +
  xlab('Methods') +
  ylim(0,1) +
  geom_text(aes(label=round(NMI,3)), vjust=-0.2, size=6)+
  scale_fill_manual(values=c(brewer.pal(6, "Dark2"))) +
  theme(axis.text.x=element_text(angle=45, hjust=1, size=rel(1.5)),
        legend.position = "none",
        axis.title.x = element_text(size = rel(2)),
        axis.title.y = element_text(size = rel(2)))

multiplot(bar1, bar2, layout=matrix(c(1,2), nrow=1))


baseSC[baseSC$NMI==max(baseSC[baseSC$type=='pairwiseSC',]$NMI),]
baseSC[baseSC$NMI==max(baseSC[baseSC$type=='centroidSC',]$NMI),]