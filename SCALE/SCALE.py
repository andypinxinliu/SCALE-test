import time
import torch

import numpy as np
import pandas as pd
import os
import scanpy as sc
import argparse

from scale import SCALE
from scale.dataset import load_dataset
from scale.utils import read_labels, cluster_report, estimate_k, binarization
from scale.plot import plot_embedding

from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction')
    
    # 可选参数，d用来找datalist，不写就是空的list
    parser.add_argument('--data_list', '-d', type=str, nargs='+', default=[])
    
    # 可选参数，k用来表示k-means的初始k，不写就是30
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=30)
    
    # 可选参数，o表示ouptput的路径
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    
    # 可选参数，如果没写就是false，用来确定是否print loss function
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    
    # 可选参数， --pretrain，是str
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    
    # 可选参数，不写就是 0.0002的learning rate
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    
    # 可选参数，b，表示batach size，不写就是32的大小
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    
    # 可选参数，用来确定师傅要使用gpu
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    
    # 可选参数，seed，不写就是18个
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    
    # 可选参数，encode_dim，不写的话就是[1024, 128]，
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[1024, 128], help='encoder structure')
    
    # 可选参数，decode_dim, 不写的话是空的list
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='encoder structure')
    
    #可选参数，l，不写的话就是10，表明的latent layer的dimesion
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    
    #最小的peak，不写的话就是100带入，peak指的是cell的row
    parser.add_argument('--min_peaks', type=float, default=100, help='Remove low quality cells with few peaks')
    
    #最小的表达，不写的话就是0.01
    parser.add_argument('--min_cells', type=float, default=0.01, help='Remove low quality peaks')
    
    #如果不使用log-transform的话，那就是直接使用原来的data，如果设置为true，就会进行差异表达处理
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    
    #最大iteration的次数
    parser.add_argument('--max_iter', '-i', type=int, default=30000, help='Max iteration')
    
    #
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    #确定是否保存impute data，不写就是false
    parser.add_argument('--impute', action='store_true', help='Save the imputed data in layer impute')
    
    #确定
    parser.add_argument('--binary', action='store_true', help='Save binary imputed data in layer binary')
    
    #
    parser.add_argument('--embed', type=str, default='UMAP')
    
    #
    parser.add_argument('--reference', type=str, default='celltype')
    
    #
    parser.add_argument('--cluster_method', type=str, default='leiden')


    args = parser.parse_args()

    # Set random seed
    #如果没选seed话，就是18
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    #如果没有GPU，就使用CPU
    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device='cpu'
        
    # 如果不写就是32
    batch_size = args.batch_size
    
    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
    print("**********************************************************************\n")


    #load_dataset的method在dataset的package里面
    adata, trainloader, testloader = load_dataset(
        args.data_list,
        batch_categories=None, 
        join='inner', 
        batch_key='batch', 
        batch_name='batch',
        min_genes=args.min_peaks,
        min_cells=args.min_cells,
        batch_size=args.batch_size, 
        log=None,
    )

    # 找到dimension
    cell_num = adata.shape[0] 
    input_dim = adata.shape[1] 	
    
#     if args.n_centroids is None:
#         k = estimate_k(adata.X.T)
#         print('Estimate k = {}'.format(k))
#     else:
#         k = args.n_centroids


    lr = args.lr
    
    #不写的话，k就是30
    k = args.n_centroids
    
    
    outdir = args.outdir+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    print("\n======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}'.format(
        cell_num, input_dim, k, args.max_iter, batch_size, args.min_peaks, args.min_cells))
    print("============================")


    #
    dims = [input_dim, args.latent, args.encode_dim, args.decode_dim]
    model = SCALE(dims, n_centroids=k)
    print(model)


    if not args.pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  weight_decay=args.weight_decay,
                  verbose=args.verbose,
                  device = device,
                  max_iter=args.max_iter,
#                   name=name,
                  outdir=outdir
                   )
        torch.save(model.state_dict(), os.path.join(outdir, 'model.pt')) # save model
    else:
        print('\n## Loading Model: {}\n'.format(args.pretrain))
        model.load_model(args.pretrain)
        model.to(device)
    
    ### output ###
    print('outdir: {}'.format(outdir))
    # 1. latent feature
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')

    # 2. cluster
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    if args.cluster_method == 'leiden':
        sc.tl.leiden(adata)
    elif args.cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)

#     if args.reference in adata.obs:
#         cluster_report(adata.obs[args.reference].cat.codes, adata.obs[args.cluster_method].astype(int))

    sc.settings.figdir = outdir
    sc.set_figure_params(dpi=80, figsize=(6,6), fontsize=10)
    if args.embed == 'UMAP':
        sc.tl.umap(adata, min_dist=0.1)
        color = [c for c in ['celltype', args.cluster_method] if c in adata.obs]
        sc.pl.umap(adata, color=color, save='.pdf', wspace=0.4, ncols=4)
    elif args.embed == 'tSNE':
        sc.tl.tsne(adata, use_rep='latent')
        color = [c for c in ['celltype', args.cluster_method] if c in adata.obs]
        sc.pl.tsne(adata, color=color, save='.pdf', wspace=0.4, ncols=4)
    
    if args.impute:
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
    if args.binary:
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
        adata.obsm['binary'] = binarization(adata.obsm['impute'], adata.X)
    
    adata.write(outdir+'adata.h5ad', compression='gzip')
