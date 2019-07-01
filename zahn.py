import numpy as np 
import collections
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from scipy.sparse import dok_matrix
from scipy.sparse import csgraph
import time
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.sparse as sps
from template import Template_5
import sys
sys.setrecursionlimit(1000000)

class Zahn:
    
    def __init__(self, template, num_nearest = 30, alpha = 0.1336766, min_cluster_size=5):
        '''
        Template - an object of Template_5
        '''
        self.num_nearest = num_nearest
        self.alpha = alpha
        self.__ktree = None
        self.min_cluster_size = min_cluster_size
        self.clusters = None
        self.template = template
        self.centers = None
        self.comp_fifth_avg = None
        self.comp_four = None
        self.centers_ktree = None
    
    def __apply_avg(self,element,avg):
        '''
        Helper function to modify it if it is less then alpha*avg
        :param avg - average edge length in graph
        :param element - item to modify
        :return modified item
        '''
        if element < self.alpha*avg:
            return 0
        else:
            return element
            
    def __getstate__(self):
        """
        This function defines the fields of the class to be pickled.
        :return: All class fields except for "pool". It cannot be serialized.
        """
        self_dict = self.__dict__.copy()
        return self_dict
    
    
    def build_graph(self, vecs, size):
        ''' 
        Build pseudo-full graph using mun_nearest heuristic and KDTree
        :param vecs - z-vectors
        :size - train size (number of vectors)
        :return pseudo fully-connected graph
        '''
        self.__ktree = KDTree(vecs)
        X = dok_matrix((size,size),dtype=np.float32)
        edge_sum = 0
        edge_num = 0
        for i in range(size):
            nearest_dist, nearest_ind = self.__ktree.query(vecs[i], k=self.num_nearest) 
            for j in range(len(nearest_dist)):
                edge = nearest_dist[j]
                X[i,nearest_ind[j]] = nearest_dist[j]
                X[nearest_ind[j],i] = nearest_dist[j]
                edge_sum += edge
                edge_num += 1
            if i % 5000 == 0:
                print(i)
                print(edge_num)
        return X, edge_sum, edge_num
    
    def get_averages(self):
        comp_fifth_avg = []
        comp_four = []
        for component in self.clusters:
            comp_fifth_avg.append(np.mean([i[-1] for i in component]))
        comp = []
        for vec in component:
            comp.append(vec[:-1])
        comp_four.append(np.array(comp))
        centers = np.array([np.mean(x,axis=0) for x in comp_four])
        self.centers = centers
        self.comp_fifth_avg = comp_fifth_avg
        self.comp_four = comp_four
        self.centers_ktree = cKDTree(data=self.centers)
        return comp_fifth_avg, comp_four, centers
    
    def predict_point(self,zvec):
        '''
        Predicts the next point of cutted z-vector (4 coordinates)
        '''
        ind = self.centers_ktree.query(zvec)[1]
        pred = self.comp_fifth_avg[ind]
        return pred
        
    def fit_clusters(self, vecs):
        '''
        Form clusters given z-vectors
        :param vecs - z-vectors
        :return a list of clusters (lists of vectors)
        '''
        size = len(vecs)
        print('---CREATING GRAPH---')
        X, edge_sum, edge_num = self.build_graph(vecs,size)
        print('---CALCULATING MST---')
        Tcsr = minimum_spanning_tree(X)
        tcopy = Tcsr.tocsr()
        print('---APPLYING AVERAGE HEURISTIC---')
        avg = edge_sum / edge_num
        l = lambda x: self.__apply_avg(x,avg)
        vfunc = np.vectorize(l)
        tcopy.data = vfunc(tcopy.data)
        tcopy_dok = dok_matrix(tcopy)
        keys = list(tcopy_dok.keys())
        vals = list(tcopy_dok.values())
        print('---CREATING CONNECTED COMPONENTS---')
        X = dok_matrix((size,size),dtype=np.float32)
        for i in range(len(keys)):
            if vals[i] > 0:
                X[keys[i][0],keys[i][1]] = vals[i]
                X[keys[i][1],keys[i][0]] = vals[i]
        n_components, labels = csgraph.connected_components(X.tocsr())
        print(n_components)
        components = [[] for i in range(n_components)]
        for i in range(size):
            components[labels[i]].append(vecs[labels[i]])
        components = [x for x in components if len(x) > self.min_cluster_size]
        print(len(components))
        self.clusters = components
        return components
    