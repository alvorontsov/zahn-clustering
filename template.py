import numpy as np

class Template_5:
    
    def __init__(self,n1,n2,n3,n4):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.size = n1 + n2 + n3 + n4 + 1
        
    def __get_zvec_from_series(self,series,start):
        vec = []
        inds = []
        inds.append(start)
        vec.append(series[start])
        v = start + self.n1
        inds.append(v)
        vec.append(series[v])
        v = v + self.n2
        inds.append(v)
        vec.append(series[v])
        v = v + self.n3
        inds.append(v)
        vec.append(series[v])
        v = v + self.n4
        inds.append(v)
        vec.append(series[v])
        return np.array(vec), np.array(inds)
        
    def get_zvectors(self,series):
        vecs = []
        inds = []
        for i in range(len(series) - self.size):
            v,ind = self.__get_zvec_from_series(series,i)
            vecs.append(v)
            inds.append(ind)
        return np.array(vecs), np.array(inds)