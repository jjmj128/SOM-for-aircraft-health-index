# -*- coding: utf-8 -*-
"""
"""

""" Modules """
from tqdm import tqdm
import math
import numpy as np
from numpy import genfromtxt,apply_along_axis
import random as rd
import csv
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import time
import datetime
import sys
from PIL import Image,ImageFont,ImageDraw

from secrets import randbelow as rb

from I_Data_process import getFile, openFile, writeFile, timeFormat, dataFusion, dataMinMax, dataNorm, dataNormMode, pathGeneration

import multiprocessing

""" Settings """
rd.seed() #Initialize basic random number generator => same generation




"""
Functions
"""
def normVector(v):
    return np.linalg.norm(v)

def euclideanDistance(v1, v2):
    """Return Euclidean distance between numpy arrays v1 and v2."""
    return normVector(v2-v1)

#Implementation of the class "Node" (for 1 neuron):
class Node(object):
    def __init__(self, position, idim, iboundaries=(0, 1)):
        """Create a node for Lattice.
        Args:
            i:      i position of Node in Lattice
            j:      j position of Node in Lattice
            idim:   dimension of input space
            iboundaries:   boundaries of attribute values (for initialization)
        """
        if(len(position) == 2):
            self.nodeDim = 2
            self.i = position[0]
            self.j = position[1]
        elif(len(position) == 3):
            self.nodeDim = 3
            self.i = position[0]
            self.j = position[1]            
            self.k = position[2]
        else:
            print("Node position error: ", position)
            sys.exit()
        self.idim = idim
        self.weights = np.array([  # Random initialization of weights
            rd.uniform(*iboundaries)
            for i in range(idim)
        ])

    def getDistance(self, ivector):  #ivector = input vector
        """Return Euclidean distance between weights and ivector.
        Args:
            ivector:    input vector
        """
        if(len(ivector) == self.idim):
            return euclideanDistance(self.weights, ivector)
        else:
            print("Error getDistance len(ivector) != map size (dimension not equal)...")
            sys.exit()
        
    def position(self):
        """Return position of self in Lattice"""
        if(self.nodeDim == 2):
            out = [self.i, self.j]
        elif(self.nodeDim == 3):
            out = [self.i, self.j, self.k]
        return np.array(out)
        

    def update(self, ivector, learning_rate, influence_rate):
        """Updates weights."""    
        self.weights = (
            self.weights
            + learning_rate
            * influence_rate
            * (ivector - self.weights)
        )
        return self.weights

#Implementation of the class "Lattice" (= grid):    
class Lattice(object):
    """Implementation of a SOM."""
    def __init__(self, size, iterations_n, learning_rate, idim, iboundaries=(0, 1)):
        """Create an initialized lattice.
        Args:
            size:           size of map
            iterations_n:   number of iterations
            learning_rate:  learning rate
            idim:           dimension of inut space
            iboundaries:    boudaries of attribute values (for Node init)
        """
        # General features
        self.size = size
        self.nodes = []
        self.iterations_n = iterations_n
        self.modulo = 0.1 #Percentage for display
        self.learning_rate = learning_rate
        self.learning_rate_min = 0.01
        self.idim = idim
        #○ Radius r
        self.radius = float(max(self.size)) / 2. # Parameters for BMU neighbourhood radius calculation
        # Lambda
        if self.radius > 1:
            self.time_cst = iterations_n / math.log(self.radius)
        else:
            self.time_cst = iterations_n / math.log(1.5)
        # Initialize the grid
        if(len(size) == 2):
            self.dim = 2
            for i in range(size[0]):
                for j in range(size[1]):
                    self.nodes.append(Node((i,j), idim, iboundaries))
        elif(len(size) == 3):
            self.dim = 3
            for i in range(size[0]):
                for j in range(size[1]):
                    for k in range(size[2]):
                        self.nodes.append(Node((i,j,k), idim, iboundaries))
        else:
            print("SOM lattice size error: ",size)
            sys.exit()

    def decayFunc(self, iter_t):  # decay(t)
        return math.exp(float(-iter_t) / self.time_cst)

    def decayedRadius(self, iter_t):  # radius(t)
        """Return radius to use around bmu at iteration iter_t."""
        return self.radius * self.decayFunc(iter_t)

    def decayedLearningRate(self, iter_t):  # learning_rate(t)
        """Return learning rate (weights update) at iteration iter_t."""
        return max(self.learning_rate_min, self.learning_rate * self.decayFunc(iter_t))

    def influenceFunc(self, iter_t, dist):  # influence(t)
        """Return influence rate (weights update) at iteration iter_t.
        Args:
            dist:   distance beetween concerned node and bmu
        """
        return math.exp(- 0.5 * dist**2 / self.decayedRadius(iter_t)**2 )

    def getBMU(self, ivector):
        """Return closest Node from ivector, in term of weights."""
        bmu = None
        best_distance = None
        for node in self.nodes:
            distance = node.getDistance(ivector) #SS: utilise la norm au carrée !
            if bmu is None:
                bmu = node               
                best_distance = distance                
            elif best_distance > distance:
                bmu = node
                best_distance = distance               
        return bmu
    
    def getBMUMulti(self, ivector):
        """Return closest Node from ivector, in term of weights."""
        bmu = None
        best_distance = None
        import pathos.multiprocessing as mp
        p = mp.Pool(mp.cpu_count() -1)
        distance = []
        def nodeDist(node):
            return node.getDistance(ivector)
        def res(out):
            distance.append(out)
        
        print("\nMulti begin")
        start=time.time()
        r = []
        for node in self.nodes:
            r = p.apply_async(nodeDist, args=(node, ivector), callback=res)
            r.wait()
        
        #r = p.map_async(nodeDist, self.nodes, callback=res)
        #r.wait()
        print(r)
        end=time.time()
        print("Multi:",end-start)
        print(distance)
        
        
        print("\nSingle begin")
        distance = []
        start=time.time()
        for node in self.nodes:
            distance.append(node.getDistance(ivector))
        end=time.time()
        print("Single:",end-start)
        print(distance)
        p.close()
        p.join()
        for node in self.nodes:
            distance = node.getDistance(ivector)
            if bmu is None:
                bmu = node               
                best_distance = distance                
            elif best_distance > distance:
                bmu = node
                best_distance = distance               
        return bmu    
     
    def getNeighbourhood(self, bmu, radius):
        """Return list of Nodes within the radius from bmu.
        From the Grid perspective.
        """
        neighbourhood = []
        neighbourhood_dist = []
        for node in self.nodes:
            dist = euclideanDistance(node.position(), bmu.position())
            if(dist <= radius): 
                neighbourhood.append(node)
                neighbourhood_dist.append(dist)
                
        return neighbourhood, neighbourhood_dist
    
    def grid3d(self):
        """Return matrix of nodes. For display in colors for inputs of 3 or 4 dimensions only.
        """
        #the last number must be equal to the idim number 
        matrix = np.ndarray(shape=(self.size[0], self.size[1], self.size[2]))
        for node in self.nodes:
            matrix[node.i][node.j][node.k] = normVector(node.weights)
        return matrix
   
  
    def train(self, ivectors):
        """Train SOM on ivectors."""
        print("SOM training...")
        idx = []
        end = len(ivectors)-1
        """
        import pathos.multiprocessing as mp
        #p = mp.Pool(mp.cpu_count() -1)
        vect = [ivectors[rd.randint(0,end)] for iter_t in range(self.iterations_n)]
        idx = [iter_t for iter_t in range(self.iterations_n)]
        with multiprocessing.Pool(mp.cpu_count() - 1) as pool:
            r = pool.starmap(self.trainStep, zip(vect,idx))

        """
        for iter_t in tqdm(range(self.iterations_n)):
            # Randomly pick a input vector for training
            index = rd.randint(0,end)
            idx.append(index)
            self.trainStep(ivectors[index], iter_t)
        
        return idx

    def trainStep(self, ivector, iter_t):
        """Train on ivector for the given iteration number."""
        #if(iter_t % (self.iterations_n * self.modulo) == 0):
            #print("Iteration: {}".format(iter_t))
        bmu = self.getBMU(ivector)
        radius = self.decayedRadius(iter_t)
        learning_rate = self.decayedLearningRate(iter_t)
        neighbourhood, neighbourhood_dist = self.getNeighbourhood(bmu, radius)

        for i in range(len(neighbourhood)):
            dist = neighbourhood_dist[i]
            neighbour = neighbourhood[i]
            influence_rate = self.influenceFunc(iter_t, dist)
            neighbour.update(ivector, learning_rate, influence_rate)      

    def load(self, data):
        for i in range(len(data)):
            self.nodes[i].weights = data[i]

    def exportation(self, file_path):
        file_instance, csv_file = writeFile(file_path)
        stockage = []
        for node in self.nodes:
            my_list = []
            my_list.append(node.i)
            my_list.append(node.j)
            if(node.nodeDim == 3):
                my_list.append(node.k)
            for weight in node.weights:
                my_list.append(weight)
            stockage.append(my_list)
        csv_file.writerows(stockage)
        file_instance.close()
        return stockage
    
"""    def plotx(self):
        data = self.grid3d()"""



def trainSOM(input_file_path, SOM_file_path, SOM_type = 2, idim_cor = 2):
    start_time=time.time()
    print("Training launching...")
    data = getFile(input_file_path)
    data = np.delete(data,[i for i in range(idim_cor)],1)
    SOM = Lattice(
        size = tuple([SOM_size]*SOM_type),  #size of map
        iterations_n = nbIter,  #number of iterations
        learning_rate = learning_rate,  #learning rate
        idim = int(len(data[0]))) #inputs dimension
    idx = SOM.train(data)
    SOM.exportation(SOM_file_path)
    idxNb = len(idx)
    idx2 = list(set(idx))
    idx2Nb = len(idx2)
    print("Value used multiple time: ",idxNb-idx2Nb)
    
    
    print("Training Done!")
    print("Iterations: {}" .format(nbIter))
    end_time = time.time()
    
    h_time, min_time, s_time = timeFormat(start_time, end_time)
    
    print("Training time: {}h {}min {}s".format(h_time,min_time,s_time))

    return SOM



def loadSOM(file_path, SOM_size, SOM_type = 2, idim_cor = 2, denug=False):
    data = getFile(SOMTrained_path)
    data = np.delete(data,[i for i in range(idim_cor)],1) #Delte node i,j position
    SOM = Lattice(
        size = tuple([SOM_size]*SOM_type),  #size of map
        iterations_n = nbIter,  #number of iterations
        learning_rate = learning_rate,  #learning rate
        idim = int(len(data[0]))) #inputs dimension
    SOM.load(data)
    return SOM



def exitedNode(SOM, train_path,nodeExited_path,norm_info_path='', norm=False, debug=False):
    """  """
    start_time = time.time()

    if(norm):
        data_instance, data = openFile(train_path.replace('Norm',''))
        _, norm_info = openFile(norm_info_path.replace('.csv','Info.csv'))
        norm_type = ''
        for item in norm_info:
            if(norm_type !=''):
                data_info.append(item)
            if(item[0]=='s'): #stand norm
                norm_type='stand'
                data_info = []
            if(item[0]=='r'):
                norm_type='rescaling'
                data_info = []                
        data_info = np.array(data_info).astype(float)
    else:
        data_instance, data = openFile(train_path)
    nodeExited = []
    for row in tqdm(data):
        if(norm): #Norm
            row_tmp, _ = dataNorm(norm_type, np.column_stack((row,row)).transpose().astype(float), data_info, 2, len(row)-1)
            row=row_tmp[0,:]
        nodeExited.append(SOM.getBMU(np.array(list(map(float,row[2:])))))
    data_instance.close()

    end_time = time.time()
    h_time, min_time, s_time = timeFormat(start_time, end_time)
    print("Cluster detection in {} h {} min {} s".format(h_time,min_time,s_time))

    file_instance, file_node = writeFile(nodeExited_path)
    for item in nodeExited:
        file_node.writerows([list(item.position())])
    file_instance.close()

    return nodeExited



def clusterMaps(nodeExited_path,SOM_size,debug=False):
    """ MAP of exited node """
    nodeExited = getFile(nodeExited_path)
    nodeMAP = np.zeros(shape=(SOM_size,SOM_size))
    for item in nodeExited:
        nodeMAP[int(item[0]),int(item[1])] = 1.
    
    """ Cluster boundaries """
    clusterMAP = []
    k = 0
    while(nodeMAP.max() == 1.):
        k = k + 1
        if(debug):
            print("nodeMAP\n",nodeMAP)
        clusterTmp = np.zeros(shape=(SOM_size,SOM_size))
        nodeFirst = []
        for i in range(SOM_size):
            flag0 = False
            for j in range(SOM_size):          
                if(nodeMAP[i,j] != 0.):
                    clusterTmp[i,j] = 1.
                    nodeFirst.append([i,j])
                    i_tmp = i
                    j_tmp = j
                    flag0 = True
                    break
            if(flag0):
                break
        while(flag0):
            nodeTmp = []
            for item in nodeFirst:
                i_tmp = item[0]
                j_tmp = item[1]
                if(j_tmp<SOM_size-1):
                    if(nodeMAP[item[0],item[1]+1] == 1. and clusterTmp[item[0],item[1]+1] != 1.):
                        nodeTmp.append([item[0],item[1]+1])
                        clusterTmp[item[0],item[1]+1] = 1.
                if(i_tmp<SOM_size-1):
                    if(nodeMAP[item[0]+1,item[1]] == 1. and clusterTmp[item[0]+1,item[1]] != 1.):
                        nodeTmp.append([item[0]+1,item[1]])
                        clusterTmp[item[0]+1,item[1]] = 1.
                if(j_tmp>0):
                    if(nodeMAP[item[0],item[1]-1] == 1. and clusterTmp[item[0],item[1]-1] != 1.):
                        nodeTmp.append([item[0],item[1]-1])
                        clusterTmp[item[0],item[1]-1] = 1.
                if(i_tmp>0):
                    if(nodeMAP[item[0]-1,item[1]] == 1. and clusterTmp[item[0]-1,item[1]] != 1.):
                        nodeTmp.append([item[0]-1,item[1]])
                        clusterTmp[item[0]-1,item[1]] = 1.
            nodeFirst = nodeTmp
            if(len(nodeFirst) == 0):
                flag0 = False
    
        clusterMAP.append(clusterTmp)
        nodeMAP -= clusterTmp
        if(debug):
            print("clusterTmp {}\n".format(k),clusterTmp)
    if(debug):
        print("nodeMAP\n",nodeMAP)
    
    clusterNb = len(clusterMAP)
    if(debug):
        print("{} cluster identified".format(clusterNb))

    for i in range(clusterNb):
        nodeMAP += clusterMAP[i]*(i+1)
    
    return clusterNb, nodeMAP



def inputLabel(train_path, nodeExited_path, nodeExited_path_Test, SOMTrained_path, SOM, label_path, debug=False):
    _, clusterMAP = clusterMaps(nodeExited_path, SOM.size[0])

    instance_SOM, data_SOM = openFile(SOMTrained_path)
    data_SOM_tab = np.zeros(shape=(SOM.size[0],SOM.size[1],SOM.idim))
    for item in data_SOM:
        data_SOM_tab[int(item[0]),int(item[1])] = item[2:]
    instance_SOM.close()
    
    instance_input, data_input = openFile(train_path)
    data_nodeExited = getFile(nodeExited_path_Test)
    
    instance_label, data_label = writeFile(label_path)
    i = -1
    for item in data_input:
        i = i + 1
        label = []
        label.append(list(map(float,item[0:2])))
        i_tmp = int(data_nodeExited[i][0])
        j_tmp = int(data_nodeExited[i][1])
        label.append([clusterMAP[i_tmp,j_tmp]])
        #label.append(data_SOM_tab[i_tmp,j_tmp])
        label.append(item[2:])
        labeled = [it for sub in label for it in sub]
        data_label.writerows([labeled])
    instance_label.close()
    instance_input.close()



def inputLabel3D(train_path, nodeExited_path, SOMTrained_path, SOM, label_path, debug=False):
    _, clusterMAP = clusterMaps(nodeExited_path, SOM.size[0])

    instance_SOM, data_SOM = openFile(SOMTrained_path)
    data_SOM_tab = np.zeros(shape=(SOM.size[0],SOM.size[1],SOM.size[2],SOM.idim))
    for item in data_SOM:
        data_SOM_tab[int(item[0]),int(item[1]),int(item[2])] = item[3:]
    instance_SOM.close()
    
    instance_input, data_input = openFile(train_path)
    data_nodeExited = getFile(nodeExited_path)
    
    instance_label, data_label = writeFile(label_path)
    i = -1
    for item in data_input:
        i = i + 1
        label = []
        label.append(list(map(float,item[0:2])))
        i_tmp = int(data_nodeExited[i][0])
        j_tmp = int(data_nodeExited[i][1])
        k_tmp = int(data_nodeExited[i][2])
        label.append([clusterMAP[i_tmp,j_tmp,k_tmp]])
        label.append(data_SOM_tab[i_tmp,j_tmp,k_tmp])
        labeled = [item for sub in label for item in sub]
        data_label.writerows([labeled])
    instance_label.close()
    instance_input.close()



def clusterLabel(nodeExited_path, SOMLabel_path, SOM_size):
    _, clusterMAP = clusterMaps(nodeExited_path, SOM_size)
    instance_SOM, data_SOM = openFile(SOMTrained_path)
    instance_SOM_Label, data_SOM_Label = writeFile(SOMLabel_path)
    for item in data_SOM:
        label = []
        i_tmp = int(item[0])
        j_tmp = int(item[1])
        label.append(item[:2])
        label.append([clusterMAP[i_tmp,j_tmp]])
        label.append(item[2:])
        labeled = [item for sub in label for item in sub]
        data_SOM_Label.writerows([labeled])
    instance_SOM.close()
    instance_SOM_Label.close()



def clusterNom(clusterNorm_path, norm_type, SOMLabel_path, SOM):
    data_cluster = getFile(SOMLabel_path)
    nbCluster = int(max(data_cluster[:,2]))
    cluster = []
    clusterMAPDeg = np.ones(shape=SOM.size)*(-1)
    for cl in range(1,nbCluster+1):
        cluster_i = []
        cluster_i_pos = []
        for item in data_cluster:
            if(int(item[2]) == cl):
                norm = normVector(item[3:])
                cluster_i_pos.append([item[0],item[1],item[2]]) #[x,y,OPM]
                cluster_i.append(norm)
        """ Get boundaries """
        #data_min, data_max = dataMinMax(np.array([cluster_i]).transpose())
        #data_bound = np.array(list((data_min,data_max))) # matrix [min;max]
        """ Normalize data """
        cluster_i, _ = dataNorm(norm_type, np.array([cluster_i]).transpose())
        cluster.append(np.concatenate((np.array(cluster_i_pos),cluster_i),axis=1)) #[x,y,norm]

        """ Degradation Map representation """
        for item in cluster[-1]:
            i_tmp = int(item[0])
            j_tmp = int(item[1])
            clusterMAPDeg[i_tmp, j_tmp] = item[-1]

    instance_clusterNorm, data_clusterNorm = writeFile(clusterNorm_path)
    _, clusterMAP = clusterMaps(nodeExited_path, SOM_size)
    for i in range(SOM.size[0]):
        for j in range(SOM.size[1]):
            row = [float(i),float(j),clusterMAP[i,j],clusterMAPDeg[i,j]]
            data_clusterNorm.writerows([row])
    instance_clusterNorm.close()
    
    return cluster



def clusterPlot(cluster, show=True, save=False, save_path='.', ext='jpg', title=None, debug=False):
    color_full = [] #https://www.w3schools.com/colors/colors_picker.asp
    color_sat = 125
    color_sat2 = 230
    #Red
    color_start=[color_sat,0,0]
    color_end=[255,color_sat2,color_sat2]
    color_full.append([color_start,color_end])
    #Blue
    color_start=[0,0,color_sat]
    color_end=[color_sat2,color_sat2,255]
    color_full.append([color_start,color_end])
    #Green
    color_start=[0,color_sat,0]
    color_end=[color_sat2,255,color_sat2]
    color_full.append([color_start,color_end])
    #Purpule
    color_start=[128,0,255]
    color_end=[230,color_sat2,255]
    color_full.append([color_start,color_end])
    #Turcoise
    color_start=[0,color_sat,color_sat]
    color_end=[color_sat2,255,255]
    color_full.append([color_start,color_end])
    #Pink
    color_start=[color_sat,0,color_sat]
    color_end=[255,color_sat2,255]
    color_full.append([color_start,color_end])
    for i in range(6*6):
        color_full.append(color_full[i])

    if(len(cluster) > len(color_full)):
        print("Not enough color for a proper plot. Modify the program")
        sys.exit()

    img = Image.new( 'RGB', SOM.size) # create a new black image
    pixels = img.load() # create the pixel map

    m=-1
    for cl in cluster:
        m = m + 1
        #nbVal = len(cl)
        color_start = color_full[m][0]
        color_end = color_full[m][1]
        for item in cl:
            i_tmp = int(item[0])
            j_tmp = int(item[1])
            color_i = [int(color_j-(color_j-color_i)*item[-1]) for color_i, color_j in zip(color_start, color_end)]
            pixels[i_tmp,j_tmp] = tuple(color_i)

    #xint = range(0,SOM.size[0])
    #yint = range(0,SOM.size[0])
    #plot.xticks(xint)
    #plot.yticks(yint)
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    if(title is not None):
        ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plot.imshow(img, origin='lower')
    plot.tight_layout()
    if(save):
        print("Plot saved in ",save_path+'.'+ext)
        if(ext == 'eps'):
            plot.savefig(save_path+'.'+ext, format=ext, bbox_inches='tight', dpi=1200)
        else:
            plot.savefig(save_path+'.'+ext, format=ext, bbox_inches='tight')
    if(show):
        plot.show()
        #img.save(save_path+".jpg")  



def datasetLoading(datasetType, datasetNumber, datasetOP=False, debug=False):
    if(datasetNumber == 1):
        train_dir = 'CMAPSSData'
        train_file2 = '_FD001Csv'
        SOM_size = 19
        if(datasetOP):
            train_file2 = train_file2+'OP'
            SOM_size = 7
    elif(datasetNumber == 2):
        train_dir = 'CMAPSSData'
        train_file2 = '_FD002Csv'
        SOM_size = 22
        if(datasetOP):
            train_file2 = train_file2+'OP'
            SOM_size = 8
    elif(datasetNumber == 3):
        train_dir = 'CMAPSSData'
        train_file2 = '_FD003Csv'
        SOM_size = 20
        if(datasetOP):
            train_file2 = train_file2+'OP'
            SOM_size = 7
    elif(datasetNumber == 4):
        train_dir = 'CMAPSSData'
        train_file2 = '_FD004Csv'
        SOM_size = 25
        #SOM_size = 35
        if(datasetOP):
            train_file2 = train_file2+'OP'
            SOM_size = 9
    elif(datasetNumber == 5):
        train_dir = 'input'
        train_file2 = 'Csv'
        SOM_size = 23
        if(datasetOP):
            train_file2 = train_file2+'OP'
            SOM_size = 8
    elif(datasetNumber == 6):
        train_dir = 'CMAPSSData'
        train_file2 = '_FD00XCsv'
        SOM_size = 42
        SOM_size = 30
        if(datasetOP):
            train_file2 = train_file2+'OP'
            SOM_size = 8    
    else:
        print("Dataset "+str(datasetNumber)+" doesn't exist !")
        sys.exit()
    return train_dir, datasetType+train_file2, SOM_size



if __name__ ==  '__main__':
    """
    Main program
    """
    sys.stdout.flush() #to force the print to appear on the console, only needed in spyder3 for linux sofar.
    global SOM_size, SOM_size_Test
    datasetNumber = 0 #The number of the dataset to use
    datasetType = 'train'
    datasetNumberTest = datasetNumber #The number of the dataset to test on the datasetNumber
    datasetTypeTest = 'train'
    
    datasetOP = False #True or False : to use the dataset for detection of operational mode with operational settings, etc.
    
    SOMtoTrain = True #True to train the SOM
    trainCombinedFileBool = False #True to use the combined training file for hybrid
    SOMtoExited = True #True to get exited node on a SOM map
    SOMtoExitedTest = False #True to get exited node of datasetNumberTest on the datasetNumber map
    SOMinputLabel = True #True to label input data
    SOMmodeLabel = True #True to label mode

    SOMClusterLabel = True #True to label cluster
    SOMClusterLabelNorm = True #True to norm the cluter HI
    SOMtoPlotMap = True #True to plot the final Map
    SOMtoPlotMapShow = True #True to show the Map
    
    SOM_size_tmp = 15 #FD001: 19 | FD002: 22 | FD005: 23 #Number of node in x and y. Only for square map
    SOM_type = 2 #Number of dimension : 2 or 3 possible. No control on this input
    learning_rate = 0.9 #Learning rate of SOM algorithm
    
    format_out = 'pdf' # jpg | pdf  #Output format to save graphic plot
    
    
    SOM_size = None
    if(datasetNumber==0): #Dataset definition
        train_dir = 'train' #Directory of the training data
        train_file2 = 'traincruise' #
        plot_title = 'Dataset' #Title of the graphic plot
    else:
        train_dir, train_file2, SOM_size = datasetLoading(datasetType, datasetNumber, datasetOP) #Load dataset
    
    if(datasetNumberTest==0):
        train_dir_Test = train_dir
        train_file2_Test = train_file2
    else:
        train_dir_Test, train_file2_Test, SOM_size_Test = datasetLoading(datasetTypeTest, datasetNumberTest, datasetOP) #Load dataset
    
    if(datasetNumber == datasetNumberTest):
        norm_option = False
    else:
        norm_option = True
    
    if(SOMtoTrain):
        SOM_size = SOM_size_tmp
        SOM_size_Test = SOM_size_tmp
    else:
        if(SOM_size is None):
            SOM_size = SOM_size_tmp

    nbIter = SOM_size**SOM_type * 500 #Number of iteration to train the SOM
    if(datasetNumber>0): #Set a graphic title if dataset is used
        plot_title = 'Dataset #'+str(datasetNumber)
    
    train_file = train_file2+'Feature' #Feature file
    train_file_Norm = train_file+'Norm' #Feature normilized file
    train_file_Test = train_file2_Test+'Feature' #Feature file
    train_file_Norm_Test = train_file_Test+'Norm' #Feature normilized file    
    
    SOM_file_dir = '.' #Path to save the SOM
    SOM_file_dir_Test = SOM_file_dir #Path to save the SOM
    
    SOM_file = train_file_Norm+'SOMTrainedIter'+str(nbIter)+'('+str(SOM_size) #File name of the SOM
    for i in range(SOM_type-1): #File name of the SOM for multiple dimension
        SOM_file += ','+str(SOM_size)
    SOM_file += ')'
    
    SOM_file_Test = train_file_Norm_Test+'SOMTrainedIter'+str(nbIter)+'('+str(SOM_size) #File name of the SOM
    for i in range(SOM_type-1): #File name of the SOM for multiple dimension
        SOM_file_Test += ','+str(SOM_size)
    SOM_file_Test += ')'    

    
    culsterNorm_type = 'rescaling' #Norm type
    
    SOMTrained_path = '' #Do not touch this
    
    nodeExited_dir = '.'
    
    label_dir = '.'
    
    SOMLabel_dir = '.'
    
    clusterNorm_dir = '.'
    
    
    
    if(nbIter is None):
        nbIter = int(input('Number of iterations:'))
    
    
    """ Train SOM """
    train_path = pathGeneration(train_dir, train_file)
    train_path_Norm = pathGeneration(train_dir, train_file_Norm)
    train_path_Test = pathGeneration(train_dir_Test, train_file_Test)
    train_path_Norm_Test = pathGeneration(train_dir_Test, train_file_Norm_Test)    
    if(SOMTrained_path == ''):
        SOMTrained_path = pathGeneration(SOM_file_dir, SOM_file)
    if(SOMtoTrain):
        SOM = trainSOM(train_path_Norm, SOMTrained_path,SOM_type)
    if(trainCombinedFileBool):
        trainCombined_path_Norm = pathGeneration(train_dir, train_file_Norm+'Comb')
    else:
        trainCombined_path_Norm = train_path_Norm
    
    #train_path2 = pathGeneration(train_dir, train_file2)
    #file_instance, file_data = openFile(train_path)
    #file_instance2, file_data2 = writeFile(train_path2)
    #for row in file_data:
        #file_data2.writerows([[row[0],row[1],row[2],row[3],row[4],1-float(row[5]),row[6],1-float(row[7]),row[8]]])
    #file_instance.close()
    #file_instance2.close()
    
    
    """ Load SOM """
    SOM = loadSOM(SOMTrained_path, SOM_size, SOM_type, SOM_type)
    
    """ Node Exited """
    nodeExited_file = SOM_file+'nodeExited'#+'Iter'+str(nbIter)+str(SOM.size)
    nodeExited_path = pathGeneration(nodeExited_dir, nodeExited_file)
    nodeExited_file_Test = SOM_file_Test+'nodeExited'#+'Iter'+str(nbIter)+str(SOM.size)
    nodeExited_path_Test = pathGeneration(nodeExited_dir, nodeExited_file_Test)    
    if(SOMtoExited):
        nodeExited = exitedNode(SOM, trainCombined_path_Norm, nodeExited_path)
    if(SOMtoExitedTest):
        nodeExited_Test = exitedNode(SOM, train_path_Norm_Test, nodeExited_path_Test, train_path_Norm, norm=norm_option)
    #nodeExitedSOM = loadSOM(nodeExited_path, SOM_size, SOM_type, SOM_type) #TBC
    
    if(SOM_type == 3):
        print("End because of 3D SOM")
        sys.exit()
    
    """ Cluster Label """
    SOMLabel_file = SOM_file+'SOMLabeled'#+'Iter'+str(nbIter)+str(SOM.size)
    SOMLabel_path = pathGeneration(SOMLabel_dir, SOMLabel_file)
    if(SOMClusterLabel):
        clusterLabel(nodeExited_path, SOMLabel_path, SOM_size)
    
    """ Cluster Norm """
    clusterNorm_file = SOMLabel_file+'Norm'
    clusterNorm_path = pathGeneration(clusterNorm_dir, clusterNorm_file)
    if(SOMClusterLabelNorm):
        cluster = clusterNom(clusterNorm_path, culsterNorm_type, SOMLabel_path, SOM)
        """ Cluster Plot """
        if(SOMtoPlotMap):
            clusterPlot(cluster,show=SOMtoPlotMapShow, save=True, save_path='./plot/'+clusterNorm_file, ext=format_out,title=plot_title)


    """ Input label """
    label_file_Test = SOM_file_Test+'inputLabeled'#+'Iter'+str(nbIter)+str(SOM.size)
    label_path_Test = pathGeneration(label_dir, label_file_Test)
    if(SOMinputLabel):
        inputLabel(train_path_Test, nodeExited_path, nodeExited_path_Test, SOMTrained_path, SOM, label_path_Test)
    
    """ Input label """
    label_file = train_file2+'Mode'#+'Iter'+str(nbIter)+str(SOM.size)
    label_path = pathGeneration(label_dir, label_file)
    train_path2 = pathGeneration(train_dir, train_file2)
    if(SOMmodeLabel):
        inputLabel(train_path2, nodeExited_path, nodeExited_path_Test, SOMTrained_path, SOM, label_path)



    """ Cluster Extraction """
    """Flight extraction"""
    """
    selected_cluster = [1, 2, 3, 4, 5, 6]
    output_file_flightExtraction = label_file
    for i in selected_cluster:
        output_file_clusterExtraction_tmp = output_file_flightExtraction+str(i)
        #Generate  path of cluster extracted
        output_path_clusterExtraction = pathGeneration(label_dir, output_file_clusterExtraction_tmp)
        #Cluster Extraction
        rowExtraction(label_path, output_path_clusterExtraction, selected_row=i, selected_col=2)
    """

    """ Next """
    
    
    
    
    
    print("End")