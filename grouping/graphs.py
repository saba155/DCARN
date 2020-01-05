import itertools
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.datasets.samples_generator import make_blobs
#------------------------------------------------------------------------------
#==============================================================================
# #surface normal methods
#==============================================================================
def surface_normal_newell(poly):

    n = np.array([0.0, 0.0, 0.0])

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly),:]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2]) 
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised


def surface_normal_cross(poly):

    n = np.cross(poly[1,:]-poly[0,:],poly[2,:]-poly[0,:])

    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised
#==============================================================================
# #Mean Shift
#==============================================================================
    
def mean_shift(xyz):
    
    num_ver = xyz.shape[0]#gives dimension of array
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    

    
    print("dimension of array", xyz.shape)
    print("Input array:", xyz)
    
#==============================================================================
#     Newel mean shift
#==============================================================================
    surface_normal_vectors_newell= surface_normal_newell(xyz)
    print("Surface Normals Newell:", surface_normal_vectors_newell)
    ms=MeanShift(bandwidth=100, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=False, n_jobs=5).fit(surface_normal_vectors_newell)

    clusters = ms.fit_predict(xyz) #------
    labels = ms.labels_
    #print("labels: ", labels)
    cluster_centers = ms.cluster_centers_ #-----
    print(cluster_centers.astype('uint32'))
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)
    print("size of clusters:", labels.shape)
    
    
#==============================================================================
#   Cross Mean shift  
#==============================================================================
    surface_normal_vectors_cross = surface_normal_cross(xyz)
    print("Surface Normals Cross:", surface_normal_vectors_cross)
    ms=MeanShift(bandwidth=100, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=False, n_jobs=8).fit(surface_normal_vectors_cross)

    clusters = ms.fit_predict(xyz) #------
    labels = ms.labels_
    #print("labels: ", labels)
    cluster_centers = ms.cluster_centers_ #-----
    print(cluster_centers.astype('uint32'))
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)
    print("size of clusters:", labels.shape)
    exit()  
    
    ''''
    #save the graph
    graph["source"] = np.matlib.repmat(range(0, num_ver)
                    , k_nn1, 1).flatten(order='F').astype('uint32')#repmat--repeat loop for matric mxn times... flatten in column major
    graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')#flatten in row major
    graph["distances"] = distances.flatten().astype('float32')#by default flatten in row major
    print("target 2nd argument: ", target2)
    print("graph output: ", graph)
    exit()
    return graph, target2
    '''
#------------------------------------------------------------------------------
def compute_graph_nn(xyz, k_nn):
    """compute the knn graph"""
    num_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = np.matlib.repmat(range(0, num_ver), k_nn, 1).flatten(order='F')
    #save the graph
    graph["source"] = source.flatten().astype('uint32')
    graph["target"] = neighbors.flatten().astype('uint32')
    graph["distances"] = distances.flatten().astype('float32')
    return graph
#------------------------------------------------------------------------------

def compute_graph_nn_2(xyz, k_nn1, k_nn2):
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    num_ver = xyz.shape[0]#gives dimension of array
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
   
    del nn
    distances = distances[:, 1:k_nn1 + 1]
    
#==============================================================================
#     Newel mean shift
#==============================================================================  
    surface_normal_vectors_newell = []

    distances_output = []
    target2_output = []
    for i_neighbour in range(len(neighbors)):

        nn_neighbours_xyz = []
        
        for ith_neighbour in range(len(neighbors[i_neighbour])):
            xyz_index = neighbors[i_neighbour][ith_neighbour]

            nn_neighbours_xyz.append(xyz[xyz_index])

        surface_normal_vector = surface_normal_newell(np.array(nn_neighbours_xyz))

        surface_normal_vectors_newell.append(surface_normal_vector)

    #bandwidth = estimate_bandwidth(surface_normal_vectors_newell, quantile=0.2, n_samples=None, random_state=0, n_jobs=8)
    #print("estimated band width", bandwidth)
    bandwidth = 5 #for s3dis dataset
#    bandwidth = 0.83 #for semantic3d dataset
    
    ms=MeanShift(bandwidth=bandwidth, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=False, n_jobs=8).fit(surface_normal_vectors_newell)


    labels = ms.labels_

    n_clusters_ = np.unique(labels)
   
     
    for cluster_Label in range(len(n_clusters_)):
        cluster = []
        index_array = []
        for labels_index in range(len(labels)):
            if n_clusters_[cluster_Label] == labels[labels_index]:
                cluster.append(surface_normal_vectors_newell[labels_index])
                index_array.append(labels_index)

#        hdbscan implemenatation
        db_scan_labels = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(cluster)
        del cluster     

        print("Number of estimated db scan_ clusters:", len(np.unique(db_scan_labels)))
        db_scan_points = []
        for dbscan_index in range(len(db_scan_labels)):

            distances_output.append(np.array(distances[index_array[dbscan_index]]))
            
            db_scan_points.append(np.array(neighbors[index_array[dbscan_index]]))
        target2_output.append(np.array(db_scan_points))
        del db_scan_points
        del index_array
        
    
    target2_output= list(itertools.chain.from_iterable(target2_output))    
    target2_output = np.asarray(target2_output)    
    target2_output= target2_output.astype('uint32')
        
    distances_output = list(itertools.chain.from_iterable(distances_output))    
    distances_output = np.asarray(distances_output)    
    distances_output= distances_output.astype('float32')
    
    
    target2 = (target2_output[:, 1:].flatten()).astype('uint32')
    
    #---knn1-----
    neighbors_output = target2_output[:, 1:k_nn1 + 1]    
    
    graph["source"] = np.matlib.repmat(range(0, num_ver), k_nn1, 1).flatten(order='F').astype('uint32')#repmat--repeat loop for matric mxn times... flatten in column major

    graph["target"] = np.transpose(neighbors_output.flatten(order='C')).astype('uint32')#flatten in row major
    graph["distances"] = distances_output.flatten().astype('float32')#by default flatten in row major
   
    return graph, target2
#------------------------------------------------------------------------------
def compute_sp_graph(xyz, d_max, in_component, components, labels, n_labels):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component)+1
    in_component = np.array(in_component)
    has_labels = len(labels) > 0
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    #---compute delaunay triangulation---
    tri = Delaunay(xyz)
    #interface select the edges between different components
    #edgx and edgxr converts from tetrahedrons to edges
	#done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r ,edg5r, edg6r))
    del edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)
    #---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0,:] +  edge_comp[1,:]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    #marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1
    #---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_count"] = np.zeros((n_com, 1), dtype='uint64')
    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_norm"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_delta_centroid"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_length_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_surface_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_volume_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_point_count_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    #---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                , bins=[float(i)-0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp,:])
        graph["sp_point_count"][i_com] = len(comp)
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp
            graph["sp_length"][i_com] = 0
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        else:
            ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0]) #descending order
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            try:
                graph["sp_length"][i_com] = ev[0]
            except TypeError:
                graph["sp_length"][i_com] = 0
            try:
                graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
            except TypeError:
                graph["sp_surface"][i_com] = 0
            try:
                graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
            except TypeError:
                graph["sp_volume"][i_com] = 0
    #---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        graph["source"][i_sedg] = com_source
        graph["target"][i_sedg] = com_target
        #---compute the ratio features---
        graph["se_delta_centroid"][i_sedg,:] = graph["sp_centroids"][com_source,:] - graph["sp_centroids"][com_target, :]
        graph["se_length_ratio"][i_sedg] = graph["sp_length"][com_source] / (graph["sp_length"][com_target] + 1e-6)
        graph["se_surface_ratio"][i_sedg] = graph["sp_surface"][com_source] / (graph["sp_surface"][com_target] + 1e-6)
        graph["se_volume_ratio"][i_sedg] = graph["sp_volume"][com_source] / (graph["sp_volume"][com_target] + 1e-6)
        graph["se_point_count_ratio"][i_sedg] = graph["sp_point_count"][com_source] / (graph["sp_point_count"][com_target] + 1e-6)
        #---compute the offset set---
        delta = xyz_source - xyz_target
        if len(delta > 1):
            graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
            graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            graph["se_delta_norm"][i_sedg] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
        else:
            graph["se_delta_mean"][i_sedg, :] = delta
            graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
            graph["se_delta_norm"][i_sedg] = np.sqrt(np.sum(delta ** 2))
    return graph
