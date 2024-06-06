from cluster import *
from log import *
from partition import *

from enum import Enum
import numpy as np


class EdgeConsistencyHeuristic(Enum):
    RANDOM = 'random'
    AND = 'and'


class TopKHeuristic(Enum):
    FIXED = 'fixed'
    COUNT_PERCENTIL = 'count_percentil'


class ClusteringHeuristic(Enum):
    FIXED_RANDOM = 'fixed_random'
    DYNAMIC_DEGREE = 'dynamic_degree'


def post_process_counts_float(counts_vector:np.ndarray, expected_sum:np.int_, tolerance:np.int_=1) -> np.ndarray:
    counts_vector_copy = counts_vector.copy().astype(np.float_)

    while (np.fabs(np.sum(counts_vector_copy) - expected_sum) > tolerance) or (counts_vector_copy < 0).any():
        if (counts_vector_copy <= 0).all():
            counts_vector_copy = expected_sum / len(counts_vector_copy)
            break
        
        counts_vector_copy[counts_vector_copy < 0] = 0
        
        total = np.sum(counts_vector_copy)
        mask = counts_vector_copy > 0
        diff = (expected_sum - total) / np.sum(mask)
        
        counts_vector_copy[mask] += diff

    return counts_vector_copy


def post_process_counts_int(input_arr:np.ndarray, expected_sum:np.int_, min_value:np.int_, max_value:np.int_) -> np.ndarray:
    arr = np.array(input_arr).astype(np.int_)

    if expected_sum < ((len(arr) * min_value)):
        return (np.ones(len(arr)) * min_value).astype(np.int_)

    arr = np.clip(arr, min_value, max_value)
    
    exceding_units = np.int_(np.sum(arr) - expected_sum)

    if exceding_units > 0:
        sign = -1
    elif exceding_units < 0:
        sign = 1
    else:
        sign = 0

    exceding_units = np.abs(exceding_units)

    if sign == -1:
        while exceding_units > len(np.where(arr > min_value)[0]):
            exceding_units -= len(arr[np.where(arr > min_value)[0]])
            arr[np.where(arr > min_value)[0]] += sign

        if exceding_units > 0:
            p = np.ones(len(arr)).astype(np.int_)
            p[np.where(arr <= min_value)[0]] = 0
            prob = p / np.sum(p)
            idx_to_decrease_1 = np.random.choice(len(arr), exceding_units, replace=False, p=prob)
            arr[idx_to_decrease_1] += sign

    elif sign == 1:
        while exceding_units > len(arr):
            exceding_units -= len(arr)
            arr += sign

        if exceding_units > 0:
            idx_to_increase_1 = np.random.choice(len(arr), exceding_units, replace=False)
            arr[idx_to_increase_1] += sign
    
    if (np.sum(arr) != expected_sum) and (len(np.where(arr < min_value)[0]) > 0):
        print('error in min l2 method')

    return arr


def get_RANL_from_adj_matrix(adj_matrix:np.ndarray) -> np.ndarray:
    return adj_matrix.flatten()


def get_edge_type_degrees_from_adj_matrix(adj_matrix:np.ndarray, edge_type:np.int_) -> np.ndarray:
    return np.sum(adj_matrix, axis=1)[:, edge_type]


def get_edge_type_counts_from_adj_matrix(adj_matrix:np.ndarray) -> np.ndarray:
    return np.sum(np.sum(adj_matrix, axis=1), axis=0)


def get_edge_type_proportions_from_adj_matrix(adj_matrix:np.ndarray) -> np.ndarray:
    edge_type_counts = get_edge_type_counts_from_adj_matrix(adj_matrix)

    return edge_type_counts / np.sum(edge_type_counts)


def build_graph_from_RANLs(ranls:np.ndarray, n_nodes:np.int_, n_edge_types:np.int_) -> np.ndarray:
    adj_matrix = ranls.reshape(n_nodes, n_nodes, n_edge_types)

    return adj_matrix


def adjust_graph_degrees_by_edge_type_from_dict(adj_matrix:np.ndarray, edge_types_degrees:dict) -> None:
    for edge_type in edge_types_degrees.keys():
        degrees = get_edge_type_degrees_from_adj_matrix(adj_matrix, edge_type)
        expected_degrees = edge_types_degrees[edge_type].responses
        
        last_degrees = np.array([])

        while not np.array_equal(degrees, expected_degrees):
            if np.array_equal(degrees, last_degrees):
                break

            last_degrees = degrees.copy()
            
            # Remove edges
            nodes_to_remove_edges = np.where(degrees > expected_degrees)[0]
            np.random.shuffle(nodes_to_remove_edges)

            for node_idx in nodes_to_remove_edges:
                if degrees[node_idx] <= expected_degrees[node_idx]:
                    continue

                candidate_nodes = np.where((adj_matrix[node_idx, :, edge_type] == 1) & (degrees > expected_degrees))[0]

                candidate_nodes = list(candidate_nodes)
                np.random.shuffle(candidate_nodes)

                while (degrees[node_idx] > expected_degrees[node_idx]) and (len(candidate_nodes) > 0):
                    dest_node = candidate_nodes.pop(0)

                    if (degrees[dest_node] > expected_degrees[dest_node]) and (node_idx != dest_node):
                        adj_matrix[node_idx, dest_node, edge_type] = 0
                        adj_matrix[dest_node, node_idx, edge_type] = 0

                        degrees[node_idx] -= 1
                        degrees[dest_node] -= 1

            # Add edges
            nodes_to_add_edges = np.where(degrees < expected_degrees)[0]
            np.random.shuffle(nodes_to_add_edges)

            for node_idx in nodes_to_add_edges:
                if degrees[node_idx] >= expected_degrees[node_idx]:
                    continue
                
                candidate_nodes = np.where((adj_matrix[node_idx, :, edge_type] == 0) & (degrees < expected_degrees))[0]

                candidate_nodes = list(candidate_nodes)
                np.random.shuffle(candidate_nodes)

                while (degrees[node_idx] < expected_degrees[node_idx]) and (len(candidate_nodes) > 0):
                    dest_node = candidate_nodes.pop(0)

                    if (degrees[dest_node] < expected_degrees[dest_node]) and (node_idx != dest_node):
                        adj_matrix[node_idx, dest_node, edge_type] = 1
                        adj_matrix[dest_node, node_idx, edge_type] = 1

                        degrees[node_idx] += 1
                        degrees[dest_node] += 1

                    else:
                        continue


def remove_self_edges(adj_matrix:np.ndarray) -> None:
    n_nodes = adj_matrix.shape[0]
    node_indices = np.arange(n_nodes)
    adj_matrix[node_indices, node_indices] = False


def fix_inconsistent_edges(adj_matrix:np.ndarray, heuristic:EdgeConsistencyHeuristic) -> None:
    remove_self_edges(adj_matrix)

    if heuristic == EdgeConsistencyHeuristic.RANDOM:
        n_nodes = adj_matrix.shape[0]
        
        choice = np.random.choice(['triu', 'tril'])

        if choice == 'triu':
            tril_indices = np.tril_indices(n_nodes, k=-1)
            adj_matrix[tril_indices[0], tril_indices[1]] = adj_matrix[tril_indices[1], tril_indices[0]]

        elif choice == 'tril':
            triu_indices = np.triu_indices(n_nodes, k=1)
            adj_matrix[triu_indices[0], triu_indices[1]] = adj_matrix[triu_indices[1], triu_indices[0]]

        else:
            raise Exception

    elif heuristic == EdgeConsistencyHeuristic.AND:
        adj_matrix_transpose = transpose_3d_matrix(adj_matrix)
        adj_matrix[:] = adj_matrix & adj_matrix_transpose

    else:
        raise Exception


def build_subsets(n_nodes:np.int_, n_subsets:np.int_) -> list[np.ndarray]:
    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    subsets = np.array_split(indices, n_subsets)

    return subsets


def build_partitions(n_nodes:np.int_, n_partitions:np.int_) -> np.ndarray:
    subsets = build_subsets(n_nodes, n_partitions)
    partitions = np.array([Partition(id, elements) for id, elements in enumerate(subsets)])

    return partitions


def build_clusters(n_nodes:np.int_, n_clusters:np.int_) -> np.ndarray:
    subsets = build_subsets(n_nodes, n_clusters)
    clusters = np.array([Cluster(id, elements) for id, elements in enumerate(subsets)])
    
    return clusters


def build_clusters_from_degrees(degrees:np.ndarray, clustering_heuristic:ClusteringHeuristic, n_clusters:np.int_) -> np.ndarray:
    sorted_node_degrees_indices = np.argsort(degrees)[::-1]

    if clustering_heuristic == ClusteringHeuristic.DYNAMIC_DEGREE:
        cumsum = np.cumsum(degrees[sorted_node_degrees_indices])
        degrees_sum_per_cluster = np.sum(degrees) / n_clusters

        clusters = []

        for i in range(n_clusters):
            inf_part = degrees_sum_per_cluster * i
            sup_part = degrees_sum_per_cluster * (i + 1)

            if i < (n_clusters - 1):
                node_ids = sorted_node_degrees_indices[np.where((cumsum > inf_part) & (cumsum <= sup_part))[0]]
            else:
                node_ids = sorted_node_degrees_indices[np.where(cumsum > inf_part)[0]]

            clusters.append(node_ids)

        clusters = np.array([Cluster(id, elements, degrees[elements]) for id, elements in enumerate(clusters)])

    else:
        raise Exception   
    
    return clusters


def get_node_cluster_id(adj_matrix:np.ndarray, node_idx:np.int_, clusters:list[Cluster]) -> np.int_:
    n_clusters = len(clusters)
    n_edges_per_cluster = np.zeros(n_clusters, dtype=np.int_)

    for cluster in clusters:
        n_edges_per_cluster[cluster.id] = np.sum(adj_matrix[node_idx, cluster.elements])

    cluster_id = np.random.choice(np.where(n_edges_per_cluster == n_edges_per_cluster.max())[0])

    return cluster_id


def get_top_k_partition_cluster_ids(estimated_counts:np.ndarray, top_k_heuristic:TopKHeuristic, k:np.int_) -> np.ndarray:
    if top_k_heuristic == TopKHeuristic.FIXED:
        cluster_ids = np.argsort(estimated_counts)[::-1][:k]

    elif top_k_heuristic == TopKHeuristic.COUNT_PERCENTIL:
        min_acceptable_count = np.percentile(estimated_counts, (k))
        cluster_ids = np.where(estimated_counts >= min_acceptable_count)[0]

    else:
        raise Exception

    return cluster_ids


def get_bit_vector_from_cluster_id(cluster_id:np.int_, d:np.int_) -> np.ndarray:
    bit_vector = np.full(d, False, dtype=np.bool_)
    bit_vector[cluster_id] = True

    return bit_vector


def build_graph_from_clustered_RANLs(ranl_fo:dict, partitions:np.ndarray, n_nodes:np.int_, n_edge_types:np.int_) -> np.ndarray:
    adj_matrix = np.full((n_nodes, n_nodes, n_edge_types), False, dtype=np.bool_)

    for partition in partitions:
        ranls = ranl_fo[partition.id].responses

        for idx, node_i in enumerate(partition.elements):
            ranl = ranls[idx]
            ranl_m = ranl.reshape(partition.get_clusters_size(), n_edge_types)

            adj_matrix[node_i, partition.get_clusters_elements()] = ranl_m

    return adj_matrix


def get_degrees_from_adj_matrix(adj_matrix:np.ndarray) -> np.ndarray:
    return np.sum(np.sum(adj_matrix, axis=1), axis=1).astype(np.int_)


def transpose_3d_matrix(adj_matrix:np.ndarray) -> np.ndarray:
    n_nodes, _, n_edge_types = adj_matrix.shape

    n_rows = n_nodes
    n_cols = n_nodes * n_edge_types

    adj_matrix_2d = adj_matrix.reshape(n_rows, n_cols)
    adj_matrix_2d_transpose = np.full((n_rows, n_cols), False, dtype=np.bool_)

    for node_idx in range(n_rows):
        from_node_idx = node_idx * n_edge_types
        to_node_idx = from_node_idx + n_edge_types

        adj_matrix_2d_transpose[:, from_node_idx:to_node_idx] = adj_matrix_2d[node_idx].reshape(n_rows, n_edge_types)

    return adj_matrix_2d_transpose.reshape(n_nodes, n_nodes, n_edge_types)


def fix_zero_degree_nodes(adj_matrix:np.ndarray) -> np.ndarray:
    n_nodes, _, n_edge_types = adj_matrix.shape

    degrees = get_degrees_from_adj_matrix(adj_matrix)

    zero_degree_nodes = list(np.where(degrees == 0)[0])
    np.random.shuffle(zero_degree_nodes)

    edge_types = list(range(n_edge_types))
    edge_type_proportions = get_edge_type_proportions_from_adj_matrix(adj_matrix)
    
    # Until there is a zero degree node
    while len(zero_degree_nodes) > 0:
        node_i_idx = zero_degree_nodes.pop(0)
        node_j_idx = np.random.randint(n_nodes)

        while node_i_idx == node_j_idx:
            node_j_idx = np.random.randint(n_nodes)

        edge_type = np.random.choice(a=edge_types, p=edge_type_proportions)

        adj_matrix[node_j_idx, node_i_idx, edge_type] = 1
        adj_matrix[node_i_idx, node_j_idx, edge_type] = 1

        if node_j_idx in zero_degree_nodes:
            zero_degree_nodes.remove(node_j_idx)


def get_edge_type_proportions_from_fos(fos:dict, edge_types:dict) -> np.ndarray:
    edge_type_counts = np.zeros(len(edge_types))
    
    for edge_type in edge_types.values():
        edge_type_counts[edge_type] = np.sum(fos[edge_type].responses)

    return edge_type_counts / np.sum(edge_type_counts)
