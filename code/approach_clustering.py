from cluster import *
from differential import *
from frequency_oracle import *
from log import *
from mechanisms import *
from partition import *
from protocols import *
from utils import *


##########################
##  Approach Clustering ##
##########################

class Approach_Clustering(Differential):
    def __init__(self) -> None:
        super(Approach_Clustering, self).__init__()


    def execute(self, graph:Graph, protocols:list[ProtocolInit], edge_heuristic:EdgeConsistencyHeuristic,
                clustering_heuristic:ClusteringHeuristic, n_partitions:np.int_, n_clusters:np.int_,
                top_k_heuristic:TopKHeuristic, k:np.int_, epsilon:np.float_, epsilon_allocation:list[np.int_]) -> Graph:
        log_msg(f'started approach_clustering execution', True)

        # Ensuring the use of 100% of the privacy budget
        assert np.sum(epsilon_allocation) == 100

        if clustering_heuristic == ClusteringHeuristic.FIXED_RANDOM:
            # Ensuring the correct number of epsilon parts
            assert len(epsilon_allocation) == 2

            epsilon_cluster = epsilon * epsilon_allocation[0] / 100
            epsilon_ranl = epsilon * epsilon_allocation[1] / 100

        elif clustering_heuristic == ClusteringHeuristic.DYNAMIC_DEGREE:
            # Ensuring the correct number of epsilon parts
            assert len(epsilon_allocation) == 3

            epsilon_degree = epsilon * epsilon_allocation[0] / 100
            epsilon_cluster = epsilon * epsilon_allocation[1] / 100
            epsilon_ranl = epsilon * epsilon_allocation[2] / 100

        else:
            raise Exception

        # Step 1: build partitions and clusters
        log_msg(f'started step 1: build partitions and clusters')
        n = graph.n_nodes

        partitions = build_partitions(n, n_partitions)
        
        if clustering_heuristic == ClusteringHeuristic.FIXED_RANDOM:
            clusters = build_clusters(n, n_clusters)

        elif clustering_heuristic == ClusteringHeuristic.DYNAMIC_DEGREE:
            fos_edge_type_degrees = dict()

            for edge_type in graph.edge_types.values():
                fo_edge_type_degrees = FrequencyOracleMechanism(n)

                edge_type_degrees = get_edge_type_degrees_from_adj_matrix(graph.adj_matrix, edge_type)
                edge_type_degrees_p = edge_type_degrees + geometric_noise(2, epsilon_degree, n)

                fo_edge_type_degrees.aggregate(edge_type_degrees_p)
                fo_edge_type_degrees.post_process_responses(0, np.max(fo_edge_type_degrees.responses))
                
                fos_edge_type_degrees[edge_type] = fo_edge_type_degrees

            # Use the edge type degrees to estimate the node degrees
            degrees = np.zeros(n, dtype=np.int_)

            for edge_type in graph.edge_types.values():
                degrees += fos_edge_type_degrees[edge_type].responses

            # Adjust nodes with zero degree to have degree of one since every node must have at least one connection
            zero_degree_nodes = np.where(degrees == 0)[0]
            
            edge_types = list(np.arange(graph.n_edge_types))
            edge_type_proportions = get_edge_type_proportions_from_fo(fos_edge_type_degrees, graph.edge_types)
            
            for zero_degree_node in zero_degree_nodes:
                edge_type = np.random.choice(a=edge_types, p=edge_type_proportions)
                
                fos_edge_type_degrees[edge_type].responses[zero_degree_node] = 1
                degrees[zero_degree_nodes] = 1

            clusters = build_clusters_from_degrees(degrees, clustering_heuristic, n_clusters)
            
        else:
            raise Exception
        log_msg(f'finished step 1')

        # Step 2: clustering
        log_msg(f'started step 2: clustering')
        d = len(clusters)

        clusters_fo = dict()

        for partition in partitions:
            protocol_cluster = init_protocol(protocols[0], partition.get_size(), d, epsilon_cluster)
            fo_cluster = FrequencyOracleProtocol(protocol_cluster.n, protocol_cluster.d, protocol_cluster.p, protocol_cluster.q)

            for node_i in partition.elements:
                cluster_id = get_node_cluster_id(graph.adj_matrix, node_i, clusters)
                cluster_bv = get_bit_vector_from_cluster_id(cluster_id, d)
                cluster_bv_p = protocol_cluster.perturb(cluster_bv)

                fo_cluster.aggregate(cluster_bv_p)

            fo_cluster.estimate()

            if clustering_heuristic == ClusteringHeuristic.FIXED_RANDOM:
                cluster_ids_p = get_top_k_partition_cluster_ids(fo_cluster.counts_e, top_k_heuristic, k)

            elif clustering_heuristic == ClusteringHeuristic.DYNAMIC_DEGREE:
                weighted_counts = np.zeros(n_clusters)

                for cluster in clusters:
                    weighted_counts[cluster.id] = fo_cluster.counts_e[cluster.id] * cluster.get_importance()
                
                cluster_ids_p = get_top_k_partition_cluster_ids(weighted_counts, top_k_heuristic, k)
                
            else:
                raise Exception
            
            partition.set_clusters(clusters[cluster_ids_p])

            clusters_fo[partition.id] = fo_cluster
        log_msg(f'finished step 2')

        # Step 3: perturb and aggregate RANL (intra-cluster)
        log_msg(f'started step 3: perturb and aggregate RANL (intra-cluster)')
        ranl_fo = dict()
        
        for partition in partitions:
            d = partition.get_clusters_size() * graph.n_edge_types

            protocol_ranl = init_protocol(protocols[1], partition.get_size(), d, epsilon_ranl)
            fo_ranl = FrequencyOracleProtocol(protocol_ranl.n, protocol_ranl.d, protocol_ranl.p, protocol_ranl.q)

            for node_i in partition.elements:
                adj_matrix_i = graph.adj_matrix[node_i, partition.get_clusters_elements()]

                ranl = get_RANL_from_adj_matrix(adj_matrix_i)
                ranl_p = protocol_ranl.perturb(ranl)

                fo_ranl.aggregate(ranl_p)

            fo_ranl.convert_responses_to_array()

            ranl_fo[partition.id] = fo_ranl
        log_msg(f'finished step 3')

        # Step 4: generate DP graph from perturbed RANLs
        log_msg(f'started step 4: generate DP graph from perturbed RANLs')
        adj_matrix_dp = build_graph_from_clustered_RANLs(ranl_fo, partitions, graph.n_nodes, graph.n_edge_types)
        log_msg(f'finished step 4')

        # Step 5: fix inconsistent edges
        log_msg(f'started step 5: fix inconsistent edges')
        fix_inconsistent_edges(adj_matrix_dp, edge_heuristic)
        log_msg(f'finished step 5')

        # Step 6: adjust graph degrees
        if clustering_heuristic == ClusteringHeuristic.DYNAMIC_DEGREE:
            log_msg(f'started step 6: adjust graph degrees')
            adjust_graph_degrees_by_edge_type_from_dict(adj_matrix_dp, fos_edge_type_degrees)
            log_msg(f'finished step 6')

        # Step 7: fix zero degree nodes
        log_msg(f'started step 7: fix zero degree nodes')
        fix_zero_degree_nodes(adj_matrix_dp)
        log_msg(f'finished step 7')
        
        graph_dp = Graph(graph.name, adj_matrix_dp, graph.edge_types)

        log_msg(f'finished approach_clustering execution')

        return graph_dp
