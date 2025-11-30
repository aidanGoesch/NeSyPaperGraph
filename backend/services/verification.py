from models.graph import PaperGraph
from models.paper import Paper

from z3 import *

from typing import List, Dict, Tuple, Set

def get_transitive_synonym_groups(topics: List[str], topic_synonyms: Dict[str, List[str]]) -> List[Set[str]]:
    """Groups topics that must merge due to direct or indirect synonym relationships."""
    parent = {topic: topic for topic in topics}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Build the full, symmetrical synonym map
    full_synonyms = {}
    for topic_a, syn_list in topic_synonyms.items():
        if topic_a in topics:
            full_synonyms.setdefault(topic_a, []).extend(syn_list)
        for topic_b in syn_list:
            if topic_b in topics:
                full_synonyms.setdefault(topic_b, []).append(topic_a)

    # Union all connected topics
    for topic_a, syn_list in full_synonyms.items():
        for topic_b in syn_list:
            if topic_a in topics and topic_b in topics:
                union(topic_a, topic_b)

    # Collect groups based on the final root
    groups: Dict[str, Set[str]] = {}
    for topic in topics:
        root = find(topic)
        groups.setdefault(root, set()).add(topic)

    return list(groups.values())


def verify_bipartite(pg: PaperGraph, new_nodes: set = None, new_edges: set = None) -> bool:
    """
    Verify bipartiteness of graph. If new_nodes/new_edges provided, only verify those incrementally.
    
    Args:
        pg: PaperGraph to verify
        new_nodes: Set of new node names to verify (None = verify all)
        new_edges: Set of new edge tuples to verify (None = verify all)
    """
    solver = Solver()

    # If incremental, only check new elements
    if new_nodes is not None or new_edges is not None:
        # Construct mapping only for relevant nodes
        relevant_nodes = set()
        if new_nodes:
            relevant_nodes.update(new_nodes)
        if new_edges:
            for n1, n2 in new_edges:
                relevant_nodes.add(n1)
                relevant_nodes.add(n2)
        
        is_paper = {node: Bool(f"paper_{node.replace(' ', '_')}") for node in relevant_nodes}
        
        # Add constraints for relevant nodes
        for node in relevant_nodes:
            if node not in pg.graph.nodes():
                continue
            if pg.graph.nodes[node].get('type') == 'paper':
                solver.add(is_paper[node])
            else:
                solver.add(Not(is_paper[node]))
        
        # Only check new edges
        edges_to_check = new_edges if new_edges else []
        for node1, node2 in edges_to_check:
            if node1 in is_paper and node2 in is_paper:
                solver.assert_and_track(Xor(is_paper[node1], is_paper[node2]), f"edge_{node1}_{node2}")
    else:
        # Full verification (original behavior)
        is_paper = {node: Bool(f"paper_{node.replace(' ', '_')}") for node in pg.graph.nodes()}

        for node in pg.graph.nodes():
            if pg.graph.nodes[node].get('type') == 'paper':
                solver.add(is_paper[node])
            else:
                solver.add(Not(is_paper[node]))

        for node1, node2 in pg.graph.edges():
            solver.assert_and_track(Xor(is_paper[node1], is_paper[node2]), f"edge_{node1}_{node2}")

    result = solver.check()

    if result == sat:
        print("constraints are satisfied")
        return True
    
    elif result == unsat:
        print("constraints are unsatisfied")
        print("problem edges:")
        for edge in solver.unsat_core():
            print("\t" + str(edge))
        return False
    
    return False

def find_optimal_topic_merge(topics: list, topic_synonyms: dict) -> dict:
    """Returns a dict mapping group_id -> list of topics in that group."""
    
    # Union-Find data structure
    parent = {topic: topic for topic in topics}
    
    def find(topic):
        if parent[topic] == topic:
            return topic
        parent[topic] = find(parent[topic])  # Path compression
        return parent[topic]
    
    def union(topic_a, topic_b):
        root_a = find(topic_a)
        root_b = find(topic_b)
        if root_a != root_b:
            parent[root_a] = root_b
    
    # Union all synonym pairs
    for topic_a, synonym_list in topic_synonyms.items():
        if topic_a not in topics:
            continue
        for topic_b in synonym_list:
            if topic_b not in topics:
                continue
            union(topic_a, topic_b)
    
    # Group topics by their root
    groups = {}
    for topic in topics:
        root = find(topic)
        if root not in groups:
            groups[root] = []
        groups[root].append(topic)
    
    # Convert to indexed format
    return {i: topic_list for i, topic_list in enumerate(groups.values())}
    parent = {topic: topic for topic in topics}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Build the full, symmetrical synonym map
    full_synonyms = {}
    for topic_a, syn_list in topic_synonyms.items():
        if topic_a in topics:
            full_synonyms.setdefault(topic_a, []).extend(syn_list)
        for topic_b in syn_list:
            if topic_b in topics:
                full_synonyms.setdefault(topic_b, []).append(topic_a)

    # Union all connected topics
    for topic_a, syn_list in full_synonyms.items():
        for topic_b in syn_list:
            if topic_a in topics and topic_b in topics:
                union(topic_a, topic_b)

    # Collect groups based on the final root
    groups: Dict[str, Set[str]] = {}
    for topic in topics:
        root = find(topic)
        groups.setdefault(root, set()).add(topic)

    return list(groups.values())


def verify_bipartite(pg: PaperGraph, new_nodes: set = None, new_edges: set = None) -> bool:
    """
    Verify bipartiteness of graph. If new_nodes/new_edges provided, only verify those incrementally.
    
    Args:
        pg: PaperGraph to verify
        new_nodes: Set of new node names to verify (None = verify all)
        new_edges: Set of new edge tuples to verify (None = verify all)
    """
    solver = Solver()

    # If incremental, only check new elements
    if new_nodes is not None or new_edges is not None:
        # Construct mapping only for relevant nodes
        relevant_nodes = set()
        if new_nodes:
            relevant_nodes.update(new_nodes)
        if new_edges:
            for n1, n2 in new_edges:
                relevant_nodes.add(n1)
                relevant_nodes.add(n2)
        
        is_paper = {node: Bool(f"paper_{node.replace(' ', '_')}") for node in relevant_nodes}
        
        # Add constraints for relevant nodes
        for node in relevant_nodes:
            if node not in pg.graph.nodes():
                continue
            if pg.graph.nodes[node].get('type') == 'paper':
                solver.add(is_paper[node])
            else:
                solver.add(Not(is_paper[node]))
        
        # Only check new edges
        edges_to_check = new_edges if new_edges else []
        for node1, node2 in edges_to_check:
            if node1 in is_paper and node2 in is_paper:
                solver.assert_and_track(Xor(is_paper[node1], is_paper[node2]), f"edge_{node1}_{node2}")
    else:
        # Full verification (original behavior)
        is_paper = {node: Bool(f"paper_{node.replace(' ', '_')}") for node in pg.graph.nodes()}

        for node in pg.graph.nodes():
            if pg.graph.nodes[node].get('type') == 'paper':
                solver.add(is_paper[node])
            else:
                solver.add(Not(is_paper[node]))

        for node1, node2 in pg.graph.edges():
            solver.assert_and_track(Xor(is_paper[node1], is_paper[node2]), f"edge_{node1}_{node2}")

    result = solver.check()

    if result == sat:
        print("constraints are satisfied")
        return True
    
    elif result == unsat:
        print("constraints are unsatisfied")
        print("problem edges:")
        for edge in solver.unsat_core():
            print("\t" + str(edge))
        return False
    
    return False

def find_optimal_topic_merge(topics: list, topic_synonyms: dict) -> dict:
    """Returns a dict mapping group_id -> list of topics in that group."""
    
    # Union-Find data structure
    parent = {topic: topic for topic in topics}
    
    def find(topic):
        if parent[topic] == topic:
            return topic
        parent[topic] = find(parent[topic])  # Path compression
        return parent[topic]
    
    def union(topic_a, topic_b):
        root_a = find(topic_a)
        root_b = find(topic_b)
        if root_a != root_b:
            parent[root_a] = root_b
    
    # Union all synonym pairs
    for topic_a, synonym_list in topic_synonyms.items():
        if topic_a not in topics:
            continue
        for topic_b in synonym_list:
            if topic_b not in topics:
                continue
            union(topic_a, topic_b)
    
    # Group topics by their root
    groups = {}
    for topic in topics:
        root = find(topic)
        if root not in groups:
            groups[root] = []
        groups[root].append(topic)
    
    # Convert to indexed format
    return {i: topic_list for i, topic_list in enumerate(groups.values())}