from models.graph import PaperGraph
from models.paper import Paper
import time
from ortools.sat.python import cp_model
from z3 import *
from typing import List, Dict, Tuple, Set


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


def set_cover(pg : PaperGraph):
    """Takes a subgraph and returns the set of papers that you would need to read to cover all topics"""
    # Collect all the topics 
    topics = {n for n, attr in pg.graph.nodes(data=True) if attr['type'] == 'topic'}
    papers = {n for n, attr in pg.graph.nodes(data=True) if attr['type'] == 'paper'}

    print(topics)
    print(papers)

    topic_to_papers = {topic: set(pg.graph.neighbors(topic)) for topic in topics}

    opt = Optimize()

    # Step 1: Create Boolean variables
    is_chosen = {paper: Bool(f"chosen_{paper}") for paper in papers}

    # Step 2: Add constraints
    for topic, papers_covering_it in topic_to_papers.items():
        opt.add(Or([is_chosen[paper] for paper in papers_covering_it]))

    opt.minimize(Sum(list(is_chosen.values())))

    if opt.check() == sat:
        print('sat')
        model = opt.model()
        chosen_papers = [paper for paper in papers if model.evaluate(is_chosen[paper])]
        print(chosen_papers)
        return chosen_papers
    else:
        print('other')
        return []

def solve_gaps_with_cpsat(gap_data, k):
    model = cp_model.CpModel()

    # Create integer data structures
    select = {}
    scores = {}
    topics = set()

    for (a, b, interesting, cost) in gap_data:
        topics.add(a)
        topics.add(b)

        var = model.NewBoolVar(f"{a}_{b}")
        select[(a,b)] = var

        # integer score to avoid floats in solver
        scores[(a,b)] = int(interesting * 1000) - cost

    # Each topic appears at most once
    # Build adjacency lists of which gap vars use each topic
    topic_to_vars = {t: [] for t in topics}

    for (a, b) in select:
        topic_to_vars[a].append(select[(a,b)])
        topic_to_vars[b].append(select[(a,b)])

    for t in topics:
        if topic_to_vars[t]:
            model.Add(sum(topic_to_vars[t]) <= 1)

    # Choose exactly k gaps
    model.Add(sum(select.values()) == k)

    # Objective: maximize score
    model.Maximize(sum(select[(a,b)] * scores[(a,b)]
                       for (a,b,_,_) in gap_data))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3.0  # optional timeout

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("No feasible solution")
        return []

    selected = [(a,b) for (a,b) in select if solver.Value(select[(a,b)]) == 1]
    return selected

    

def identify_research_gap(pg: PaperGraph, k=5, weight=1, top_n=5000):
    """
    Optimized gap identification:
      1. Computes interestingness for *all* topic pairs
      2. Keeps only the top-N pairs (default: 5000)
      3. Halves the space by removing (b,a) duplicates
      4. Uses Z3 with Integer 0/1 selection variables
      5. Enforces exactly k selections
      6. Maximizes score = weight * interestingness - cost
    """

    start_time = time.time()

    # -------------------------
    # Step 1: topic similarity
    # -------------------------
    def topic_similarity(topic_a, topic_b):
        papers_a = list(pg.graph.neighbors(topic_a))
        papers_b = list(pg.graph.neighbors(topic_b))

        total_similarity = 0
        count = 0
        for pa in papers_a:
            for pb in papers_b:
                if pg.graph.has_edge(pa, pb):
                    total_similarity += pg.graph[pa][pb].get('weight', 0)
                    count += 1
        
        return total_similarity / count if count > 0 else 0

    topics = [n for n, attr in pg.graph.nodes(data=True) if attr['type'] == 'topic']
    num_topics = len(topics)
    print(f"[Gap Analysis] Found {num_topics} topics")

    # -------------------------
    # Step 2: Compute raw gaps
    # -------------------------
    print("[Gap Analysis] Enumerating and scoring gaps...")
    gap_data = []  # (topic_a, topic_b, interestingness, cost)

    for i in range(num_topics):
        for j in range(i + 1, num_topics):  # remove symmetric duplicates
            a = topics[i]
            b = topics[j]

            path_length = len(pg.find_path(a, b))

            # Only consider "gaps" where the connection is long or nonexistent
            if path_length > 4 or path_length == 0:
                sim = topic_similarity(a, b)
                interesting = path_length * sim

                papers_a = len(list(pg.graph.neighbors(a)))
                papers_b = len(list(pg.graph.neighbors(b)))
                cost = papers_a + papers_b

                gap_data.append((a, b, interesting, cost))

    print(f"[Gap Analysis] Total raw gaps: {len(gap_data)}")

    # -------------------------
    # Step 3: keep only top-N interesting
    # -------------------------
    print(f"[Gap Analysis] Selecting top {top_n} most interesting gaps...")
    gap_data.sort(key=lambda x: x[2], reverse=True)
    gap_data = gap_data[:top_n]
    print(f"[Gap Analysis] Reduced to {len(gap_data)} gaps sent to Z3")

    selected = solve_gaps_with_cpsat(gap_data, k)

    print(f"[Gap Analysis] Selected {len(selected)} gaps.")
    return selected

