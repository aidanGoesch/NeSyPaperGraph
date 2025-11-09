from .services.graph_builder import GraphBuilder

if __name__ == "__main__":
    # Example usage:
    builder = GraphBuilder()
    graph = builder.build_graph(".")  # Replace with actual path
    print(f"Graph has {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges.")