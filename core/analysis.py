"""Indicadores de estructura, conectividad y aislamiento de la red.

Sustenta teóricamente (ABP §9.5): grado de un vértice, intermediación y
impacto de retirar vías o vértices sobre la accesibilidad hacia los
destinos seguros.

Nota de eficiencia: para decidir qué intersecciones pierden acceso a los
puntos seguros no se corre un Dijkstra por cada nodo. Basta con invertir el
grafo y correr un Dijkstra desde cada punto seguro; los nodos alcanzados
desde el punto seguro en el grafo invertido son exactamente los que podían
llegar a él en el grafo original.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx


def node_degree(graph: nx.Graph, node) -> int:
    """Número de conexiones incidentes de la intersección."""
    return int(graph.degree(node))


def approximate_betweenness(
    graph: nx.Graph,
    sample_size: int = 120,
    weight: str = "weight_min",
) -> dict:
    """Intermediación aproximada por muestreo de caminos mínimos.

    El muestreo mantiene el cálculo viable en la interfaz; es una
    estimación, no el valor exacto, y así se documenta ante el usuario.
    """
    if graph.number_of_nodes() == 0:
        return {}

    sample = min(sample_size, graph.number_of_nodes())
    if sample < 3:
        return {node: 0.0 for node in graph.nodes}

    return nx.betweenness_centrality(
        graph, k=sample, normalized=True, weight=weight, seed=7
    )


def reachable_in(graph: nx.Graph, source, targets: Iterable) -> set:
    """Destinos alcanzables desde un origen con un Dijkstra."""
    lengths = nx.single_source_dijkstra_path_length(
        graph, source, weight="weight_min"
    )
    return {target for target in targets if target in lengths}


def access_set(graph: nx.Graph, safe_points: Iterable) -> set:
    """Nodos con acceso a al menos un punto seguro (grafo invertido)."""
    safe_points = [point for point in safe_points if point in graph]
    if not safe_points:
        return set()

    reversed_graph = graph.reverse(copy=False)
    reachable: set = set()
    for point in safe_points:
        lengths = nx.single_source_dijkstra_path_length(
            reversed_graph, point, weight="weight_min"
        )
        reachable.update(lengths)
    return reachable


def isolated_nodes(
    base_graph: nx.Graph,
    scenario_graph: nx.Graph,
    safe_points: Iterable,
) -> set:
    """Nodos que pierden todo acceso a los puntos seguros en el escenario."""
    base_access = access_set(base_graph, safe_points)
    scenario_access = access_set(scenario_graph, safe_points)
    return set(base_access) - set(scenario_access)


def reachability_summary(graph: nx.Graph, safe_points: Iterable) -> float:
    """Proporción de nodos que aún alcanzan algún punto seguro."""
    if graph.number_of_nodes() == 0:
        return 0.0
    return len(access_set(graph, safe_points)) / graph.number_of_nodes()


def disconnected_from_point(scenario_graph: nx.Graph, origin, safe_point) -> bool:
    """Indica si un origen perdió el acceso a un punto específico."""
    try:
        lengths = nx.single_source_dijkstra_path_length(
            scenario_graph, origin, weight="weight_min"
        )
    except nx.NodeNotFound:
        return True
    return safe_point not in lengths


def road_has_multiple_alternatives(
    graph: nx.Graph, start, end, max_paths: int = 3
) -> bool:
    """Comprueba si existen rutas mínimas alternativas fuera de (start, end)."""
    sanitized = graph.copy()
    for key in list(sanitized.get_edge_data(start, end) or {}):
        sanitized.remove_edge(start, end, key=key)
    try:
        nx.dijkstra_path(sanitized, start, end, weight="weight_min")
        return True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False


def structured_indicators(
    base_graph: nx.Graph,
    scenario_graph: nx.Graph,
    safe_points: Iterable,
    blocked_edges: Iterable[tuple] = (),
    affected_edges: Iterable[tuple] = (),
    isolation: set | None = None,
    baseline_betweenness: dict | None = None,
) -> dict:
    """Une todos los indicadores por nodo en un solo cálculo reutilizable."""
    safe_points = [point for point in safe_points if point in base_graph]
    blocked = {frozenset(edge) for edge in blocked_edges}
    affected = {frozenset(edge) for edge in affected_edges}

    if isolation is None:
        isolation = isolated_nodes(base_graph, scenario_graph, safe_points)

    if baseline_betweenness is None:
        baseline_betweenness = approximate_betweenness(base_graph)
    scenario_betweenness = approximate_betweenness(scenario_graph)

    records = []
    for node in base_graph.nodes:
        if base_graph.is_directed():
            if base_graph.is_multigraph():
                incident_edges = list(base_graph.in_edges(node, keys=False))
                incident_edges.extend(base_graph.out_edges(node, keys=False))
            else:
                incident_edges = list(base_graph.in_edges(node))
                incident_edges.extend(base_graph.out_edges(node))
        elif base_graph.is_multigraph():
            incident_edges = base_graph.edges(node, keys=False)
        else:
            incident_edges = base_graph.edges(node)
        incident_roads = {frozenset((start, end)) for start, end in incident_edges}
        affected_count = len((incident_roads & affected) - blocked)
        blocked_count = len(incident_roads & blocked)
        records.append(
            {
                "node": node,
                "degree": node_degree(base_graph, node),
                "scenario_degree": node_degree(scenario_graph, node)
                if node in scenario_graph
                else 0,
                "betweenness": baseline_betweenness.get(node, 0.0),
                "scenario_betweenness": scenario_betweenness.get(node, 0.0),
                "affected_edges": affected_count,
                "blocked_edges": blocked_count,
                "isolated": node in isolation,
            }
        )

    return {
        "nodes": records,
        "isolated_count": len(isolation),
        "reachability": reachability_summary(scenario_graph, safe_points),
    }
