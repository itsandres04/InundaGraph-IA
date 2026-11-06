"""Motor de rutas y modelado de escenarios.

Fórmula de costo adoptada del documento ABP (InundaGraph IA, §9.3):

    we = te * (1 + beta * re)

donde te es el tiempo base del segmento, re ∈ [0, 1] es el nivel de afectación
simulado y beta ≥ 0 controla cuánto penaliza el escenario. Los costos permanecen
no negativos, condición necesaria para Dijkstra. Los escenarios son
experimentales: no representan mediciones hidrológicas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import networkx as nx
from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class RouteResult:
    """Resultado de una búsqueda de ruta de menor costo."""

    path: list
    cost: float


def adjusted_cost(base_time: float, impact_ratio: float, beta: float) -> float:
    """Aplica la fórmula experimental: we = te * (1 + beta * re)."""
    return float(base_time) * (1.0 + float(beta) * float(impact_ratio))


def create_demo_graph() -> nx.DiGraph:
    """Red dirigida sintética para pruebas del motor sin conexión a red."""
    graph = nx.DiGraph()

    demo_positions: dict[str, tuple[float, float]] = {
        "A": (0.0, 1.0), "B": (1.0, 2.0), "C": (1.0, 0.0),
        "D": (2.0, 2.0), "E": (2.0, 0.0), "F": (3.0, 1.0),
    }
    demo_roads: tuple[tuple[str, str, float, float], ...] = (
        ("A", "B", 4.0, 0.40), ("A", "C", 3.0, 0.30), ("B", "C", 2.0, 0.22),
        ("B", "D", 3.0, 0.31), ("C", "E", 4.0, 0.42), ("D", "E", 2.0, 0.20),
        ("D", "F", 4.0, 0.40), ("E", "F", 3.0, 0.32),
    )

    for node, (x, y) in demo_positions.items():
        graph.add_node(node, x=x, y=y)

    for start, end, time_min, distance_km in demo_roads:
        attributes = {
            "base_time_min": float(time_min),
            "distance_km": float(distance_km),
            "weight_min": float(time_min),
            "impact_ratio": 0.0,
        }
        graph.add_edge(start, end, **attributes)
        graph.add_edge(end, start, **attributes)

    return graph


def get_road_options(graph: nx.Graph) -> dict[str, tuple]:
    """Devuelve cada segmento no dirigido una sola vez para la interfaz.

    El valor del diccionario es la pareja (origen, destino), válida para
    bloquear o afectar todas las aristas paralelas entre ambos extremos.
    """
    options: dict[str, tuple] = {}
    seen: set[frozenset] = set()

    for start, end, *_ in graph.edges:
        road = frozenset((start, end))
        if road in seen:
            continue
        seen.add(road)
        # La pareja (start, end) es la que usa apply_scenario para afectar o
        # cerrar todas las aristas paralelas entre los mismos extremos.
        options[f"{start} — {end}"] = (start, end)

    return options


def apply_scenario(
    base_graph: nx.DiGraph | nx.MultiDiGraph,
    blocked_edges: Iterable[tuple] = (),
    affected_edges: Iterable[tuple] = (),
    severity: float = 0.0,
    beta: float = 1.0,
) -> nx.Graph:
    """Copia la red, retira las vías cerradas y ajusta costos por afectación.

    Los parámetros blocked_edges y affected_edges reciben parejas (u, v).
    Si alguna vía sufre bloqueo y afectación a la vez, prevalece el bloqueo.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity debe estar entre 0 y 1 (escala re del ABP).")
    if beta < 0:
        raise ValueError("beta no puede ser negativo.")

    graph = base_graph.copy()
    blocked = {frozenset(edge) for edge in blocked_edges}
    affected = {frozenset(edge) for edge in affected_edges}

    for start, end, data in list(graph.edges(data=True)):
        road = frozenset((start, end))
        if road in blocked:
            graph.remove_edge(start, end)
            continue

        base_time = float(data.get("base_time_min", 0.0))
        road_severity = severity if road in affected else 0.0
        data["impact_ratio"] = road_severity
        data["weight_min"] = adjusted_cost(base_time, road_severity, beta)

    return graph


def calculate_route(
    graph: nx.DiGraph | nx.MultiDiGraph,
    source,
    target,
) -> RouteResult | None:
    """Calcula la ruta mínimo-costo con Dijkstra; None si no hay camino."""
    if source not in graph:
        raise nx.NodeNotFound(f"El origen '{source}' no existe en la red.")
    if target not in graph:
        raise nx.NodeNotFound(f"El destino '{target}' no existe en la red.")

    try:
        path = nx.dijkstra_path(graph, source, target, weight="weight_min")
        cost = nx.dijkstra_path_length(graph, source, target, weight="weight_min")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    return RouteResult(path=list(path), cost=float(cost))


def route_metadata(graph: nx.Graph, path: list) -> dict:
    """Distancia y tiempo base acumulados de una ruta calculada."""
    total_km = 0.0
    total_min = 0.0
    for index in range(len(path) - 1):
        candidates = graph.get_edge_data(path[index], path[index + 1]) or {}
        if "base_time_min" in candidates:
            # DiGraph: el dict devuelto son los atributos de la arista única.
            best = candidates
        else:
            # MultiDiGraph: el dict devuelve {key: atributos}; elige la arista
            # cuyo tiempo base es menor.
            best = min(
                candidates.values(),
                key=lambda item: item.get("base_time_min", float("inf")),
            )
        total_km += float(best.get("distance_km", 0.0))
        total_min += float(best.get("base_time_min", 0.0))
    return {"distance_km": total_km, "base_time_min": total_min}


def route_streets(graph: nx.Graph, path: list) -> list[str]:
    """Devuelve los nombres de vías en orden, agrupando tramos consecutivos."""
    streets = []
    for index in range(len(path) - 1):
        candidates = graph.get_edge_data(path[index], path[index + 1]) or {}
        if graph.is_multigraph() and candidates:
            best = min(
                candidates.values(),
                key=lambda item: item.get("weight_min", float("inf")),
            )
        elif candidates:
            best = candidates
        else:
            streets.append("Vía sin nombre en OpenStreetMap")
            continue

        name = best.get("name")
        if isinstance(name, (list, tuple)):
            name = next((value for value in name if value), None)
        name = str(name).strip() if name else ""
        streets.append(name or "Vía sin nombre en OpenStreetMap")

    grouped = []
    for street in streets:
        if not grouped or grouped[-1] != street:
            grouped.append(street)
    return grouped


def intersection_name(graph: nx.Graph, node) -> str:
    """Describe una intersección con los nombres de sus vías conectadas."""
    if node not in graph:
        raise nx.NodeNotFound(f"El nodo '{node}' no existe en la red.")

    if graph.is_directed():
        if graph.is_multigraph():
            edges = list(graph.in_edges(node, keys=True, data=True))
            edges.extend(graph.out_edges(node, keys=True, data=True))
        else:
            edges = list(graph.in_edges(node, data=True))
            edges.extend(graph.out_edges(node, data=True))
    elif graph.is_multigraph():
        edges = list(graph.edges(node, keys=True, data=True))
    else:
        edges = list(graph.edges(node, data=True))

    street_names = set()
    for edge in edges:
        data = edge[-1]
        names = data.get("name", ())
        if isinstance(names, str):
            names = (names,)
        elif not isinstance(names, (list, tuple, set)):
            names = ()
        name = next((str(value).strip() for value in names if value), "")
        if name:
            street_names.add(name)

    ordered_names = sorted(street_names, key=str.casefold)
    if len(ordered_names) >= 2:
        return f"Cruce de {ordered_names[0]} con {ordered_names[1]}"
    if ordered_names:
        return f"Sobre {ordered_names[0]}"
    return "Intersección sin nombre de vía disponible"


def classify_edges_by_radius(
    graph: nx.Graph,
    center,
    radius_m: float,
    closure_share: float,
) -> tuple[list[tuple], list[tuple]]:
    """Clasifica vías por su distancia mínima al centro, en CRS métricos."""
    if radius_m < 0:
        raise ValueError("radius_m no puede ser negativo.")
    if not 0.0 <= closure_share <= 100.0:
        raise ValueError("closure_share debe estar entre 0 y 100.")
    if center not in graph:
        raise nx.NodeNotFound(f"El centro '{center}' no existe en la red.")

    center_data = graph.nodes[center]
    center_point = Point(float(center_data["x"]), float(center_data["y"]))
    closure_radius_m = radius_m * closure_share / 100.0
    min_distances: dict[frozenset, tuple[float, tuple]] = {}

    if graph.is_multigraph():
        edges = graph.edges(keys=True, data=True)
    else:
        edges = ((start, end, None, data) for start, end, data in graph.edges(data=True))

    for start, end, _, data in edges:
        road = frozenset((start, end))
        geometry = data.get("geometry")
        if geometry is None:
            start_data, end_data = graph.nodes[start], graph.nodes[end]
            geometry = LineString(
                [
                    (float(start_data["x"]), float(start_data["y"])),
                    (float(end_data["x"]), float(end_data["y"])),
                ]
            )
        distance = float(geometry.distance(center_point))
        current = min_distances.get(road)
        if current is None or distance < current[0]:
            min_distances[road] = (distance, (start, end))

    blocked_edges = []
    affected_edges = []
    for distance, (start, end) in min_distances.values():
        if distance <= closure_radius_m:
            blocked_edges.append((start, end))
        elif distance <= radius_m:
            affected_edges.append((start, end))

    return blocked_edges, affected_edges
