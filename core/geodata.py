"""Gestión de zonas piloto y carga de redes viales reales (OpenStreetMap).

Giros del documento ABP (InundaGraph IA):
- Los datos geográficos reales provienen de OpenStreetMap vía OSMnx.
- Los datos geográficos se separan de la lógica de análisis.
- La descarga se registra (fecha, área, filtros) y se almacena en caché
  para no repetir peticiones al servidor de Overpass.

Atribución requerida por la licencia ODbL:
    Datos © Colaboradores de OpenStreetMap (openstreetmap.org/copyright)
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import osmnx as ox
import requests

from osmnx.graph import _create_graph as create_graph_from_overpass

ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.requests_headers = {
    "User-Agent": "InundaGraphIA/2.0 (prototipo academico Universidad del Sinu)",
}

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OVERPASS_TIMEOUT_S = 120

# Filtro equivalente a network_type="drive" en OSMnx 2.x.
_DRIVE_FILTER = (
    'way["highway"]'
    '["highway"!~"^(abandoned|bus_guideway|construction|corridor|'
    'cycleway|footway|footpath|path|pedestrian|plinth|platform|'
    'raceway|steps|track)$"]'
    '["service"!~"^(alley|driveway|emergency_access|parking|parking_aisle)$"]'
    '["access"!~"^private$"]'
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OSM_ATTRIBUTION = (
    "Datos © Colaboradores de OpenStreetMap (openstreetmap.org/copyright) · ODbL"
)


@dataclass(frozen=True)
class PilotZone:
    """Zona piloto delimitada por un marco geográfico (bbox)."""

    key: str
    name: str
    north: float
    south: float
    east: float
    west: float
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    description: str = ""


# Nota: los recuadros son aproximados y delimitan una muestra compacta de cada
# sector; no pretenden coincidir con límites administrativos o de planeación.
PILOT_ZONES: dict[str, PilotZone] = {
    "centro_getsemani": PilotZone(
        key="centro_getsemani",
        name="Centro Histórico y Getsemaní",
        north=10.4325,
        south=10.4145,
        east=-75.5355,
        west=-75.5625,
        description="Núcleo histórico compacto: trama angular y vías peatonales.",
        landmarks={
            "Plaza de los Coches": (10.4235, -75.5462),
            "Torre del Reloj": (10.4242, -75.5442),
            "Plaza Trinidad (Getsemaní)": (10.4228, -75.5412),
            "Parque Centenario": (10.4215, -75.5442),
        },
    ),
    "manga": PilotZone(
        key="manga",
        name="Manga",
        north=10.4330,
        south=10.4140,
        east=-75.5060,
        west=-75.5400,
        description="Trama regular de manzanas residenciales sobre Av. Santander.",
        landmarks={
            "Plaza de Toros": (10.4215, -75.5312),
            "Colegio Berchmans": (10.4230, -75.5232),
            "Universidad Tecnológica": (10.4256, -75.5140),
        },
    ),
    "bocagrande": PilotZone(
        key="bocagrande",
        name="Bocagrande",
        north=10.4150,
        south=10.3800,
        east=-75.5355,
        west=-75.5700,
        description="Península con pocas rutas alternativas: ideal para ensayar pérdida de conectividad.",
        landmarks={
            "Playa Bocagrande": (10.4035, -75.5610),
            "Galerías Bocagrande": (10.4000, -75.5480),
            "Castillogrande": (10.4060, -75.5330),
        },
    ),
    "pie_lapopa": PilotZone(
        key="pie_lapopa",
        name="Pie de la Popa y Centro Amplio",
        north=10.4510,
        south=10.4220,
        east=-75.5130,
        west=-75.5520,
        description="Denso núcleo administrativo con avenidas principales.",
        landmarks={
            "Camellón de Marta": (10.4300, -75.5470),
            "Liceo Bolívar": (10.4335, -75.5405),
            "Barrio San Diego": (10.4400, -75.5320),
        },
    ),
}


def _cache_paths(zone_key: str) -> tuple[Path, Path]:
    """Rutas de la caché en disco (grapho rápido y GeoJSON interoperable)."""
    pickle_path = DATA_DIR / f"zone_{zone_key}.graphpickle"
    geojson_path = DATA_DIR / f"zone_{zone_key}.geojson"
    return pickle_path, geojson_path


def _network_config(zone: PilotZone) -> dict:
    """Registra la configuración de la descarga para su documentación."""
    return {
        "network_type": "drive",
        "bbox": (zone.south, zone.west, zone.north, zone.east),
        "simplify": True,
    }


def _overpass_query(zone: PilotZone) -> str:
    """Consulta Overpass con filtro drive, incluida la recursión de nodos."""
    south, west, north, east = _network_config(zone)["bbox"]
    return (
        "[out:json][timeout:{timeout}];"
        "({filter}({south},{west},{north},{east});(._;>;););"
        "out body;"
    ).format(
        timeout=OVERPASS_TIMEOUT_S,
        filter=_DRIVE_FILTER,
        south=south,
        west=west,
        north=north,
        east=east,
    )


def _download_overpass_response(zone: PilotZone) -> dict:
    """Descarga los elementos OSM probando varios servidores con espera.

    Algunos servidores de Overpass rechazan o saturan peticiones; se prueba
    cada endpoint con reintentos y conferencia en espera exponencial.
    """
    query = _overpass_query(zone)
    attempts = 2
    delays = (5, 12)

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(attempts):
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=OVERPASS_TIMEOUT_S,
                    headers={"User-Agent": "InundaGraphIA/2.0 (prototipo academico)"},
                )
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("elements"):
                        return payload
                elif response.status_code in (429, 502, 503, 504):
                    time.sleep(delays[attempt])
                    continue
                else:
                    break  # otro código: probar endpoint siguiente
            except requests.RequestException:
                time.sleep(delays[attempt])
                continue

    raise ConnectionError(
        "No se pudo descargar la red desde ningún servidor de Overpass. "
        "Inténtalo de nuevo más tarde."
    )


def _rebuild_from_geojson(zone_key: str) -> nx.MultiDiGraph | None:
    """Reconstruye el grafo desde GeoJSON cuando el pickle no existe."""
    _, geojson_path = _cache_paths(zone_key)
    if not geojson_path.exists():
        return None

    with geojson_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    graph = nx.MultiDiGraph()
    graph.graph.update(payload.get("graph_meta", {}))
    graph.graph.setdefault("zone", zone_key)
    graph.graph.setdefault("zone_name", PILOT_ZONES[zone_key].name)
    graph.graph.setdefault("attribution", OSM_ATTRIBUTION)
    for feature in payload["features"]:
        item = feature["properties"]
        geometry = feature["geometry"]
        if item.get("element") == "node":
            lon, lat = geometry["coordinates"]
            graph.add_node(item["id"], x=lon, y=lat)
        else:
            graph.add_edge(
                item["source"], item["target"], key=item.get("key", 0), **{k: v for k, v in item.items() if k not in ("element", "source", "target", "key")}
            )
    return graph


def _export_geojson(graph: nx.MultiDiGraph, zone: PilotZone) -> None:
    """Guarda nodos y aristas en GeoJSON (formato interoperable del ABP)."""
    features: list[dict] = []

    for node, data in graph.nodes(data=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [data.get("x"), data.get("y")]},
                "properties": {"element": "node", "id": node},
            }
        )

    for start, end, key, data in graph.edges(keys=True, data=True):
        # Los geometry de Shapely no son JSON-serializables; guardan sus
        # trazos en la geometría del feature cuando están disponibles.
        props = {}
        coords: list = []
        geometry_item = data.get("geometry")
        if geometry_item is not None and hasattr(geometry_item, "coords"):
            coords = [list(point) for point in geometry_item.coords]
        for attr, value in data.items():
            if attr == "geometry":
                continue
            try:
                json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
            props[attr] = value
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "element": "edge",
                    "source": start,
                    "target": end,
                    "key": key,
                    **props,
                },
            }
        )

    payload = {
        "type": "FeatureCollection",
        "graph_meta": {
            "zone": zone.key,
            "zone_name": zone.name,
            "attribution": OSM_ATTRIBUTION,
            "downloaded_utc": graph.graph.get("downloaded_utc"),
            "filters": graph.graph.get("filters", _network_config(zone)),
        },
        "features": features,
    }
    _, geojson_path = _cache_paths(zone.key)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=geojson_path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary_path, geojson_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _build_graph_from_response(response: dict, zone: PilotZone) -> nx.MultiDiGraph:
    """Construye el grafo final desde una respuesta Overpass cruda."""
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo el grafo…", flush=True)
    graph = create_graph_from_overpass([response], bidirectional=False)
    graph = ox.simplification.simplify_graph(graph)
    print(
        f"[{time.strftime('%H:%M:%S')}] Grafo: {graph.number_of_nodes():,} nodos / "
        f"{graph.number_of_edges():,} aristas",
        flush=True,
    )

    config = _network_config(zone)
    graph.graph["zone"] = zone.key
    graph.graph["zone_name"] = zone.name
    graph.graph["attribution"] = OSM_ATTRIBUTION
    graph.graph["downloaded_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    graph.graph["filters"] = config

    # Velocidades implícitas y tiempo de viaje (segundos) por arista.
    graph = ox.routing.add_edge_speeds(graph)
    graph = ox.routing.add_edge_travel_times(graph)

    for _, _, data in graph.edges(data=True):
        length_km = float(data.get("length", 0.0)) / 1000.0
        travel_min = float(data.get("travel_time", 0.0)) / 60.0
        data["distance_km"] = round(length_km, 4)
        data["base_time_min"] = round(travel_min, 4)
        data["weight_min"] = round(travel_min, 4)
        data["impact_level"] = 0.0

    return graph


def load_zone_graph(zone_key: str, force_refresh: bool = False) -> nx.MultiDiGraph:
    """Carga la red vial de una zona piloto, con caché en disco.

    Primera llamada: descarga desde OpenStreetMap, calcula velocidades y
    tiempos de recorrido, y guarda en caché. Llamadas siguientes: carga
    instantánea desde disco. Con caché intermedia: reutiliza la respuesta
    cruda de Overpass sin repetir la petición al servidor.
    """
    if zone_key not in PILOT_ZONES:
        raise ValueError(f"Zona desconocida: '{zone_key}'.")

    zone = PILOT_ZONES[zone_key]
    pickle_path, _ = _cache_paths(zone_key)
    raw_path = DATA_DIR / f"zone_{zone.key}.overpass.json"

    if not force_refresh and pickle_path.exists():
        import pickle

        with pickle_path.open("rb") as handle:
            return pickle.load(handle)

    if not force_refresh and raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            response = json.load(handle)
        if response.get("elements"):
            print(
                f"[{time.strftime('%H:%M:%S')}] Reutilizando respuesta cruda "
                f"cacheada de '{zone.key}'…",
                flush=True,
            )
            return _build_and_store(zone, response)

    if not force_refresh:
        graph = _rebuild_from_geojson(zone_key)
        if graph is not None:
            import pickle

            with pickle_path.open("wb") as handle:
                pickle.dump(graph, handle)
            return graph

    print(f"[{time.strftime('%H:%M:%S')}] Descargando '{zone.key}'…", flush=True)
    response = _download_overpass_response(zone)
    print(
        f"[{time.strftime('%H:%M:%S')}] Elementos recibidos: "
        f"{len(response['elements']):,}",
        flush=True,
    )
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(response, handle, ensure_ascii=False)

    return _build_and_store(zone, response)


def _build_and_store(zone: PilotZone, response: dict) -> nx.MultiDiGraph:
    """Construye el grafo, guarda el pickle y el GeoJSON."""
    graph = _build_graph_from_response(response, zone)

    import pickle

    pickle_path, _ = _cache_paths(zone.key)
    with pickle_path.open("wb") as handle:
        pickle.dump(graph, handle)

    _export_geojson(graph, zone)
    return graph


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float):
    """Intersección del grafo más cercana a una coordenada geográfica."""
    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)


def zone_download_summary(zone_key: str) -> dict:
    """Resumen de la descarga registrada en el grafo cacheado."""
    pickle_path, _ = _cache_paths(zone_key)
    if not pickle_path.exists():
        return {"cached": False}
    import pickle

    with pickle_path.open("rb") as handle:
        graph = pickle.load(handle)
    return {
        "cached": True,
        "zone": zone_key,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "downloaded_utc": graph.graph.get("downloaded_utc"),
        "attribution": graph.graph.get("attribution"),
    }
