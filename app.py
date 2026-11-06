"""InundaGraph IA · Interfaz del prototipo (etapa 2 del ABP).

Red vial real de zonas piloto de Cartagena (OpenStreetMap vía OSMnx),
escenarios de inundación simulados y rutas mínimas (Dijkstra). Los resultados
son simulaciones académicas y no instrucciones de evacuación.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import osmnx as ox

from core import analysis, criticality, geodata
from core.network import (
    apply_scenario,
    calculate_route,
    classify_edges_by_radius,
    intersection_name,
    route_metadata,
    route_streets,
)
from core.visualization import draw_network_interactive
from core.geodata import PILOT_ZONES, OSM_ATTRIBUTION

st.set_page_config(
    page_title="InundaGraph IA",
    page_icon="🌊",
    layout="wide",
)
title_column, help_column = st.columns([0.88, 0.12], vertical_alignment="center")
with title_column:
    st.title("🌊 InundaGraph IA")
    st.caption("Explora rutas y escenarios simulados de inundación en Cartagena.")
with help_column:
    show_help_button = st.button(
        "?",
        key="show_help",
        help="Ayuda sobre el mapa y los controles",
        type="tertiary",
        width="stretch",
    )


@st.dialog("Ayuda rápida · InundaGraph IA", width="medium", icon=":material/help:")
def show_help():
    st.markdown(
        """
**¿Qué puedes hacer?**  
Explorar rutas entre puntos de referencia y observar cómo cambiarían ante
un escenario de inundación simulado.

**Cómo usarla**
1. Elige una zona y los puntos de origen y destino en la barra lateral.
2. Selecciona un centro de inundación y ajusta el radio y el núcleo de cierre.
3. En el mapa, pasa el cursor —o enfoca con el teclado— sobre los marcadores
   para previsualizar cada centro. La vista previa no cambia la ruta elegida.
4. Revisa la ruta, el costo estimado y los indicadores debajo del mapa.

**Glosario de parámetros del escenario**
- **Centro de la inundación:** referencia desde la que se mide la distancia
  hacia las vías. «Sin inundación» no cierra ni afecta vías.
- **Radio de afectación (km):** distancia exterior máxima del escenario.
  Las vías más lejanas que este radio quedan sin cambios; al aumentarlo,
  más vías pueden quedar afectadas o cerradas.
- **Radio interior de cierre (%):** qué proporción del radio de afectación
  forma el núcleo de cierre total. Por ejemplo, con 0,35 km y 50 %, se cierran
  las vías a hasta 0,175 km del centro; las vías entre 0,175 y 0,35 km quedan
  afectadas parcialmente. Si se aumenta, más vías se cierran y menos quedan
  solo afectadas.
- **Nivel de afectación (re, 0–1):** intensidad asignada a cada vía afectada
  parcialmente. Un valor mayor incrementa su costo; no cambia el número de
  vías afectadas ni cerradas.
- **Penalización del costo (β):** cuánto pesa esa afectación al buscar la ruta.
  Se calcula `costo = tiempo base × (1 + β × re)`. Por ejemplo, re = 0,6 y
  β = 1 aumentan en 60 % el costo de cada vía afectada. β = 0 elimina ese
  recargo, pero no revierte cierres ni cambia la clasificación de las vías.

**Colores del mapa**
- Verde: ruta calculada.
- Ámbar: vía afectada en la vista previa.
- Rojo: vía cerrada en la vista previa.
- Rombo dorado: punto de referencia.

Los valores son simulaciones académicas; no representan niveles de agua
observados ni deben usarse para tomar decisiones de evacuación.
        """
    )


if show_help_button:
    show_help()


@st.cache_resource(show_spinner="Cargando la red vial de la zona piloto…")
def load_graph(zone_key: str, force_refresh: bool = False):
    return geodata.load_zone_graph(zone_key, force_refresh=force_refresh)


@st.cache_resource(show_spinner="Proyectando la red vial a coordenadas métricas…")
def load_projected_graph(zone_key: str):
    return ox.project_graph(load_graph(zone_key))


@st.cache_data(show_spinner="Analizando criticidad de la red…")
def load_criticality_indicators(
    zone_key: str,
    center_node,
    safe_points: tuple,
    flood_radius: float,
    closure_share: float,
    severity: float,
    beta: float,
):
    graph = load_graph(zone_key)
    projected_graph = load_projected_graph(zone_key)
    blocked_edges, affected_edges = classify_edges_by_radius(
        projected_graph,
        center_node,
        flood_radius * 1000.0,
        closure_share,
    )
    scenario_graph = apply_scenario(
        graph,
        blocked_edges=blocked_edges,
        affected_edges=affected_edges,
        severity=severity,
        beta=beta,
    )
    isolation = analysis.isolated_nodes(graph, scenario_graph, safe_points)
    return analysis.structured_indicators(
        graph,
        scenario_graph,
        safe_points,
        blocked_edges=blocked_edges,
        affected_edges=affected_edges,
        isolation=isolation,
    )


# ------------------------------------------------------------------ #
# Barra lateral: zona, geografía y escenario simulado
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Análisis")
    zone_key = st.selectbox(
        "Zona de análisis",
        options=list(PILOT_ZONES),
        format_func=lambda key: PILOT_ZONES[key].name,
    )
    zone = PILOT_ZONES[zone_key]
    st.caption(zone.description)

    with st.expander("Datos de la red", expanded=False):
        refresh = st.button(
            "Actualizar datos de OpenStreetMap",
            help="Vuelve a descargar la red vial de esta zona.",
            width="stretch",
        )
        if refresh:
            st.cache_resource.clear()
            st.cache_data.clear()

    graph = load_graph(zone_key, force_refresh=refresh)
    projected_graph = load_projected_graph(zone_key)
    with st.expander("Estado de la red", expanded=False):
        st.caption(
            f"{graph.number_of_nodes():,} intersecciones · "
            f"{graph.number_of_edges():,} segmentos"
        )
        st.caption(f"Datos OSM: {graph.graph.get('downloaded_utc', 'n/d')}")

    info = {n: d for n, d in graph.nodes(data=True)}
    landmarks = zone.landmarks
    landmark_nodes = {}
    for label, (lat, lon) in landmarks.items():
        try:
            landmark_nodes[label] = geodata.nearest_node(graph, lat, lon)
        except Exception:  # ref dataset vacío o asimétrico
            continue

    st.divider()
    st.subheader("Ruta")
    labels = list(landmark_nodes)  # solo las referencias que resolvieron
    if not labels:
        st.error("Ninguna referencia de la zona pudo anclar a la red vial.")
        st.stop()
    source_label = st.selectbox("Desde", labels, index=0)
    destination_label = st.selectbox(
        "Hasta (punto de análisis)",
        labels,
        index=min(1, len(labels) - 1),
        help=(
            "Destino para comparar rutas y accesibilidad; no implica que sea "
            "un refugio oficial o verificado."
        ),
    )
    chosen_source = landmark_nodes[source_label]
    chosen_target = landmark_nodes[destination_label]
    safe_points = sorted({n for n in landmark_nodes.values()})

    st.divider()
    st.subheader("Escenario simulado")
    flood_center_label = st.selectbox(
        "Centro de la inundación",
        options=["Sin inundación", *labels],
        index=0,
        help="El mapa permite comparar la vista previa de cada centro.",
    )

    flood_radius = st.slider(
        "Radio de afectación (km)",
        min_value=0.0,
        max_value=1.2,
        value=0.35,
        step=0.05,
        help="Las vías dentro de este radio se afectan o se cierran.",
    )
    closure_share = st.slider(
        "Radio interior de cierre (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Proporción interior del radio en la que las vías se cierran.",
    )
    with st.expander("Parámetros avanzados", expanded=False):
        severity = st.slider(
            "Nivel de afectación",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Intensidad de la afectación parcial (0–1).",
        )
        beta = st.slider(
            "Penalización del costo",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.25,
            help="Cuánto influye la afectación en el costo de la ruta.",
        )

# ------------------------------------------------------------------ #
# Cálculo de escenarios
# ------------------------------------------------------------------ #
affected_edges: list[tuple] = []
blocked_edges: list[tuple] = []
preview_scenarios = {}
for label, center_node in landmark_nodes.items():
    preview_blocked, preview_affected = classify_edges_by_radius(
        projected_graph,
        center_node,
        flood_radius * 1000.0,
        closure_share,
    )
    preview_scenarios[label] = {
        "blocked_edges": preview_blocked,
        "affected_edges": preview_affected,
    }

if flood_center_label != "Sin inundación" and flood_radius > 0:
    scenario = preview_scenarios[flood_center_label]
    blocked_edges = scenario["blocked_edges"]
    affected_edges = scenario["affected_edges"]

scenario_graph = apply_scenario(
    base_graph=graph,
    blocked_edges=blocked_edges,
    affected_edges=affected_edges,
    severity=severity,
    beta=beta,
)

base_result = calculate_route(graph, chosen_source, chosen_target)
scenario_result = calculate_route(scenario_graph, chosen_source, chosen_target)

isolation: set = set()
if blocked_edges or affected_edges:
    isolation = analysis.isolated_nodes(graph, scenario_graph, safe_points)

# ------------------------------------------------------------------ #
# Métricas de ruta
# ------------------------------------------------------------------ #
c1, c2, c3 = st.columns(3)
if scenario_result:
    length_km = route_metadata(graph, scenario_result.path)["distance_km"]
    c1.metric("Costo con escenario", f"{scenario_result.cost:.1f} min",
              help="Suma de we = te·(1 + β·re) sobre la ruta mínima.")
    c2.metric("Longitud de ruta", f"{length_km:.2f} km")
    if base_result and base_result.cost > 0:
        delta = scenario_result.cost - base_result.cost
        c3.metric("Variación de costo", f"{delta:+.1f} min", delta_color="inverse")
    else:
        c3.metric("Variación de costo", "Sin ruta base")
elif base_result:
    length_km = route_metadata(graph, base_result.path)["distance_km"]
    c1.metric("Costo con escenario", "—", help="Sin ruta disponible.")
    c2.metric("Ruta base", f"{base_result.cost:.1f} min · {length_km:.2f} km")
    c3.metric("Estado del escenario", "Ruta interrumpida")

if not scenario_result and not base_result:
    st.error("Ni la red base ni el escenario tienen ruta entre los puntos elegidos.")
if not scenario_result and base_result:
    st.error(
        "⚠ El escenario **interrumpe por completo** la ruta entre los puntos "
        f"elegidos. {len(isolation)} intersecciones pierden acceso a todos "
        "los destinos configurados."
    )

# ------------------------------------------------------------------ #
# Mapa y comparativo
# ------------------------------------------------------------------ #
st.subheader("Mapa de la zona y ruta calculada")
map_html = draw_network_interactive(
    graph=projected_graph,
    route=scenario_result.path if scenario_result else (base_result.path if base_result else None),
    blocked_edges=blocked_edges,
    affected_edges=affected_edges,
    safe_points=None if not landmark_nodes else safe_points,
    isolated=isolation,
    scenario_centers=landmark_nodes,
    preview_scenarios=preview_scenarios,
    selected_center=landmark_nodes.get(flood_center_label),
    flood_radius_km=flood_radius,
    closure_share=closure_share,
    title="Ruta con escenario simulado",
    show_attribution=True,
)
components.html(map_html, height=840, scrolling=False)
st.caption(
    "Pasa el cursor por un marcador para comparar: ámbar = afectada, "
    "rojo = cerrada, verde = ruta. " + OSM_ATTRIBUTION + "."
)

st.subheader("Indicadores de conectividad y criticidad")
reachability = analysis.reachability_summary(scenario_graph, safe_points)
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Accesibilidad a puntos de referencia",
    f"{reachability * 100:.1f} %",
    help=(
        "Porcentaje de intersecciones que aún puede llegar a algún punto de "
        "referencia de la zona. Estos puntos no están validados como refugios."
    ),
)
m2.metric("Nodos aislados", f"{len(isolation):,}")
m3.metric("Vías cerradas", f"{len(blocked_edges):,}")
m4.metric("Vías afectadas", f"{len(affected_edges):,}")

with st.expander("Intersecciones críticas", expanded=False):
    st.caption(
        "Clasificación explicable mediante reglas del equipo; no es una "
        "predicción del árbol ni una evaluación de riesgo real."
    )
    if flood_center_label == "Sin inundación" or not (
        blocked_edges or affected_edges
    ):
        st.info("Selecciona un escenario que afecte vías para ver su análisis.")
    else:
        indicators = load_criticality_indicators(
            zone_key,
            landmark_nodes[flood_center_label],
            tuple(safe_points),
            flood_radius,
            closure_share,
            severity,
            beta,
        )
        critical_nodes = []
        for record in indicators["nodes"]:
            label, reason = criticality.explain_node_criticality(record)
            if label != "baja":
                critical_nodes.append((record, label, reason))

        priority = {"alta": 0, "media": 1}
        critical_nodes.sort(
            key=lambda item: (
                priority[item[1]],
                -item[0]["blocked_edges"],
                -item[0]["affected_edges"],
            )
        )
        counts = {
            label: sum(1 for _, node_label, _ in critical_nodes if node_label == label)
            for label in ("alta", "media")
        }
        st.caption(
            f"Alta: {counts['alta']} · Media: {counts['media']}"
        )
        if not critical_nodes:
            st.success("Este escenario no activa los criterios de criticidad.")
        else:
            for record, label, reason in critical_nodes[:8]:
                st.write(
                    f"{label.capitalize()} · "
                    f"{intersection_name(graph, record['node'])} — {reason}"
                )
            if len(critical_nodes) > 8:
                st.caption(
                    f"Se muestran 8 de {len(critical_nodes)} intersecciones; "
                    "se priorizan las de criticidad alta."
                )

with st.expander("Ver instrucciones de la ruta", expanded=False):
    base_route_column, scenario_route_column = st.columns(2)
    with base_route_column:
        st.markdown("**Sin inundación**")
        if base_result:
            base_md = route_metadata(graph, base_result.path)
            for step, street in enumerate(
                route_streets(graph, base_result.path), start=1
            ):
                st.write(f"{step}. {street}")
            st.caption(
                f"{base_result.cost:.1f} min · "
                f"{base_md['distance_km']:.2f} km"
            )
        else:
            st.warning("No hay ruta para este recorrido.")

    with scenario_route_column:
        st.markdown("**Con el escenario seleccionado**")
        if scenario_result:
            scen_md = route_metadata(graph, scenario_result.path)
            for step, street in enumerate(
                route_streets(scenario_graph, scenario_result.path), start=1
            ):
                st.write(f"{step}. {street}")
            st.caption(
                f"{scenario_result.cost:.1f} min · "
                f"{scen_md['distance_km']:.2f} km"
            )
        else:
            st.warning("El escenario interrumpe esta ruta.")
