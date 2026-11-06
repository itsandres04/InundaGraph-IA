"""Visualización de la red vial sobre coordenadas geográficas.

Funciona tanto con la red sintética de pruebas (coordenadas esquemáticas)
como con redes reales de OpenStreetMap (x = longitud, y = latitud).
"""
from __future__ import annotations

from html import escape

import networkx as nx
import matplotlib.pyplot as plt

from core.network import intersection_name

COLOR_EDGE = "#9AA4B2"
COLOR_BLOCKED = "#D9534F"
COLOR_ROUTE = "#16865B"
COLOR_SAFE = "#E3A008"
COLOR_ISOLATED = "#8B0000"


def draw_network_interactive(
    graph: nx.Graph,
    route=None,
    blocked_edges=None,
    affected_edges=None,
    safe_points=None,
    isolated=None,
    scenario_centers=None,
    preview_scenarios=None,
    selected_center=None,
    flood_radius_km: float = 0.0,
    closure_share: float = 0.0,
    title: str = "Red vial del escenario",
    show_attribution: bool = False,
) -> str:
    """Genera un mapa SVG con geometrías OSM y vista previa al pasar el cursor."""
    width, height, margin = 1000.0, 820.0, 44.0
    raw_positions = {
        node: (float(data["x"]), float(data["y"]))
        for node, data in graph.nodes(data=True)
        if data.get("x") is not None and data.get("y") is not None
    }
    if not raw_positions:
        raise ValueError("La red no contiene coordenadas para dibujar el mapa.")

    road_shapes = {}
    road_names: dict[frozenset, set[str]] = {}
    edges = (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else ((start, end, None, data) for start, end, data in graph.edges(data=True))
    )
    for start, end, _, data in edges:
        key = frozenset((start, end))
        names = data.get("name", ())
        if isinstance(names, str):
            names = (names,)
        elif not isinstance(names, (list, tuple, set)):
            names = ()
        road_names.setdefault(key, set()).update(
            str(name).strip() for name in names if name and str(name).strip()
        )
        if key in road_shapes or start not in raw_positions or end not in raw_positions:
            continue
        geometry = data.get("geometry")
        if geometry is None:
            road_shapes[key] = [[raw_positions[start], raw_positions[end]]]
        else:
            parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
            road_shapes[key] = [
                [(float(x), float(y)) for x, y, *_ in part.coords]
                for part in parts
                if hasattr(part, "coords") and len(part.coords) >= 2
            ]

    all_points = [
        point
        for shapes in road_shapes.values()
        for shape in shapes
        for point in shape
    ] or list(raw_positions.values())
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    scale = min(
        (width - 2 * margin) / max(max_x - min_x, 1e-9),
        (height - 2 * margin) / max(max_y - min_y, 1e-9),
    )
    positions = {
        node: (
            margin + (x - min_x) * scale,
            height - margin - (y - min_y) * scale,
        )
        for node, (x, y) in raw_positions.items()
    }

    def svg_point(coordinate):
        x = margin + (coordinate[0] - min_x) * scale
        y = height - margin - (coordinate[1] - min_y) * scale
        return f"{x:.2f},{y:.2f}"

    def draw_shape(key, color, stroke_width, class_name="", title_text=None):
        shapes = road_shapes.get(key, [])
        class_attribute = f' class="{class_name}"' if class_name else ""
        tooltip_attribute = (
            f' data-tooltip="{escape(title_text, quote=True)}"'
            if title_text is not None
            else ""
        )
        return "".join(
            f'<polyline{class_attribute}{tooltip_attribute} '
            f'points="{" ".join(svg_point(point) for point in shape)}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            'stroke-linecap="round" stroke-linejoin="round"></polyline>'
            for shape in shapes
        )

    def road_name(key):
        names = sorted(road_names.get(key, ()), key=str.casefold)
        return " / ".join(names) if names else "Vía sin nombre en OpenStreetMap"

    def road_tooltip(key, state):
        return f"Vía {state}: {road_name(key)}"

    closed = {frozenset(edge) for edge in (blocked_edges or [])}
    affected = {frozenset(edge) for edge in (affected_edges or [])}
    route_edges = {
        frozenset((route[index], route[index + 1]))
        for index in range(len(route or []) - 1)
    }

    lines = []
    for key in road_shapes:
        if key in closed or key in affected:
            lines.append(draw_shape(key, COLOR_EDGE, 1.8, "road-base"))
        else:
            lines.append(
                draw_shape(
                    key,
                    COLOR_EDGE,
                    1.8,
                    "unaffected-road",
                    road_tooltip(key, "sin afectación"),
                )
            )
    for edge in affected:
        lines.append(draw_shape(edge, "#FFFFFF", 6.5))
        lines.append(
            draw_shape(
                edge,
                "#B7791F",
                4.5,
                "affected-road",
                road_tooltip(edge, "afectada"),
            )
        )
    for edge in closed:
        lines.append(draw_shape(edge, "#FFFFFF", 8))
        lines.append(
            draw_shape(
                edge,
                COLOR_BLOCKED,
                6,
                "blocked-road",
                road_tooltip(edge, "cerrada"),
            )
        )
    for edge in route_edges:
        lines.append(draw_shape(edge, "#FFFFFF", 9, "route-halo"))
        lines.append(draw_shape(edge, COLOR_ROUTE, 6.5, "route-road"))

    safe_points = set(safe_points or [])
    safe_markers = []
    for node in safe_points:
        if node not in positions:
            continue
        x, y = positions[node]
        safe_markers.append(
            f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" '
            'transform="rotate(45 '
            f'{x:.2f} {y:.2f})" fill="{COLOR_SAFE}" stroke="#7A5A00" />'
        )

    isolated_markers = []
    for node in set(isolated or []):
        if node in positions:
            x, y = positions[node]
            node_name = intersection_name(graph, node)
            tooltip = escape(f"Intersección: {node_name}", quote=True)
            isolated_markers.append(
                f'<circle class="isolated-marker" cx="{x:.2f}" cy="{y:.2f}" r="5" '
                f'fill="{COLOR_ISOLATED}" stroke="white" stroke-width="1.5" '
                f'data-tooltip="{tooltip}" aria-label="{tooltip}" tabindex="0"></circle>'
            )

    radius = max(0.0, float(flood_radius_km)) * 1000.0 * scale
    closure_radius = radius * min(100.0, max(0.0, float(closure_share))) / 100.0
    center_markers = []
    preview_overlays = []
    for marker_index, (label, node) in enumerate((scenario_centers or {}).items()):
        if node not in positions:
            continue
        x, y = positions[node]
        safe_label = escape(str(label), quote=True)
        selected_class = " selected" if node == selected_center else ""
        preview_id = f"scenario-preview-{marker_index}"
        center_markers.append(
            f'<g class="scenario-marker{selected_class}" data-preview="{preview_id}" '
            f'data-x="{x:.2f}" data-y="{y:.2f}" '
            'tabindex="0" role="img" '
            f'aria-label="Previsualizar escenario: {safe_label}">'
            f'<title>{safe_label} — pasa el cursor para ver el área de afectación</title>'
            f'<circle class="preview-area" cx="{x:.2f}" cy="{y:.2f}" '
            f'r="{radius:.2f}" />'
            f'<circle class="preview-closure" cx="{x:.2f}" cy="{y:.2f}" '
            f'r="{closure_radius:.2f}" />'
            f'<circle class="marker-halo" cx="{x:.2f}" cy="{y:.2f}" r="10" />'
            f'<circle class="marker-dot" cx="{x:.2f}" cy="{y:.2f}" r="5.5" />'
            f'<text class="marker-label" x="{x + 10:.2f}" y="{y - 10:.2f}">'
            f'{safe_label}</text></g>'
        )
        scenario = (preview_scenarios or {}).get(label, {})
        preview_blocked = {
            frozenset(edge) for edge in scenario.get("blocked_edges", [])
        }
        preview_affected = {
            frozenset(edge) for edge in scenario.get("affected_edges", [])
        }
        preview_overlays.append(
            f'<g id="{preview_id}" class="preview-overlay">'
            + "".join(
                draw_shape(key, "#FFFFFF", 14, "preview-road-halo")
                + draw_shape(
                    key,
                    "#E87500",
                    9,
                    "preview-road affected-road",
                    road_tooltip(key, "afectada"),
                )
                for key in preview_affected
            )
            + "".join(
                draw_shape(key, "#FFFFFF", 16, "preview-road-halo")
                + draw_shape(
                    key,
                    "#C62828",
                    10,
                    "preview-road blocked-road",
                    road_tooltip(key, "cerrada"),
                )
                for key in preview_blocked
            )
            + "</g>"
        )

    attribution = (
        '<text x="12" y="805" class="attribution">'
        '© Colaboradores de OpenStreetMap · ODbL</text>'
        if show_attribution
        else ""
    )
    map_title = escape(
        f"{title}{' · ' + str(graph.graph.get('zone_name')) if graph.graph.get('zone_name') else ''}"
    )

    return f"""
<div class="interactive-map">
  <style>
    .interactive-map {{ position: relative; font-family: sans-serif; }}
    .interactive-map svg {{ display: block; width: 100%; height: auto; background: #fff; }}
    .map-title {{ font-size: 19px; font-weight: 600; fill: #202124; }}
    .scenario-marker {{ cursor: pointer; outline: none; }}
    .preview-area {{ fill: #e3a008; fill-opacity: 0; stroke: #b7791f; stroke-opacity: 0; stroke-width: 3; transition: fill-opacity .15s, stroke-opacity .15s; }}
    .preview-closure {{ fill: #d9534f; fill-opacity: 0; stroke: #b33a35; stroke-opacity: 0; stroke-width: 2.5; transition: fill-opacity .15s, stroke-opacity .15s; }}
    .scenario-marker:hover .preview-area, .scenario-marker:focus .preview-area {{ fill-opacity: .035; stroke-opacity: .9; }}
    .scenario-marker:hover .preview-closure, .scenario-marker:focus .preview-closure {{ fill-opacity: .06; stroke-opacity: .95; }}
    .preview-overlay {{ display: none; pointer-events: none; }}
    .preview-overlay.active {{ display: inline; pointer-events: stroke; }}
    .preview-road-halo, .route-halo, .route-road {{ pointer-events: none; }}
    .blocked-road, .affected-road, .unaffected-road {{ pointer-events: stroke; cursor: help; }}
    .isolated-marker {{ cursor: help; }}
    .preview-area, .preview-closure {{ pointer-events: none; }}
    .marker-halo {{ fill: #fff; stroke: #7a5a00; stroke-width: 1.5; }}
    .marker-dot {{ fill: #e3a008; stroke: #7a5a00; stroke-width: 1.3; }}
    .scenario-marker.selected .marker-dot {{ fill: #d9534f; }}
    .marker-label {{ display: none; font-size: 13px; font-weight: 600; fill: #202124; paint-order: stroke; stroke: white; stroke-width: 4px; stroke-linejoin: round; pointer-events: none; }}
    .scenario-marker:hover .marker-label, .scenario-marker:focus .marker-label {{ display: block; }}
    .attribution {{ font-size: 10px; fill: #666; }}
    .map-tooltip {{ display: none; position: absolute; z-index: 5; max-width: 280px; padding: 8px 11px; border-radius: 6px; background: #202124; color: #fff; font-size: 13px; line-height: 1.35; box-shadow: 0 2px 8px #0004; pointer-events: none; }}
    .map-tooltip.visible {{ display: block; }}
  </style>
  <svg id="scenario-map" viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="{map_title}">
    <text x="24" y="32" class="map-title">{map_title}</text>
    <g fill="none" stroke-linecap="round" stroke-linejoin="round">{''.join(lines)}</g>
    {''.join(preview_overlays)}
    <g>{''.join(safe_markers)}{''.join(isolated_markers)}</g>
    <g>{''.join(center_markers)}</g>
    {attribution}
  </svg>
  <div class="map-tooltip" role="status" aria-live="polite"></div>
  <script>
    const map = document.currentScript.parentElement;
    const svg = map.querySelector("#scenario-map");
    const markers = map.querySelectorAll(".scenario-marker");
    const tooltip = map.querySelector(".map-tooltip");
    const originalViewBox = svg.getAttribute("viewBox").split(" ").map(Number);
    const clearPreview = () => map.querySelectorAll(".preview-overlay.active")
      .forEach((overlay) => overlay.classList.remove("active"));
    const showPreview = (marker) => {{
      clearPreview();
      const overlay = map.querySelector("#" + marker.dataset.preview);
      if (overlay) overlay.classList.add("active");
    }};
    const zoomAtPointer = (event) => {{
      const pointer = svg.createSVGPoint();
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      const mapPoint = pointer.matrixTransform(svg.getScreenCTM().inverse());
      const zoom = 1.7;
      const width = originalViewBox[2] / zoom;
      const height = originalViewBox[3] / zoom;
      const x = mapPoint.x - (mapPoint.x - originalViewBox[0]) / zoom;
      const y = mapPoint.y - (mapPoint.y - originalViewBox[1]) / zoom;
      svg.setAttribute("viewBox", `${{x}} ${{y}} ${{width}} ${{height}}`);
    }};
    const resetZoom = () => svg.setAttribute("viewBox", originalViewBox.join(" "));
    const showRoadTooltip = (event, target) => {{
      tooltip.textContent = target.dataset.tooltip;
      tooltip.classList.add("visible");
      const bounds = map.getBoundingClientRect();
      const tooltipBounds = tooltip.getBoundingClientRect();
      const targetBounds = target.getBoundingClientRect();
      const clientX = Number.isFinite(event.clientX)
        ? event.clientX
        : targetBounds.left + targetBounds.width / 2;
      const clientY = Number.isFinite(event.clientY)
        ? event.clientY
        : targetBounds.top + targetBounds.height / 2;
      const left = Math.min(
        clientX - bounds.left + 12,
        bounds.width - tooltipBounds.width - 8
      );
      const top = Math.min(
        clientY - bounds.top + 12,
        bounds.height - tooltipBounds.height - 8
      );
      tooltip.style.left = `${{Math.max(8, left)}}px`;
      tooltip.style.top = `${{Math.max(8, top)}}px`;
    }};
    const hideRoadTooltip = () => tooltip.classList.remove("visible");
    map.querySelectorAll(
      ".blocked-road, .affected-road, .unaffected-road, .isolated-marker"
    ).forEach((target) => {{
      target.addEventListener("pointerenter", (event) => showRoadTooltip(event, target));
      target.addEventListener("pointermove", (event) => showRoadTooltip(event, target));
      target.addEventListener("pointerleave", hideRoadTooltip);
      target.addEventListener("focus", (event) => showRoadTooltip(event, target));
      target.addEventListener("blur", hideRoadTooltip);
    }});
    markers.forEach((marker) => {{
      marker.addEventListener("mouseenter", (event) => {{
        showPreview(marker);
        zoomAtPointer(event);
      }});
      marker.addEventListener("mouseleave", () => {{
        clearPreview();
        resetZoom();
      }});
      marker.addEventListener("focus", () => showPreview(marker));
      marker.addEventListener("blur", () => {{
        clearPreview();
        resetZoom();
      }});
    }});
  </script>
</div>
"""


def draw_network(
    graph: nx.Graph,
    route=None,
    blocked_edges=None,
    affected_edges=None,
    safe_points=None,
    isolated=None,
    title: str = "Red vial del escenario",
    show_attribution: bool = False,
):
    """Dibuja nodos y aristas, resaltando ruta, cierres y puntos seguros."""
    closed = {frozenset(edge) for edge in (blocked_edges or [])}
    affected = {frozenset(edge) for edge in (affected_edges or [])}
    safe_points = list(safe_points or [])
    isolated = set(isolated or [])

    route_edges: set[frozenset] = set()
    if route and len(route) > 1:
        route_edges = {
            frozenset((route[i], route[i + 1])) for i in range(len(route) - 1)
        }

    # Posiciones: grafos reales usan x (longitud) e y (latitud).
    positions = {
        node: (data.get("x"), data.get("y"))
        for node, data in graph.nodes(data=True)
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    drawn: set[frozenset] = set()
    edges = graph.edges(keys=False) if graph.is_multigraph() else graph.edges
    for start, end in edges:
        key = frozenset((start, end))
        if key in drawn:
            continue
        drawn.add(key)
        if start not in positions or end not in positions:
            continue
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        if key in closed:
            continue
        if key in route_edges:
            continue
        ax.plot(
            [x1, x2], [y1, y2],
            linewidth=2.4 if key in affected else 1.0,
            color="#B7791F" if key in affected else COLOR_EDGE,
            zorder=1,
        )

    for key in closed:
        if len(key) != 2:
            continue
        start, end = tuple(key)
        if start not in positions or end not in positions:
            continue
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax.plot([x1, x2], [y1, y2], linewidth=3.6, color=COLOR_BLOCKED, zorder=2)

    for key in route_edges:
        start, end = tuple(key)
        if start not in positions or end not in positions:
            continue
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax.plot([x1, x2], [y1, y2], linewidth=3.6, color=COLOR_ROUTE, zorder=3)

    if safe_points:
        xs = [positions[n][0] for n in safe_points if n in positions]
        ys = [positions[n][1] for n in safe_points if n in positions]
        ax.scatter(
            xs, ys, marker="D", s=130, color=COLOR_SAFE,
            edgecolors="#7A5A00", linewidths=1.2, zorder=4,
            label="Punto seguro",
        )

    if isolated:
        xs = [positions[n][0] for n in isolated if n in positions]
        ys = [positions[n][1] for n in isolated if n in positions]
        ax.scatter(
            xs, ys, marker="o", s=110, color=COLOR_ISOLATED,
            edgecolors="white", linewidths=1.0, zorder=5,
            label="Nodo aislado",
        )

    # Etiquetas solo con pocos nodos (red sintética); con OSM se omiten.
    if graph.number_of_nodes() <= 12:
        nx.draw_networkx_labels(
            graph, positions, font_size=11, font_weight="bold", ax=ax
        )

    origin = graph.graph.get("zone_name") or ""
    ax.set_title(
        f"{title}{' · ' + origin if origin else ''}",
        fontsize=13,
    )
    ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    if show_attribution:
        fig.text(
            0.01, 0.01,
            "© Colaboradores de OpenStreetMap · ODbL",
            fontsize=7, color="#666666",
        )

    fig.tight_layout()
    return fig
