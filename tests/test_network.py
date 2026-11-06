import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pandas as pd
from shapely.geometry import LineString

from core import analysis, criticality
from core.network import (
    adjusted_cost,
    apply_scenario,
    calculate_route,
    classify_edges_by_radius,
    create_demo_graph,
    intersection_name,
    route_metadata,
    route_streets,
)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.graph = create_demo_graph()

    def test_base_route_exists(self):
        result = calculate_route(self.graph, "A", "F")
        self.assertIsNotNone(result)
        self.assertEqual(result.path[0], "A")
        self.assertEqual(result.path[-1], "F")

    def test_blocked_road_changes_or_preserves_valid_route(self):
        scenario = apply_scenario(self.graph, blocked_edges=[("D", "F")])
        result = calculate_route(scenario, "A", "F")
        self.assertIsNotNone(result)
        route_edges = {
            frozenset((result.path[index], result.path[index + 1]))
            for index in range(len(result.path) - 1)
        }
        self.assertNotIn(frozenset(("D", "F")), route_edges)

    def test_total_cut_can_remove_all_routes(self):
        scenario = apply_scenario(
            self.graph,
            blocked_edges=[("D", "F"), ("E", "F")],
        )
        result = calculate_route(scenario, "A", "F")
        self.assertIsNone(result)

    def test_impact_increases_cost_per_abp_formula(self):
        # Fórmula del ABP: we = te * (1 + beta * re), con re ∈ [0, 1].
        self.assertAlmostEqual(adjusted_cost(4.0, 0.0, 1.0), 4.0)
        self.assertAlmostEqual(adjusted_cost(4.0, 1.0, 1.0), 8.0)
        self.assertAlmostEqual(adjusted_cost(4.0, 0.5, 2.0), 8.0)

        baseline = apply_scenario(self.graph, severity=0.0)
        affected = apply_scenario(
            self.graph,
            affected_edges=[("C", "E")],
            severity=1.0,
            beta=1.0,
        )
        base_result = calculate_route(baseline, "A", "F")
        affected_result = calculate_route(affected, "A", "F")
        self.assertIsNotNone(base_result)
        self.assertIsNotNone(affected_result)
        self.assertGreater(affected_result.cost, base_result.cost)
        self.assertNotEqual(affected_result.path, base_result.path)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_scenario(self.graph, severity=1.5)
        with self.assertRaises(ValueError):
            apply_scenario(self.graph, beta=-1.0)

    def test_route_metadata_adds_lengths(self):
        result = calculate_route(self.graph, "A", "F")
        metadata = route_metadata(self.graph, result.path)
        self.assertGreater(metadata["distance_km"], 0.0)
        self.assertGreater(metadata["base_time_min"], 0.0)

    def test_route_streets_groups_names_and_handles_unnamed_segments(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", name="Calle 30", weight_min=1)
        graph.add_edge("B", "C", name=["Calle 30", "Avenida X"], weight_min=1)
        graph.add_edge("C", "D", weight_min=1)
        graph.add_edge("D", "E", weight_min=1)
        self.assertEqual(
            route_streets(graph, ["A", "B", "C", "D", "E"]),
            ["Calle 30", "Vía sin nombre en OpenStreetMap"],
        )

    def test_route_streets_uses_lowest_cost_parallel_road(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", name="Vía alterna", weight_min=5)
        graph.add_edge("A", "B", name="Avenida principal", weight_min=2)
        self.assertEqual(route_streets(graph, ["A", "B"]), ["Avenida principal"])

    def test_intersection_name_uses_connected_street_names(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", name="Calle del Parque")
        graph.add_edge("B", "A", name="Calle del Parque")
        graph.add_edge("B", "C", name=["Avenida Central", "Calle Central"])
        graph.add_edge("C", "B", name=["Avenida Central", "Calle Central"])
        graph.add_edge("B", "D")
        self.assertEqual(
            intersection_name(graph, "B"),
            "Cruce de Avenida Central con Calle del Parque",
        )

    def test_intersection_name_has_readable_fallback_without_street_names(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")
        self.assertEqual(
            intersection_name(graph, "B"),
            "Intersección sin nombre de vía disponible",
        )

    def test_interactive_map_exposes_hover_scenario_preview(self):
        from core.visualization import draw_network_interactive

        graph = nx.MultiDiGraph()
        graph.add_node("A", x=0, y=0)
        graph.add_node("B", x=10, y=0)
        graph.add_edge(
            "A", "B", key=0, weight_min=1,
            name="Avenida <Central>",
            geometry=LineString([(0, 0), (5, 2), (10, 0)]),
        )
        markup = draw_network_interactive(
            graph,
            scenario_centers={"Centro <norte>": "A"},
            preview_scenarios={
                "Centro <norte>": {
                    "blocked_edges": [("A", "B")],
                    "affected_edges": [],
                }
            },
            selected_center="A",
            flood_radius_km=0.4,
            closure_share=50,
        )
        self.assertIn(".scenario-marker:hover .preview-area", markup)
        self.assertIn(".scenario-marker:hover .preview-closure", markup)
        self.assertIn("Centro &lt;norte&gt;", markup)
        self.assertIn('viewBox="0 0 1000 820"', markup)
        self.assertIn('points="44.00,776.00 500.00,593.60 956.00,776.00"', markup)
        self.assertIn('class="preview-road blocked-road"', markup)
        self.assertIn('class="preview-road-halo"', markup)
        self.assertIn("Vía cerrada: Avenida &lt;Central&gt;", markup)
        self.assertIn('target.addEventListener("pointermove"', markup)
        self.assertIn("addEventListener(\"mouseenter\"", markup)
        self.assertIn("zoomAtPointer(event)", markup)
        self.assertIn('id="scenario-map"', markup)

    def test_hover_labels_affected_and_unaffected_roads(self):
        from core.visualization import draw_network_interactive

        graph = nx.DiGraph()
        graph.add_node("A", x=0, y=0)
        graph.add_node("B", x=10, y=0)
        graph.add_node("C", x=20, y=0)
        graph.add_edge("A", "B", name="Calle A")
        graph.add_edge("B", "C", name="Calle B")
        markup = draw_network_interactive(
            graph,
            affected_edges=[("A", "B")],
            scenario_centers={"Centro": "A"},
            preview_scenarios={
                "Centro": {
                    "blocked_edges": [],
                    "affected_edges": [("A", "B")],
                }
            },
        )
        self.assertIn('class="affected-road"', markup)
        self.assertIn('data-tooltip="Vía afectada: Calle A"', markup)
        self.assertIn('class="unaffected-road"', markup)
        self.assertIn('data-tooltip="Vía sin afectación: Calle B"', markup)
        self.assertIn('class="preview-road affected-road"', markup)

    def test_closed_road_hover_exposes_road_name(self):
        from core.visualization import draw_network_interactive

        graph = nx.MultiDiGraph()
        graph.add_node("A", x=0, y=0)
        graph.add_node("B", x=10, y=0)
        graph.add_edge("A", "B", key=0, name="Calle de la Plaza")
        graph.add_edge("B", "A", key=0, name="Calle de la Plaza")
        markup = draw_network_interactive(
            graph,
            blocked_edges=[("A", "B")],
        )
        self.assertIn('class="blocked-road"', markup)
        self.assertIn(
            'data-tooltip="Vía cerrada: Calle de la Plaza"',
            markup,
        )
        self.assertIn('map.querySelector(".map-tooltip")', markup)
        self.assertIn('target.addEventListener("pointerenter"', markup)

    def test_closed_unnamed_road_hover_has_fallback(self):
        from core.visualization import draw_network_interactive

        graph = nx.DiGraph()
        graph.add_node("A", x=0, y=0)
        graph.add_node("B", x=10, y=0)
        graph.add_edge("A", "B")
        markup = draw_network_interactive(graph, blocked_edges=[("A", "B")])
        self.assertIn(
            'data-tooltip="Vía cerrada: Vía sin nombre en OpenStreetMap"',
            markup,
        )

    def test_isolated_node_hover_exposes_intersection_name(self):
        from core.visualization import draw_network_interactive

        graph = nx.MultiDiGraph()
        graph.add_node("A", x=0, y=0)
        graph.add_node("B", x=10, y=0)
        graph.add_edge("A", "B", key=0, name="Calle del Parque")
        graph.add_edge("B", "A", key=0, name="Calle del Parque")
        markup = draw_network_interactive(graph, isolated={"B"})
        self.assertIn(
            'data-tooltip="Intersección: Sobre Calle del Parque"',
            markup,
        )
        self.assertNotIn(
            "<title>Intersección: Sobre Calle del Parque</title>",
            markup,
        )

    def test_radius_uses_geometry_distance_in_projected_meters(self):
        graph = nx.MultiDiGraph()
        graph.add_node("center", x=0, y=0)
        graph.add_node("a", x=-20, y=5)
        graph.add_node("b", x=20, y=5)
        graph.add_node("c", x=-20, y=8)
        graph.add_node("d", x=20, y=8)
        graph.add_edge(
            "a", "b", key=0,
            geometry=LineString([(-20, 5), (0, 5), (20, 5)]),
        )
        graph.add_edge(
            "b", "a", key=0,
            geometry=LineString([(20, 5), (0, 5), (-20, 5)]),
        )
        graph.add_edge(
            "c", "d", key=0,
            geometry=LineString([(-20, 8), (0, 8), (20, 8)]),
        )

        blocked, affected = classify_edges_by_radius(
            graph, "center", radius_m=10, closure_share=50
        )
        self.assertEqual({frozenset(edge) for edge in blocked}, {frozenset(("a", "b"))})
        self.assertEqual({frozenset(edge) for edge in affected}, {frozenset(("c", "d"))})


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.graph = create_demo_graph()

    def test_isolated_nodes_after_total_cut(self):
        scenario = apply_scenario(
            self.graph,
            blocked_edges=[("D", "F"), ("E", "F")],
        )
        isolated = analysis.isolated_nodes(
            self.graph, scenario, safe_points=["F"]
        )
        # Todas las intersecciones quedan sin acceso; el punto seguro F
        # siempre "accede" a sí mismo y no se reporta como aislado.
        self.assertEqual(isolated, {"A", "B", "C", "D", "E"})

    def test_no_isolation_without_closures(self):
        scenario = apply_scenario(self.graph)
        isolated = analysis.isolated_nodes(
            self.graph, scenario, safe_points=["F"]
        )
        self.assertEqual(isolated, set())

    def test_reachability_summary(self):
        summary = analysis.reachability_summary(self.graph, ["F"])
        self.assertEqual(summary, 1.0)

        scenario = apply_scenario(
            self.graph, blocked_edges=[("D", "F"), ("E", "F")]
        )
        summary = analysis.reachability_summary(scenario, ["F"])
        self.assertLess(summary, 1.0)

    def test_structured_indicators_support_multidigraph(self):
        graph = nx.MultiDiGraph(self.graph)
        scenario = apply_scenario(graph, blocked_edges=[("D", "F")])
        indicators = analysis.structured_indicators(
            graph,
            scenario,
            safe_points=["F"],
            blocked_edges=[("D", "F")],
        )
        self.assertEqual(len(indicators["nodes"]), graph.number_of_nodes())

    def test_structured_indicators_count_physical_roads_once(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", base_time_min=1.0, weight_min=1.0)
        graph.add_edge("B", "A", base_time_min=1.0, weight_min=1.0)
        graph.add_edge("B", "C", base_time_min=1.0, weight_min=1.0)
        graph.add_edge("C", "B", base_time_min=1.0, weight_min=1.0)
        scenario = apply_scenario(graph, blocked_edges=[("A", "B")])
        indicators = analysis.structured_indicators(
            graph,
            scenario,
            safe_points=["C"],
            blocked_edges=[("A", "B")],
        )
        by_node = {record["node"]: record for record in indicators["nodes"]}
        self.assertEqual(by_node["A"]["blocked_edges"], 1)
        self.assertEqual(by_node["B"]["blocked_edges"], 1)


class CriticalityTests(unittest.TestCase):
    def setUp(self):
        self.graph = create_demo_graph()

    def _indicators(self, scenario):
        return analysis.structured_indicators(
            base_graph=self.graph,
            scenario_graph=scenario,
            safe_points=["F"],
            blocked_edges=[("D", "F"), ("E", "F")][:2],
            affected_edges=[],
        )

    def test_labels_and_rule_documentation(self):
        scenario = apply_scenario(
            self.graph, blocked_edges=[("D", "F"), ("E", "F")]
        )
        indicators = self._indicators(scenario)
        features = criticality.build_features(indicators)
        self.assertIn("isolated", features.columns)
        labels = set(features["criticality"])
        self.assertLessEqual(labels, {"baja", "media", "alta"})
        # Etiquetar nodos aislados debe resultar en criticidad alta.
        isolated_rows = features[features["isolated"] == 1]
        if not isolated_rows.empty:
            self.assertIn("alta", set(isolated_rows["criticality"]))

    def test_criticality_explanation_matches_rule(self):
        label, reason = criticality.explain_node_criticality(
            {
                "isolated": True,
                "blocked_edges": 1,
                "scenario_degree": 1,
                "affected_edges": 0,
                "betweenness": 0.0,
                "scenario_betweenness": 0.0,
            }
        )
        self.assertEqual(label, "alta")
        self.assertIn("acceso", reason)

    def test_big_change_in_connectivity_raises_isolated_count(self):
        scenario = apply_scenario(self.graph, blocked_edges=[("D", "F"), ("E", "F")])
        indicators = self._indicators(scenario)
        self.assertEqual(indicators["isolated_count"], 5)
        self.assertTrue(
            any(row["isolated"] for row in indicators["nodes"])
        )

    def test_evaluation_holds_out_complete_scenarios(self):
        rows = []
        class_features = {
            "baja": {
                "degree": 3,
                "scenario_degree": 3,
                "betweenness": 0.4,
                "scenario_betweenness": 0.4,
                "affected_edges": 0,
                "blocked_edges": 0,
                "isolated": 0,
            },
            "media": {
                "degree": 3,
                "scenario_degree": 3,
                "betweenness": 0.4,
                "scenario_betweenness": 0.4,
                "affected_edges": 1,
                "blocked_edges": 0,
                "isolated": 0,
            },
            "alta": {
                "degree": 3,
                "scenario_degree": 0,
                "betweenness": 0.4,
                "scenario_betweenness": 0.0,
                "affected_edges": 0,
                "blocked_edges": 2,
                "isolated": 1,
            },
        }
        for scenario_index in range(4):
            for label, values in class_features.items():
                rows.append(
                    {
                        **values,
                        "criticality": label,
                        "scenario": f"scenario-{scenario_index}",
                    }
                )

        metrics = criticality.evaluate_by_scenario(
            pd.DataFrame(rows)
        )
        self.assertEqual(metrics["scenario_count"], 4)
        self.assertEqual(metrics["folds"], 4)
        self.assertEqual(metrics["skipped_folds"], 0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_grouped_evaluation_requires_three_scenarios(self):
        features = criticality.build_features(
            self._indicators(apply_scenario(self.graph))
        )
        features["scenario"] = "only-one"
        self.assertEqual(criticality.evaluate_by_scenario(features), {})


class RealNetworkTests(unittest.TestCase):
    """Pruebas de red real; se omiten si la caché no existe todavía."""

    def setUp(self):
        from core import geodata

        import pickle as _p
        cache = geodata.DATA_DIR / "zone_centro_getsemani.graphpickle"
        if not cache.exists():
            self.skipTest("Caché de red real no descargada aún.")
        with cache.open("rb") as handle:
            self.graph = _p.load(handle)

    def test_real_network_exports_geojson(self):
        from core import geodata

        _, geo = geodata._cache_paths("centro_getsemani")
        self.assertTrue(geo.exists())
        with geo.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["type"], "FeatureCollection")
        self.assertGreater(len(payload["features"]), 0)

    def test_real_network_is_connected_enough(self):
        self.assertGreater(self.graph.number_of_nodes(), 50)
        for _, _, data in list(self.graph.edges(data=True))[:20]:
            self.assertIn("base_time_min", data)
            self.assertGreaterEqual(data["base_time_min"], 0.0)

    def test_real_network_renders(self):
        import matplotlib.pyplot as plt

        from core.visualization import draw_network

        figure = draw_network(self.graph)
        self.assertIsNotNone(figure)
        plt.close(figure)

    def test_geojson_cache_rebuilds_graph_without_download(self):
        from core import geodata

        zone = geodata.PILOT_ZONES["manga"]
        graph = nx.MultiDiGraph()
        graph.graph.update(
            zone=zone.key,
            zone_name=zone.name,
            downloaded_utc="2026-01-01 00:00 UTC",
        )
        graph.add_node(1, x=-75.5, y=10.4)
        graph.add_node(2, x=-75.4, y=10.5)
        graph.add_edge(1, 2, key=0, base_time_min=3.0, weight_min=3.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(geodata, "DATA_DIR", Path(temporary_directory)):
                geodata._export_geojson(graph, zone)
                with patch.object(
                    geodata,
                    "_download_overpass_response",
                    side_effect=AssertionError("No debería descargar la red"),
                ):
                    rebuilt = geodata.load_zone_graph(zone.key)

                self.assertEqual(rebuilt.number_of_nodes(), 2)
                self.assertEqual(rebuilt.number_of_edges(), 1)
                self.assertEqual(rebuilt.graph["zone_name"], zone.name)
                self.assertEqual(
                    rebuilt[1][2][0]["base_time_min"],
                    3.0,
                )
                self.assertTrue(geodata._cache_paths(zone.key)[0].exists())


if __name__ == "__main__":
    unittest.main()
