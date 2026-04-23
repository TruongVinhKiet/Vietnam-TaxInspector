"""
graph_intelligence.py – Graph Intelligence 2.0 (Program B)
============================================================
Extends the existing GNN model with:
    1. Motif Detection – identify suspicious transaction patterns (triangles, stars, chains)
    2. Link Prediction – predict likely future fraudulent connections
    3. Ownership Graph – detect shell company networks via ownership links
    4. Ring Scoring – score circular transaction rings by complexity and amount

Designed for 12GB RAM: uses NetworkX for motif analysis (no GPU needed),
integrates with existing GATv2-based GNN for scoring.
"""

import math
import os
import numpy as np
import networkx as nx
from datetime import date, timedelta
from typing import Optional
from collections import defaultdict


class MotifDetector:
    """
    Detect suspicious transaction motifs in the invoice graph.
    Motifs: triangles (3-cycles), stars (hub-spoke), chains (A→B→C→...),
    fan-out (1 seller → many buyers), fan-in (many sellers → 1 buyer).
    """

    def __init__(self, min_triangle_amount: float = 50_000_000):
        self.min_triangle_amount = min_triangle_amount  # 50M VND minimum for alerting

    def detect_all(self, companies: list[dict], invoices: list[dict]) -> dict:
        """Run all motif detections and return categorized results."""
        G = self._build_directed_graph(invoices)

        triangles = self._detect_triangles(G, invoices)
        stars = self._detect_star_patterns(G)
        chains = self._detect_long_chains(G)
        fan_out = self._detect_fan_out(G)
        fan_in = self._detect_fan_in(G)

        # Aggregate into company-level motif scores
        company_motif_scores = self._aggregate_scores(
            G, companies, triangles, stars, chains, fan_out, fan_in
        )

        return {
            "motifs": {
                "triangles": triangles,
                "stars": stars,
                "chains": chains,
                "fan_out": fan_out,
                "fan_in": fan_in,
            },
            "summary": {
                "total_triangles": len(triangles),
                "total_stars": len(stars),
                "total_chains": len(chains),
                "total_fan_out": len(fan_out),
                "total_fan_in": len(fan_in),
            },
            "company_motif_scores": company_motif_scores,
        }

    def _build_directed_graph(self, invoices: list[dict]) -> nx.DiGraph:
        G = nx.DiGraph()
        for inv in invoices:
            s = inv.get("seller_tax_code", inv.get("from", ""))
            b = inv.get("buyer_tax_code", inv.get("to", ""))
            amount = float(inv.get("amount", 0))
            if s and b:
                if G.has_edge(s, b):
                    G[s][b]["total_amount"] += amount
                    G[s][b]["invoice_count"] += 1
                else:
                    G.add_edge(s, b, total_amount=amount, invoice_count=1)
        return G

    def _detect_triangles(self, G: nx.DiGraph, invoices: list[dict]) -> list[dict]:
        """Detect directed triangles (A→B→C→A) – classic carousel fraud pattern."""
        triangles = []
        undirected = G.to_undirected()

        # Find triangles in undirected, then check directionality
        for clique in nx.enumerate_all_cliques(undirected):
            if len(clique) < 3:
                continue
            if len(clique) > 3:
                break  # Only interested in triangles

            a, b, c = clique
            # Check if it forms a directed cycle
            cycles = [
                [a, b, c],  # a→b→c→a
                [a, c, b],  # a→c→b→a
            ]
            for cycle in cycles:
                if (G.has_edge(cycle[0], cycle[1]) and
                    G.has_edge(cycle[1], cycle[2]) and
                    G.has_edge(cycle[2], cycle[0])):

                    total_amount = sum(
                        G[cycle[i]][cycle[(i+1) % 3]].get("total_amount", 0)
                        for i in range(3)
                    )

                    if total_amount >= self.min_triangle_amount:
                        triangles.append({
                            "nodes": cycle,
                            "total_amount": round(total_amount, 0),
                            "type": "directed_triangle",
                            "risk_level": "critical" if total_amount > 500_000_000 else "high",
                        })

            if len(triangles) >= 50:  # Cap for memory
                break

        return triangles

    def _detect_star_patterns(self, G: nx.DiGraph, min_degree: int = 5) -> list[dict]:
        """Detect star patterns – one hub company with many connections (potential shell)."""
        stars = []
        for node in G.nodes():
            out_deg = G.out_degree(node)
            in_deg = G.in_degree(node)

            if out_deg >= min_degree:
                targets = list(G.successors(node))
                total_out = sum(G[node][t].get("total_amount", 0) for t in targets)
                stars.append({
                    "hub": node,
                    "type": "fan_out_star",
                    "spoke_count": out_deg,
                    "total_amount": round(total_out, 0),
                    "risk_level": "high" if out_deg >= 10 else "medium",
                })

            if in_deg >= min_degree:
                sources = list(G.predecessors(node))
                total_in = sum(G[s][node].get("total_amount", 0) for s in sources)
                stars.append({
                    "hub": node,
                    "type": "fan_in_star",
                    "spoke_count": in_deg,
                    "total_amount": round(total_in, 0),
                    "risk_level": "high" if in_deg >= 10 else "medium",
                })

        return stars[:30]

    def _detect_long_chains(self, G: nx.DiGraph, min_length: int = 4) -> list[dict]:
        """Detect long linear chains (A→B→C→D→...) – layering pattern."""
        chains = []
        visited = set()

        for node in G.nodes():
            if node in visited:
                continue
            if G.in_degree(node) == 0 or G.in_degree(node) == 1:
                # Start BFS from potential chain head
                chain = [node]
                current = node
                while True:
                    successors = [s for s in G.successors(current) if s not in chain]
                    if len(successors) != 1:
                        break
                    current = successors[0]
                    chain.append(current)
                    if len(chain) > 10:
                        break

                if len(chain) >= min_length:
                    total_amount = sum(
                        G[chain[i]][chain[i+1]].get("total_amount", 0)
                        for i in range(len(chain)-1)
                        if G.has_edge(chain[i], chain[i+1])
                    )
                    chains.append({
                        "nodes": chain,
                        "length": len(chain),
                        "total_amount": round(total_amount, 0),
                        "type": "linear_chain",
                        "risk_level": "high" if len(chain) >= 6 else "medium",
                    })
                    visited.update(chain)

        return chains[:20]

    def _detect_fan_out(self, G: nx.DiGraph, threshold: int = 8) -> list[dict]:
        """Detect extreme fan-out patterns (1 company sells to many)."""
        results = []
        for node in G.nodes():
            out_deg = G.out_degree(node)
            if out_deg >= threshold:
                targets = list(G.successors(node))
                amounts = [G[node][t].get("total_amount", 0) for t in targets]
                cv = float(np.std(amounts) / max(np.mean(amounts), 1))

                results.append({
                    "source": node,
                    "target_count": out_deg,
                    "total_amount": round(sum(amounts), 0),
                    "amount_cv": round(cv, 4),
                    "uniform_distribution": cv < 0.3,
                    "type": "fan_out",
                    "risk_level": "high" if cv < 0.3 and out_deg >= 10 else "medium",
                })

        return sorted(results, key=lambda x: x["target_count"], reverse=True)[:15]

    def _detect_fan_in(self, G: nx.DiGraph, threshold: int = 8) -> list[dict]:
        """Detect extreme fan-in patterns (many companies sell to 1)."""
        results = []
        for node in G.nodes():
            in_deg = G.in_degree(node)
            if in_deg >= threshold:
                sources = list(G.predecessors(node))
                amounts = [G[s][node].get("total_amount", 0) for s in sources]
                cv = float(np.std(amounts) / max(np.mean(amounts), 1))

                results.append({
                    "target": node,
                    "source_count": in_deg,
                    "total_amount": round(sum(amounts), 0),
                    "amount_cv": round(cv, 4),
                    "uniform_distribution": cv < 0.3,
                    "type": "fan_in",
                    "risk_level": "high" if cv < 0.3 and in_deg >= 10 else "medium",
                })

        return sorted(results, key=lambda x: x["source_count"], reverse=True)[:15]

    def _aggregate_scores(self, G, companies, triangles, stars, chains, fan_out, fan_in):
        """Compute per-company motif risk score aggregation."""
        scores = {}
        for c in companies:
            tc = c.get("tax_code", "")
            score = 0.0

            # Triangle participation
            tri_count = sum(1 for t in triangles if tc in t.get("nodes", []))
            score += min(tri_count * 0.2, 0.6)

            # Star hub
            star_count = sum(1 for s in stars if s.get("hub") == tc)
            score += min(star_count * 0.15, 0.4)

            # Chain participation
            chain_count = sum(1 for ch in chains if tc in ch.get("nodes", []))
            score += min(chain_count * 0.1, 0.3)

            # Fan patterns
            fo_match = [f for f in fan_out if f.get("source") == tc]
            fi_match = [f for f in fan_in if f.get("target") == tc]
            if fo_match:
                score += 0.15
            if fi_match:
                score += 0.15

            scores[tc] = round(min(1.0, score), 4)

        return scores


class LinkPredictor:
    """
    Predict likely future fraudulent connections using graph topology features.
    Uses Jaccard coefficient, Adamic-Adar, and common neighbors as predictors.
    No GPU needed – pure NetworkX computation.
    """

    def predict_new_links(
        self,
        companies: list[dict],
        invoices: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """Predict company pairs likely to form new (possibly fraudulent) connections."""
        G = nx.DiGraph()
        for inv in invoices:
            s = inv.get("seller_tax_code", inv.get("from", ""))
            b = inv.get("buyer_tax_code", inv.get("to", ""))
            if s and b:
                G.add_edge(s, b, amount=float(inv.get("amount", 0)))

        if G.number_of_nodes() < 3:
            return []

        U = G.to_undirected()
        predictions = []

        non_edges = list(nx.non_edges(U))
        if len(non_edges) > 5000:
            # Sample for memory efficiency
            indices = np.random.choice(len(non_edges), 5000, replace=False)
            non_edges = [non_edges[i] for i in indices]

        jaccard_preds = nx.jaccard_coefficient(U, non_edges)
        adamic_preds = {
            (u, v): float(score)
            for u, v, score in nx.adamic_adar_index(U, non_edges)
        }

        for u, v, jaccard_score in jaccard_preds:
            if jaccard_score <= 0:
                continue

            adamic_score = adamic_preds.get((u, v), adamic_preds.get((v, u), 0.0))

            # Combined score
            combined = jaccard_score * 0.5 + min(adamic_score / 10, 0.5) * 0.5

            # Check if both nodes are "risky"
            u_out = G.out_degree(u) if G.has_node(u) else 0
            v_in = G.in_degree(v) if G.has_node(v) else 0
            topology_risk = min(1.0, (u_out + v_in) / 20)

            final_score = combined * 0.6 + topology_risk * 0.4

            if final_score > 0.1:
                predictions.append({
                    "source": u,
                    "target": v,
                    "prediction_score": round(final_score, 4),
                    "jaccard": round(jaccard_score, 4),
                    "adamic_adar": round(adamic_score, 4),
                    "common_neighbors": len(list(nx.common_neighbors(U, u, v))),
                    "risk_level": "high" if final_score > 0.5 else "medium" if final_score > 0.3 else "low",
                })

        predictions.sort(key=lambda x: x["prediction_score"], reverse=True)
        return predictions[:top_k]


class OwnershipGraphAnalyzer:
    """
    Analyze company ownership relationships to detect shell company networks.
    Uses ownership_links data to find:
    - Common controllers (same person owns multiple companies)
    - Ownership chains (A owns B owns C)
    - Cross-ownership with invoice connections (ownership + trade = high risk)
    """

    def analyze(
        self,
        ownership_links: list[dict],
        invoices: list[dict],
    ) -> dict:
        """Full ownership analysis."""
        if not ownership_links:
            return {
                "status": "no_data",
                "message": "Chưa có dữ liệu quan hệ sở hữu.",
                "clusters": [],
                "common_controllers": [],
                "cross_ownership_trades": [],
            }

        # Build ownership graph
        O = nx.DiGraph()
        person_companies = defaultdict(set)

        for link in ownership_links:
            parent = link.get("parent_tax_code", "")
            child = link.get("child_tax_code", "")
            pct = float(link.get("ownership_percent", 0))
            rel_type = link.get("relationship_type", "shareholder")
            person = link.get("person_id") or link.get("person_name", "")

            if parent and child:
                O.add_edge(parent, child, ownership_percent=pct, relationship_type=rel_type)

            if person:
                person_companies[person].add(parent)
                person_companies[person].add(child)

        # Detect common controllers
        common_controllers = []
        for person, companies in person_companies.items():
            if len(companies) >= 2:
                common_controllers.append({
                    "person_id": person,
                    "company_count": len(companies),
                    "companies": list(companies),
                    "risk_level": "high" if len(companies) >= 3 else "medium",
                })

        common_controllers.sort(key=lambda x: x["company_count"], reverse=True)

        # Detect ownership chains
        clusters = []
        for component in nx.weakly_connected_components(O):
            if len(component) >= 2:
                subgraph = O.subgraph(component)
                total_ownership = sum(
                    subgraph[u][v].get("ownership_percent", 0)
                    for u, v in subgraph.edges()
                )
                clusters.append({
                    "companies": list(component),
                    "size": len(component),
                    "total_edges": subgraph.number_of_edges(),
                    "avg_ownership": round(total_ownership / max(1, subgraph.number_of_edges()), 1),
                    "risk_level": "high" if len(component) >= 4 else "medium",
                })

        clusters.sort(key=lambda x: x["size"], reverse=True)

        # Cross-ownership trades: find invoice connections between owned companies
        trade_graph = set()
        for inv in invoices:
            s = inv.get("seller_tax_code", inv.get("from", ""))
            b = inv.get("buyer_tax_code", inv.get("to", ""))
            if s and b:
                trade_graph.add((s, b))

        cross_trades = []
        for parent, child in O.edges():
            if (parent, child) in trade_graph or (child, parent) in trade_graph:
                cross_trades.append({
                    "parent": parent,
                    "child": child,
                    "ownership_percent": O[parent][child].get("ownership_percent", 0),
                    "trade_direction": "parent→child" if (parent, child) in trade_graph else "child→parent",
                    "risk_level": "critical",
                    "reason": "Giao dịch giữa công ty có quan hệ sở hữu (related-party transaction)",
                })

        return {
            "status": "analyzed",
            "clusters": clusters[:20],
            "common_controllers": common_controllers[:20],
            "cross_ownership_trades": cross_trades[:50],
            "summary": {
                "total_ownership_links": len(ownership_links),
                "total_clusters": len(clusters),
                "total_common_controllers": len(common_controllers),
                "total_cross_trades": len(cross_trades),
            },
        }


class RingScorer:
    """Score circular transaction rings by economic implausibility."""

    def score_rings(self, cycles: list[list], invoices: list[dict]) -> list[dict]:
        """Score each detected ring with multi-factor analysis."""
        if not cycles:
            return []

        max_rings_output = max(1, min(500, int(os.getenv("RING_SCORING_MAX_OUTPUT", "150"))))

        # Build edge lookup
        edge_amounts = {}
        edge_dates = {}
        for inv in invoices:
            s = inv.get("seller_tax_code", inv.get("from", ""))
            b = inv.get("buyer_tax_code", inv.get("to", ""))
            key = (s, b)
            edge_amounts[key] = edge_amounts.get(key, 0) + float(inv.get("amount", 0))
            d = inv.get("date", "")
            if d:
                edge_dates.setdefault(key, []).append(str(d))

        scored_rings = []
        for cycle in cycles[:max_rings_output]:
            ring_amount = 0
            time_span_days = None
            edge_count = len(cycle)
            ring_start_date = None
            ring_end_date = None

            for i in range(len(cycle)):
                src = cycle[i]
                dst = cycle[(i + 1) % len(cycle)]
                ring_amount += edge_amounts.get((src, dst), 0)

                dates = edge_dates.get((src, dst), [])
                if dates:
                    sorted_dates = sorted(dates)
                    try:
                        first = date.fromisoformat(sorted_dates[0])
                        last = date.fromisoformat(sorted_dates[-1])
                        if ring_start_date is None or first < ring_start_date:
                            ring_start_date = first
                        if ring_end_date is None or last > ring_end_date:
                            ring_end_date = last
                    except (ValueError, TypeError):
                        pass

            if ring_start_date is not None and ring_end_date is not None:
                # Keep ring timing semantics consistent across backend/UI:
                # span reflects the full observation window of the ring.
                time_span_days = max(0, (ring_end_date - ring_start_date).days)

            # Scoring factors
            amount_score = min(1.0, ring_amount / 5_000_000_000)  # 5B VND ceiling
            speed_score = 0.0
            if time_span_days is not None:
                speed_score = max(0, 1.0 - time_span_days / 90)  # Fast = high risk

            complexity_score = min(1.0, edge_count / 8)

            # Composite score
            ring_score = round(
                amount_score * 0.4 + speed_score * 0.35 + complexity_score * 0.25, 4
            )

            scored_rings.append({
                "nodes": cycle,
                "ring_size": edge_count,
                "total_amount": round(ring_amount, 0),
                "time_span_days": time_span_days,
                "start_date": ring_start_date.isoformat() if ring_start_date else None,
                "end_date": ring_end_date.isoformat() if ring_end_date else None,
                "span_method": "global_ring_window",
                "ring_score": ring_score,
                "amount_factor": round(amount_score, 4),
                "speed_factor": round(speed_score, 4),
                "complexity_factor": round(complexity_score, 4),
                "risk_level": "critical" if ring_score > 0.7 else "high" if ring_score > 0.4 else "medium",
            })

        scored_rings.sort(key=lambda x: x["ring_score"], reverse=True)
        return scored_rings
