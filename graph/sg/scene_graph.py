"""The SceneGraph container: nodes + edges, plus the traversal/query surface the
rest of the pipeline builds on.

This is where the "graph architecture" earns its keep: instead of iterating a
list, downstream stages ask the graph questions —
  * `ordered_beats()`      — follow NEXT edges to get narrative order
  * `entities_in(beat)`     — resolve APPEARS_IN so a prompt can re-describe the
                              exact same character every cut
  * `recurring_entities()`  — entities in ≥2 beats → need a consistency anchor
  * `predecessor(beat)` / `is_continuation(beat)` — CONTINUES_FROM chaining
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel

from .model import Beat, Concept, Edge, EdgeType, Entity, Location, Node, Style


class SceneGraph(BaseModel):
    nodes: List[Node] = []
    edges: List[Edge] = []

    # --- mutation ----------------------------------------------------------
    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str, etype: EdgeType, **attrs: str) -> Edge:
        edge = Edge(source=source, target=target, type=etype,
                    attrs={k: str(v) for k, v in attrs.items()})
        # de-dupe identical edges
        if not any(e.key() == edge.key() for e in self.edges):
            self.edges.append(edge)
        return edge

    # --- indexing (runtime, not persisted) ---------------------------------
    def _index(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def node(self, node_id: str) -> Optional[Node]:
        return self._index().get(node_id)

    def of_kind(self, kind: str) -> List[Node]:
        return [n for n in self.nodes if n.kind == kind]

    def concept(self) -> Optional[Concept]:
        cs = self.of_kind("concept")
        return cs[0] if cs else None

    def style(self) -> Optional[Style]:
        ss = self.of_kind("style")
        return ss[0] if ss else None

    def entities(self) -> List[Entity]:
        return self.of_kind("entity")  # type: ignore[return-value]

    def locations(self) -> List[Location]:
        return self.of_kind("location")  # type: ignore[return-value]

    # --- edge queries ------------------------------------------------------
    def out_edges(self, node_id: str, etype: Optional[EdgeType] = None) -> List[Edge]:
        return [e for e in self.edges
                if e.source == node_id and (etype is None or e.type == etype)]

    def in_edges(self, node_id: str, etype: Optional[EdgeType] = None) -> List[Edge]:
        return [e for e in self.edges
                if e.target == node_id and (etype is None or e.type == etype)]

    # --- beat-oriented traversal ------------------------------------------
    def ordered_beats(self) -> List[Beat]:
        """Beats in narrative order. Follows NEXT edges if present; otherwise
        falls back to `beat_no`. Robust to missing/partial NEXT chains."""
        beats: List[Beat] = self.of_kind("beat")  # type: ignore[assignment]
        if not beats:
            return []
        by_id = {b.id: b for b in beats}
        succ = {e.source: e.target for e in self.edges
                if e.type == EdgeType.NEXT and e.target in by_id}
        targets = set(succ.values())
        heads = [b for b in beats if b.id not in targets]  # no incoming NEXT
        if len(heads) != 1 or len(succ) < len(beats) - 1:
            return sorted(beats, key=lambda b: b.beat_no)  # incomplete chain → fallback
        order: List[Beat] = []
        cur = heads[0].id
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            order.append(by_id[cur])
            cur = succ.get(cur)
        # append any strays (defensive)
        for b in sorted(beats, key=lambda b: b.beat_no):
            if b.id not in seen:
                order.append(b)
        return order

    def predecessor(self, beat_id: str) -> Optional[Beat]:
        """The beat this one continues from (CONTINUES_FROM), if any."""
        for e in self.out_edges(beat_id, EdgeType.CONTINUES_FROM):
            n = self.node(e.target)
            if isinstance(n, Beat):
                return n
        return None

    def entities_in(self, beat_id: str) -> List[Entity]:
        ids = {e.source for e in self.in_edges(beat_id, EdgeType.APPEARS_IN)}
        return [n for n in self.entities() if n.id in ids]

    def location_of(self, beat_id: str) -> Optional[Location]:
        for e in self.out_edges(beat_id, EdgeType.LOCATED_IN):
            n = self.node(e.target)
            if isinstance(n, Location):
                return n
        return None

    def appearance_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            if e.type == EdgeType.APPEARS_IN:
                counts[e.source] += 1
        return dict(counts)

    def recurring_entities(self, threshold: int = 2) -> List[Entity]:
        counts = self.appearance_counts()
        return [e for e in self.entities() if counts.get(e.id, 0) >= threshold]

    # --- integrity ---------------------------------------------------------
    def validate_graph(self) -> List[str]:
        """Return a list of structural problems (empty == healthy)."""
        problems: List[str] = []
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            problems.append("duplicate node ids")
        for e in self.edges:
            if e.source not in ids:
                problems.append(f"edge {e.type} has unknown source {e.source}")
            if e.target not in ids:
                problems.append(f"edge {e.type} has unknown target {e.target}")
        if not self.concept():
            problems.append("no Concept node")
        beats = self.of_kind("beat")
        if not beats:
            problems.append("no Beat nodes")
        nums = sorted(b.beat_no for b in beats)  # type: ignore[attr-defined]
        if nums and nums != list(range(1, len(nums) + 1)):
            problems.append(f"beat_no not 1..N contiguous: {nums}")
        return problems
