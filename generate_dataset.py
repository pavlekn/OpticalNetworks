import argparse
import json
import math
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable

import pandas as pd

from config.consts import NetDir, DistDir
from config.paths import DATA_SETS_DIR


class Sha256Prng:
    def __init__(self, seed_bytes: bytes, label: bytes = b"ON_GEN"):
        self._seed = hashlib.sha256(seed_bytes).digest()
        self._label = label
        self._counter = 0
        self._buffer = b""

    def _refill(self, min_bytes: int = 32) -> None:
        while len(self._buffer) < min_bytes:
            h = hashlib.sha256(self._seed + self._label + self._counter.to_bytes(16, 'big')).digest()
            self._counter += 1
            self._buffer += h

    def randbytes(self, n: int) -> bytes:
        self._refill(n)
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def _rand_uint64(self) -> int:
        return int.from_bytes(self.randbytes(8), 'big')

    def random(self) -> float:
        # 53-bit precision uniform in [0,1)
        x = self._rand_uint64() >> 11  # keep top 53 bits
        return x / (1 << 53)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def randbelow(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be > 0")
        # rejection sampling to avoid modulo bias
        t = (-n) % n
        while True:
            r = self._rand_uint64()
            if r >= t:
                return r % n

    def randrange(self, start: int, stop: 'int | None' = None) -> int:
        if stop is None:
            stop = start
            start = 0
        if stop <= start:
            raise ValueError("empty range for randrange()")
        return start + self.randbelow(stop - start)

    def randint(self, a: int, b: int) -> int:
        return a + self.randbelow(b - a + 1)

    def choice(self, seq: Iterable):
        seq = list(seq)
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[self.randrange(len(seq))]

    def sample(self, population: Iterable, k: int) -> List:
        pop = list(population)
        if k > len(pop):
            raise ValueError("Sample larger than population")
        # Floyd's algorithm
        selected = set()
        result = []
        n = len(pop)
        for i in range(n - k, n):
            t = self.randrange(i + 1)
            if t in selected:
                t = i
            selected.add(t)
            result.append(pop[t])
        return result

    def shuffle(self, x: List) -> None:
        # Fisher–Yates
        for i in range(len(x) - 1, 0, -1):
            j = self.randrange(i + 1)
            x[i], x[j] = x[j], x[i]


@dataclass
class LinkSpec:
    link_id: int
    source_id: int
    destination_id: int
    distance: float
    width: int


@dataclass
class DemandSpec:
    path_id: int
    source_id: int
    destination_id: int
    width: int


def _ensure_dir(lightpath: Path) -> None:
    lightpath.mkdir(parents=True, exist_ok=True)


def _euclid_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _generate_connected_graph(num_nodes: int, target_degree: int, total_colors: int,
                              field_w: float, field_h: float, rnd) -> Tuple[List[LinkSpec], Dict[int, Tuple[float, float]]]:
    positions: Dict[int, Tuple[float, float]] = {i: (rnd.uniform(0, field_w), rnd.uniform(0, field_h)) for i in range(num_nodes)}

    # Start with a random spanning tree to ensure connectivity
    nodes = list(range(num_nodes))
    rnd.shuffle(nodes)
    parent = {nodes[0]: None}
    undirected_edges: Set[Tuple[int, int]] = set()

    for v in nodes[1:]:
        u = rnd.choice(list(parent.keys()))
        a, b = min(u, v), max(u, v)
        undirected_edges.add((a, b))
        parent[v] = u

    # Add extra edges until average degree ~ target_degree
    desired_edges = max(num_nodes - 1, (num_nodes * target_degree) // 2)
    attempts = 0
    while len(undirected_edges) < desired_edges and attempts < num_nodes * num_nodes:
        u, v = rnd.sample(range(num_nodes), 2)
        a, b = min(u, v), max(u, v)
        if a == b or (a, b) in undirected_edges:
            attempts += 1
            continue
        undirected_edges.add((a, b))

    # Create a single record per undirected edge in links.csv (source_id < destination_id)
    links: List[LinkSpec] = []
    link_id = 0
    for a, b in sorted(undirected_edges):
        dist = _euclid_distance(positions[a], positions[b])
        cap = total_colors
        s, t = (a, b) if a < b else (b, a)
        links.append(LinkSpec(link_id, s, t, round(dist, 2), cap))
        link_id += 1

    return links, positions


def _find_path_with_distance_bound(neighbors: Dict[int, List[int]], pair_distance: Dict[Tuple[int, int], float],
                                   src: int, dst: int, max_total_distance: float, rnd):
    # Randomized DFS with pruning by cumulative distance
    best: List[int] = []
    # seen is not strictly needed here since we avoid cycles via path_nodes membership

    def dfs(u: int, path_nodes: List[int], dist_so_far: float) -> bool:
        nonlocal best
        if dist_so_far > max_total_distance:
            return False
        if u == dst:
            best = list(path_nodes)
            return True
        nbrs = list(neighbors.get(u, []))
        rnd.shuffle(nbrs)
        for v in nbrs:
            edge = (u, v)
            if v in path_nodes:
                continue
            step = pair_distance.get(edge, float('inf'))
            if dfs(v, path_nodes + [v], dist_so_far + step):
                return True
        return False

    return best if dfs(src, [src], 0.0) else []


def _build_successors(links: List[LinkSpec]) -> Dict[int, List[int]]:
    # Deprecated: kept for backward compatibility in other modules
    succ: Dict[int, List[int]] = {}
    for link in links:
        succ.setdefault(link.source_id, []).append(link.destination_id)
    return succ


def _path_to_links(path_nodes: List[int], links_by_pair: Dict[Tuple[int, int], int]) -> List[int]:
    link_ids: List[int] = []
    for u, v in zip(path_nodes, path_nodes[1:]):
        link_ids.append(links_by_pair[(u, v)])
    return link_ids


def _assign_first_color_for_lightpath(link_ids: List[int], link_widths: Dict[int, int], used: Dict[Tuple[int, int], bool], width: int):
    # Greedy: choose the smallest first_color such that [first_color, first_color+width) is free on all links
    if not link_ids:
        return None
    cap = min(link_widths[lid] for lid in link_ids)
    if width > cap:
        return None
    for first_color in range(cap - width + 1):
        ok = True
        for lid in link_ids:
            for delta in range(width):
                if used.get((lid, first_color + delta), False):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for lid in link_ids:
                for delta in range(width):
                    used[(lid, first_color + delta)] = True
            return first_color
    return None


def generate_dataset(out_dir: Path, num_nodes: int, num_demands: int, target_degree: int,
                     total_colors: int, max_total_distance: float, seed=None, modulus=None,
                     p1: float = 0.85, p2: float = 0.075, p4: float = 0.075,
                     field_w: float = 1000.0, field_h: float = 1000.0):
    if modulus is not None:
        rnd = Sha256Prng(str(modulus).encode('utf-8'))
        prng_kind = "sha256"
        prng_seed = hashlib.sha256(str(modulus).encode('utf-8')).hexdigest()
    else:
        rnd = random.Random(seed)
        prng_kind = "python_random"
        prng_seed = seed

    # sanitize and normalize width probabilities
    probs = [max(0.0, float(p1)), max(0.0, float(p2)), max(0.0, float(p4))]
    ssum = probs[0] + probs[1] + probs[2]
    if ssum <= 0:
        probs = [0.85, 0.075, 0.075]
        ssum = 1.0
    probs = [x / ssum for x in probs]

    links, positions = _generate_connected_graph(num_nodes, target_degree, total_colors, field_w, field_h, rnd)
    # Build undirected neighbors and pair->distance map (both directions)
    neighbors: Dict[int, List[int]] = {}
    pair_distance: Dict[Tuple[int, int], float] = {}
    for link in links:
        s, t, dist = link.source_id, link.destination_id, link.distance
        pair_distance[(s, t)] = dist
        pair_distance[(t, s)] = dist
        neighbors.setdefault(s, []).append(t)
        neighbors.setdefault(t, []).append(s)
    links_by_pair: Dict[Tuple[int, int], int] = {}
    link_widths: Dict[int, int] = {}
    for link in links:
        s, t = link.source_id, link.destination_id
        links_by_pair[(s, t)] = link.link_id
        links_by_pair[(t, s)] = link.link_id
        link_widths[link.link_id] = link.width

    # Demands: pick random src/dst, distance-bounded lightpath search, width sampled by (p1,p2,p4)
    demands: List[DemandSpec] = []
    routing_rows: List[Tuple[int, int, int]] = []  # path_id, link_id, first_color
    used_link_color: Dict[Tuple[int, int], bool] = {}

    path_id = 0
    attempts = 0
    while path_id < num_demands and attempts < num_demands * 10:
        attempts += 1
        src, dst = rnd.sample(range(num_nodes), 2)
        path_nodes = _find_path_with_distance_bound(neighbors, pair_distance, src, dst, max_total_distance, rnd)
        if not path_nodes:
            continue
        link_ids = _path_to_links(path_nodes, links_by_pair)
        # Sample width according to probabilities (deterministic via rnd)
        r = rnd.random()
        if r < probs[0]:
            width = 1
        elif r < probs[0] + probs[1]:
            width = 2
        else:
            width = 4
        first_color = _assign_first_color_for_lightpath(link_ids, link_widths, used_link_color, width)
        if first_color is None:
            continue
        demands.append(DemandSpec(path_id, src, dst, width))
        for lid in link_ids:
            routing_rows.append((path_id, lid, first_color))
        path_id += 1

    # link service statistics
    total_services: Dict[int, int] = {link.link_id: 0 for link in links}
    one_slot: Dict[int, int] = {link.link_id: 0 for link in links}
    two_slot: Dict[int, int] = {link.link_id: 0 for link in links}
    four_slot: Dict[int, int] = {link.link_id: 0 for link in links}

    for d in demands:
        d_link_ids = [lid for pid, lid, _ in routing_rows if pid == d.path_id]
        for lid in d_link_ids:
            total_services[lid] += 1
            if d.width == 1:
                one_slot[lid] += 1
            elif d.width == 2:
                two_slot[lid] += 1
            elif d.width == 4:
                four_slot[lid] += 1

    # Write outputs
    _ensure_dir(out_dir)

    # links.csv
    links_df = pd.DataFrame([{"link_id": l.link_id, "source_id": l.source_id, "destination_id": l.destination_id,
                              "distance": l.distance, "width": l.width} for l in links])
    links_df.set_index("link_id", inplace=True)
    links_df.to_csv(out_dir / "links.csv")

    # demands.csv
    demands_df = pd.DataFrame([{"path_id": d.path_id, "source_id": d.source_id,
                                "destination_id": d.destination_id, "width": d.width} for d in demands])
    demands_df.set_index("path_id", inplace=True)
    demands_df.to_csv(out_dir / "demands.csv")

    # routing.csv
    routing_df = pd.DataFrame(routing_rows, columns=["path_id", "link_id", "first_color"]).set_index("path_id")
    routing_df.to_csv(out_dir / "routing.csv")

    # linkservicestatistics.csv
    lss_rows = []
    for link in links:
        lid = link.link_id
        lss_rows.append({
            "link_id": lid,
            "total_services": total_services[lid],
            "1_slot_services": one_slot[lid],
            "2_slot_services": two_slot[lid],
            "4_slot_services": four_slot[lid],
        })
    lss_df = pd.DataFrame(lss_rows).set_index("link_id")
    lss_df.to_csv(out_dir / "linkservicestatistics.csv")

    # nodes.csv (store coordinates for visualization)
    try:
        import pandas as _pd
        _pd.DataFrame([{"node_id": nid, "x": positions[nid][0], "y": positions[nid][1]} for nid in sorted(positions.keys())]) \
            .set_index("node_id").to_csv(out_dir / "nodes.csv")
    except Exception:
        pass

    # maxW.txt — use the configured reachability bound so all generated lightpaths satisfy it
    max_dist = float(max_total_distance)
    with open(out_dir / "maxW.txt", "w") as f:
        f.write(f"MaxDistanceReachability = {max_dist}\n")

    # provenance.json
    provenance = {
        "generator": "optical_network_dataset_generator",
        "prng": prng_kind,
        "seed": prng_seed,
        "modulus": str(modulus) if modulus is not None else None,
        "params": {
            "nodes": num_nodes,
            "demands": num_demands,
            "degree": target_degree,
            "colors": total_colors,
            "reach": max_total_distance,
            "field_w": field_w,
            "field_h": field_h,
            "width_probs": {"1": probs[0], "2": probs[1], "4": probs[2]},
        },
        "counts": {
            "links": int(len(links_df)),
            "demands": int(len(demands_df)),
        }
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # subset_sum.json (optional artifact demonstrating a reduction-style instance derived from modulus)
    if modulus is not None:
        # Derive reproducible item list from modulus
        k_items = min(64, max(16, num_demands))
        items: List[int] = []
        for i in range(k_items):
            h = hashlib.sha256(str(modulus).encode('utf-8') + b"|item|" + str(i).encode('utf-8')).digest()
            # 56-bit positive integers to keep sums in range
            items.append(int.from_bytes(h[:7], 'big') | 1)
        # Choose a random subset and compute target using PRNG
        prng = Sha256Prng(str(modulus).encode('utf-8'), label=b"SUBSETSUM")
        chosen_indices = []
        target = 0
        for i in range(k_items):
            if prng.random() < 0.5:
                chosen_indices.append(i)
                target += items[i]
        # Map first items to existing path_ids for traceability
        mapped_count = min(k_items, len(demands))
        item_to_lightpath = {int(i): int(demands[i].path_id) for i in range(mapped_count)}
        subset_sum = {
            "modulus": str(modulus),
            "num_items": k_items,
            "items": items,
            "target": int(target),
            "chosen_indices": chosen_indices,
            "mapping": {
                "item_index_to_path_id": item_to_lightpath
            }
        }
        with open(out_dir / "subset_sum.json", "w") as f:
            json.dump(subset_sum, f, indent=2)

        # mapping.json: describes the (didactic) transformation f from (S, T) to dataset files
        mapping = {
            "source_problem": "SubsetSum",
            "target_problem": "Optical lightpath coloring with widths",
            "f_description": {
                "nodes_generation": "Deterministic from SHA256(modulus).",
                "edges_generation": "Spanning tree + extra edges to reach average degree.",
                "demands_lightpaths": "For each demand, BFS lightpath between random src/dst under hop cap.",
                "width_assignment": "Width in {1,2,4}, mostly 1; colors assigned as contiguous ranges across links.",
                "subset_sum_embedding": "Items i map to first demands (path_id=i) where possible; witness corresponds to selecting those demands."
            },
            "soundness_note": "This mapping is pedagogical: it preserves a verifiable correspondence but does not alter the project loader.",
            "files": ["links.csv", "demands.csv", "routing.csv", "linkservicestatistics.csv", "maxW.txt", "subset_sum.json", "provenance.json", "mapping.json"]
        }
        with open(out_dir / "mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset files for Optical Network project")
    parser.add_argument("dataset", type=str, help="Dataset name, e.g. RegDataSet_1909")
    parser.add_argument("net_dir", type=str, choices=[e.value for e in NetDir], help="Network size dir")
    parser.add_argument("dist_dir", type=str, choices=[e.value for e in DistDir], help="Distance dir")
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes")
    parser.add_argument("--demands", type=int, default=500, help="Number of demands (lightpaths)")
    parser.add_argument("--degree", type=int, default=3, help="Target average degree (undirected)")
    parser.add_argument("--colors", type=int, default=80, help="Total spectrum slots per link")
    parser.add_argument("--reach", type=float, default=1200.0, help="Max total distance for a lightpath (reach)")
    parser.add_argument("--p1", type=float, default=0.85, help="Probability of width=1")
    parser.add_argument("--p2", type=float, default=0.075, help="Probability of width=2")
    parser.add_argument("--p4", type=float, default=0.075, help="Probability of width=4")
    parser.add_argument("--field-w", type=float, default=1000.0, help="Width of placement field (units)")
    parser.add_argument("--field-h", type=float, default=1000.0, help="Height of placement field (units)")
    parser.add_argument("--seed", type=int, default=None, help="Legacy RNG seed (ignored if --modulus set)")
    parser.add_argument("--modulus", type=str, default=None, help="Composite modulus N=p*q to seed SHA256 PRNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = DATA_SETS_DIR / args.dataset / args.net_dir / args.dist_dir
    modulus_int = int(args.modulus, 0) if args.modulus is not None else None
    generate_dataset(out_dir=out_dir,
                     num_nodes=args.nodes,
                     num_demands=args.demands,
                     target_degree=args.degree,
                     total_colors=args.colors,
                     max_total_distance=args.reach,
                     seed=args.seed,
                     modulus=modulus_int,
                     p1=args.p1, p2=args.p2, p4=args.p4,
                     field_w=args.field_w, field_h=args.field_h)
    print(f"Generated dataset at {out_dir}")


if __name__ == "__main__":
    main()