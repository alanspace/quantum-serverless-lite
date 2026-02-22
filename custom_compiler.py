"""
custom_compiler.py — Quantum Compilation Optimisation & Parallel Processing Demo
=================================================================================

Demonstrates three advanced Qiskit concepts:

1. **Custom Transpiler Pass** (``RedundantCXCancellation``)
   A ``TransformationPass`` that traverses the DAGCircuit and cancels
   back-to-back CX gates on identical qubit pairs.  Two adjacent CX gates
   on the same control/target are logically equivalent to the identity, so
   removing them reduces circuit depth without changing the unitary.

2. **Heavy Circuit Generator** (``generate_heavy_circuit``)
   Builds random, deep circuits with intentionally injected redundant CX
   pairs so the optimisation pass has measurable work to do.

3. **Parallel Compiler** (``ParallelCompiler``)
   Distributes a list of circuits across all available CPU cores using
   ``concurrent.futures.ProcessPoolExecutor``.  Each worker process runs
   the full Qiskit ``PassManager`` independently — no shared state.

4. **Benchmark** (``__main__``)
   Generates 1 000 circuits, times sequential vs. parallel compilation,
   and reports total time saved and average depth reduction.

Usage
-----
    conda run -n qiskit_env python custom_compiler.py

Requirements
------------
    qiskit >= 2.0
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Decompose
from qiskit_aer import AerSimulator  # noqa: F401 — confirms Aer is importable


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Custom Transpiler Pass
# ══════════════════════════════════════════════════════════════════════════════


class RedundantCXCancellation(TransformationPass):
    """Remove pairs of adjacent, identical CX gates from a DAGCircuit.

    **Mathematical basis**
    CX · CX = I  (the CNOT gate is its own inverse).
    Therefore any two consecutive CX operations sharing the same
    control and target qubit can be safely eliminated together, reducing
    circuit depth by 2 per cancelled pair.

    **DAG traversal strategy**
    The DAGCircuit represents a circuit as a directed acyclic graph where
    each node is a gate (``DAGOpNode``) and edges encode data-flow
    dependencies between qubits.  We perform a single topological sweep:

    1. Visit every gate node in topological order.
    2. When a CXGate is found, inspect its *successor* in the DAG on
       the same qubit wires.
    3. If the successor is *also* a CXGate on the *identical* (control,
       target) pair, both nodes are removed from the DAG.
    4. We collect all cancellation pairs first, then remove in bulk to
       avoid mutating the graph while iterating over it.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Apply the pass to *dag* and return the optimised DAGCircuit.

        Parameters
        ----------
        dag:
            Input DAGCircuit produced by the Qiskit transpilation pipeline.

        Returns
        -------
        DAGCircuit
            The same DAG with redundant CX pairs removed.
        """
        # Collect (node_a, node_b) pairs to remove — we must not mutate
        # the DAG during iteration, so we stage all deletions first.
        to_remove: list[tuple] = []
        already_marked: set = set()

        # topological_op_nodes() yields DAGOpNode objects in causal order.
        for node in dag.topological_op_nodes():
            if node in already_marked:
                continue  # already scheduled for removal; skip
            if not isinstance(node.op, CXGate):
                continue  # not a CX gate; nothing to cancel

            # Retrieve the two qubit arguments: [control, target]
            control_qubit, target_qubit = node.qargs

            # Look for the immediate successor on the control qubit wire.
            # dag.successors() returns both DAGOpNode and DAGOutNode objects;
            # we filter for DAGOpNode instances that are CX gates.
            for successor in dag.successors(node):
                # Guard: successor must be an operation node (not a wire end)
                if not hasattr(successor, "op"):
                    continue
                if not isinstance(successor.op, CXGate):
                    continue
                if successor in already_marked:
                    continue

                succ_control, succ_target = successor.qargs

                # Cancellation condition: identical control AND target qubits.
                if succ_control == control_qubit and succ_target == target_qubit:
                    to_remove.append((node, successor))
                    already_marked.add(node)
                    already_marked.add(successor)
                    break  # only cancel with the first eligible successor

        # ── Bulk removal ────────────────────────────────────────────────────
        # dag.remove_op_node() relinks the surrounding DAG edges so data flow
        # is preserved across the deleted gate.
        for node_a, node_b in to_remove:
            dag.remove_op_node(node_a)
            dag.remove_op_node(node_b)

        return dag


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Heavy Circuit Generator
# ══════════════════════════════════════════════════════════════════════════════


def generate_heavy_circuit(
    num_qubits: int = 6,
    depth: int = 40,
    redundant_cx_pairs: int = 8,
    seed: int | None = None,
) -> QuantumCircuit:
    """Build a deep random circuit with intentionally redundant CX pairs.

    Parameters
    ----------
    num_qubits:
        Number of qubits in the circuit.
    depth:
        Number of random gate layers applied before injecting redundancy.
    redundant_cx_pairs:
        How many back-to-back CX(a, b) · CX(a, b) pairs to inject.  Each
        pair adds 2 gates that should be cancelled by our custom pass.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    QuantumCircuit
        A circuit whose depth can be significantly reduced by
        ``RedundantCXCancellation``.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)

    # ── Random base layers ─────────────────────────────────────────────────
    single_qubit_ops = ["h", "x", "y", "z", "s", "t", "sdg", "tdg"]

    for _ in range(depth):
        # Random single-qubit gates on every qubit
        for q in range(num_qubits):
            gate = rng.choice(single_qubit_ops)
            getattr(qc, gate)(q)

        # Random CX gate (not intentionally redundant here)
        ctrl, tgt = rng.sample(range(num_qubits), 2)
        qc.cx(ctrl, tgt)

    # ── Inject redundant CX pairs ─────────────────────────────────────────
    # Each pair CX(a,b) · CX(a,b) cancels to identity — the optimiser's
    # job is to find and remove all of these.
    for _ in range(redundant_cx_pairs):
        ctrl, tgt = rng.sample(range(num_qubits), 2)
        qc.cx(ctrl, tgt)  # first  CX — these two together …
        qc.cx(ctrl, tgt)  # second CX — … are the identity

    return qc


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Parallel Compiler
# ══════════════════════════════════════════════════════════════════════════════


def _compile_single(args: Tuple[QuantumCircuit, PassManager]) -> Tuple[int, int]:
    """Worker function executed inside each subprocess.

    Kept as a plain module-level function (not a method) so that Python's
    ``multiprocessing`` pickle mechanism can serialise it correctly across
    process boundaries.

    Parameters
    ----------
    args:
        A ``(circuit, pass_manager)`` tuple.  Tuples are used instead of
        keyword arguments because ``ProcessPoolExecutor.map`` passes a
        single iterable element per call.

    Returns
    -------
    (depth_before, depth_after)
        The circuit depths before and after compilation, useful for
        benchmarking depth reduction.
    """
    circuit, pm = args
    depth_before = circuit.depth()
    optimised = pm.run(circuit)
    depth_after = optimised.depth()
    return depth_before, depth_after


class ParallelCompiler:
    """Distribute circuit compilation across all available CPU cores.

    Wraps ``concurrent.futures.ProcessPoolExecutor`` to fan out a
    ``PassManager`` run over a list of circuits.  Each subprocess is
    independent — Qiskit's transpiler is not thread-safe, but separate
    *processes* each have their own GIL, making true parallelism possible.

    Parameters
    ----------
    circuits:
        The list of ``QuantumCircuit`` objects to compile.
    pass_manager:
        A configured Qiskit ``PassManager`` (e.g. containing
        ``RedundantCXCancellation``).
    max_workers:
        Number of parallel processes.  Defaults to the CPU core count.
    """

    def __init__(
        self,
        circuits: List[QuantumCircuit],
        pass_manager: PassManager,
        max_workers: int | None = None,
    ) -> None:
        self.circuits = circuits
        self.pass_manager = pass_manager
        self.max_workers = max_workers  # None → os.cpu_count()

    def compile(self) -> List[Tuple[int, int]]:
        """Run the pass manager over all circuits in parallel.

        Returns
        -------
        List[Tuple[int, int]]
            List of ``(depth_before, depth_after)`` pairs, one per circuit,
            in the original submission order.
        """
        # Build argument tuples — one per circuit.  The PassManager is
        # serialised once per worker process (via pickle), then reused for
        # all circuits assigned to that worker.
        args = [(circ, self.pass_manager) for circ in self.circuits]

        results: list[tuple[int, int]] = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map preserves submission order, which matches self.circuits.
            for depth_pair in executor.map(_compile_single, args, chunksize=20):
                results.append(depth_pair)

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Sequential baseline (single-process, for benchmarking)
# ══════════════════════════════════════════════════════════════════════════════


def compile_sequential(
    circuits: List[QuantumCircuit],
    pass_manager: PassManager,
) -> List[Tuple[int, int]]:
    """Compile circuits one-by-one in the calling process.

    Used as the baseline for the parallel speedup benchmark.

    Returns
    -------
    List[Tuple[int, int]]
        ``(depth_before, depth_after)`` pairs in order.
    """
    results = []
    for circ in circuits:
        depth_before = circ.depth()
        optimised = pass_manager.run(circ)
        depth_after = optimised.depth()
        results.append((depth_before, depth_after))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Benchmark entry point
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import os

    NUM_CIRCUITS = 1_000
    NUM_QUBITS = 6
    CIRCUIT_DEPTH = 30
    REDUNDANT_PAIRS = 8  # gates saved per circuit = 2 × REDUNDANT_PAIRS

    print("=" * 70)
    print("  Quantum Compilation Benchmark — RedundantCXCancellation")
    print("=" * 70)
    print(f"  Circuits        : {NUM_CIRCUITS:,}")
    print(f"  Qubits          : {NUM_QUBITS}")
    print(f"  Base depth      : {CIRCUIT_DEPTH}")
    print(f"  Redundant pairs : {REDUNDANT_PAIRS}  ({REDUNDANT_PAIRS * 2} gates/circuit)")
    print(f"  CPU cores       : {os.cpu_count()}")
    print()

    # ── Build circuits ───────────────────────────────────────────────────────
    print("[1/4] Generating circuits …", flush=True)
    circuits = [
        generate_heavy_circuit(
            num_qubits=NUM_QUBITS,
            depth=CIRCUIT_DEPTH,
            redundant_cx_pairs=REDUNDANT_PAIRS,
            seed=i,  # deterministic but varied per circuit
        )
        for i in range(NUM_CIRCUITS)
    ]
    print(f"      Done.  Example circuit depth (before): {circuits[0].depth()}")
    print()

    # ── Build pass manager ───────────────────────────────────────────────────
    print("[2/4] Configuring PassManager …", flush=True)
    pm = PassManager([RedundantCXCancellation()])
    print("      PassManager: [RedundantCXCancellation]")
    print()

    # ── Sequential benchmark ─────────────────────────────────────────────────
    print("[3/4] Sequential compilation …", flush=True)
    t0 = time.perf_counter()
    seq_results = compile_sequential(circuits, pm)
    t_seq = time.perf_counter() - t0
    print(f"      Completed in {t_seq:.3f} s")
    print()

    # ── Parallel benchmark ───────────────────────────────────────────────────
    print("[4/4] Parallel compilation …", flush=True)
    compiler = ParallelCompiler(circuits, pm)
    t0 = time.perf_counter()
    par_results = compiler.compile()
    t_par = time.perf_counter() - t0
    print(f"      Completed in {t_par:.3f} s")
    print()

    # ── Results ──────────────────────────────────────────────────────────────
    depth_reductions = [before - after for before, after in seq_results]
    avg_reduction = sum(depth_reductions) / len(depth_reductions)
    speedup = t_seq / t_par if t_par > 0 else float("inf")
    time_saved = t_seq - t_par

    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Sequential time   : {t_seq:.3f} s")
    print(f"  Parallel time     : {t_par:.3f} s")
    print(f"  Time saved        : {time_saved:.3f} s  ({(time_saved/t_seq)*100:.1f}%)")
    print(f"  Speedup factor    : {speedup:.2f}×")
    print()
    print(f"  Avg depth before  : {sum(b for b, _ in seq_results) / NUM_CIRCUITS:.1f}")
    print(f"  Avg depth after   : {sum(a for _, a in seq_results) / NUM_CIRCUITS:.1f}")
    print(f"  Avg depth reduction: {avg_reduction:.1f} gates/circuit")
    print(f"  Expected reduction: {REDUNDANT_PAIRS * 2}   gates/circuit  (2 × {REDUNDANT_PAIRS} pairs)")
    print("=" * 70)
