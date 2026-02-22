## 📊 Compiler Benchmark Insights

Included in this repository is `custom_compiler.py`, a custom Qiskit `TransformationPass` demonstrating Directed Acyclic Graph (DAG) manipulation and parallel compilation via `ProcessPoolExecutor`.

**Benchmark Results (1,000 circuits, depth=30, 12 CPU Cores):**
* **Parallelization Overhead:** Sequential compilation took 0.752s vs. Parallel at 0.944s. This benchmark explicitly demonstrates the Inter-Process Communication (IPC) and pickling overhead. It proves that parallel pass managers are only advantageous for deep, utility-scale circuits where the $O(N)$ graph traversal cost strictly outweighs the IPC serialization penalty.
* **DAG Adjacency vs. Commutativity:** The custom pass successfully reduced circuit depth by an average of 11.0 gates per circuit (an ~18% reduction). It did not catch all 16 injected redundant gates because the V1 pass relies on strict topological adjacency. Future research involves implementing Commutativity Analysis to push non-adjacent, commuting gates through the DAG to uncover hidden cancellations.