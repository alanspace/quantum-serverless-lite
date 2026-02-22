"""
Celery task definitions for Quantum Serverless Lite.

Each task is registered against the shared `celery_app` instance.
"""

from __future__ import annotations

from core.celery_app import celery_app


@celery_app.task(name="run_quantum_estimator", bind=True)
def run_quantum_estimator(self, *, qasm: str, observable: str) -> dict:
    """
    Execute a Qiskit v2.x Estimator primitive locally via AerSimulator.

    Args:
        qasm:       OpenQASM 3.0 circuit string.
        observable: Pauli observable string (e.g. 'ZZ', 'XY').

    Returns:
        {
            "expectation_value": float,
            "standard_deviation": float,
            "metadata": {"simulator": "AerSimulator"},
        }

    Raises:
        Re-raises any exception so Celery marks the task as FAILED.
    """
    try:
        # ── 1. Parse QASM 3.0 → QuantumCircuit ────────────────────────────
        from qiskit.qasm3 import loads as qasm3_loads

        circuit = qasm3_loads(qasm)

        # ── 2. Build observable ────────────────────────────────────────────
        from qiskit.quantum_info import SparsePauliOp

        obs = SparsePauliOp.from_list([(observable, 1.0)])

        # ── 3. Initialise local AerSimulator backend ───────────────────────
        from qiskit_aer import AerSimulator

        local_backend = AerSimulator()

        # ── 4. Initialise EstimatorV2 bound to the local backend ───────────
        from qiskit_ibm_runtime import EstimatorV2

        estimator = EstimatorV2(mode=local_backend)

        # ── 5. Build PUB (Primitive Unified Bloc) and run ──────────────────
        pub = (circuit, obs)
        job = estimator.run([pub])
        result = job.result()

        # ── 6. Extract EVS / STDs and convert numpy scalars → Python floats
        pub_result = result[0]
        evs = pub_result.data.evs
        stds = pub_result.data.stds

        expectation_value = float(evs.item()) if hasattr(evs, "item") else float(evs)
        standard_deviation = float(stds.item()) if hasattr(stds, "item") else float(stds)

        return {
            "expectation_value": expectation_value,
            "standard_deviation": standard_deviation,
            "metadata": {"simulator": "AerSimulator"},
        }

    except Exception as exc:
        # Re-raise so Celery correctly sets the task state to FAILURE
        raise exc
