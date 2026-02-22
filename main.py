"""
Quantum Serverless Lite — FastAPI application.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

from typing import Any, Literal

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.celery_app import celery_app
from tasks import run_quantum_estimator

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quantum Serverless Lite",
    description="Serverless orchestrator for Qiskit quantum workloads via Celery + Redis.",
    version="0.1.0",
)

# ── Pydantic models ────────────────────────────────────────────────────────────


class EstimatorRequest(BaseModel):
    """Payload for submitting a quantum estimator job."""

    qasm: str = Field(
        ...,
        description="OpenQASM 3.0 circuit string to execute.",
        examples=["OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];"],
    )
    observable: str = Field(
        ...,
        description="Pauli observable string, e.g. 'ZZ' or 'XY'.",
        examples=["ZZ"],
    )


class EstimatorResponse(BaseModel):
    """Immediate acknowledgement returned after enqueuing a job."""

    task_id: str = Field(..., description="Celery task ID — use this to poll for results.")
    status: Literal["QUEUED"] = Field("QUEUED", description="Always 'QUEUED' on submission.")


class ResultResponse(BaseModel):
    """Poll response for a previously submitted job."""

    task_id: str
    status: Literal["PENDING", "COMPLETED", "FAILED"]
    result: Any | None = Field(
        None,
        description="The task result payload, populated only when status is 'COMPLETED'.",
    )
    error: str | None = Field(
        None,
        description="Error message, populated only when status is 'FAILED'.",
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/quantum/estimator",
    response_model=EstimatorResponse,
    status_code=202,
    summary="Submit a quantum estimator job",
    tags=["Quantum"],
)
def submit_estimator(payload: EstimatorRequest) -> EstimatorResponse:
    """
    Enqueue a Celery task that runs the Qiskit Estimator primitive.

    - Accepts an **OpenQASM 3.0** circuit and a **Pauli observable** string.
    - Returns a `task_id` immediately; the job executes asynchronously.
    """
    task = run_quantum_estimator.delay(
        qasm=payload.qasm,
        observable=payload.observable,
    )
    return EstimatorResponse(task_id=task.id, status="QUEUED")


@app.get(
    "/api/v1/quantum/results/{task_id}",
    response_model=ResultResponse,
    summary="Poll the result of a quantum estimator job",
    tags=["Quantum"],
)
def get_result(task_id: str) -> ResultResponse:
    """
    Check the status of a previously submitted estimator job.

    | Celery state | Returned `status` |
    |---|---|
    | PENDING / STARTED / RETRY | `PENDING` |
    | SUCCESS | `COMPLETED` |
    | FAILURE / REVOKED | `FAILED` |
    """
    async_result: AsyncResult = celery_app.AsyncResult(task_id)

    if async_result.state in {"PENDING", "STARTED", "RETRY"}:
        return ResultResponse(task_id=task_id, status="PENDING")

    if async_result.state == "SUCCESS":
        return ResultResponse(
            task_id=task_id,
            status="COMPLETED",
            result=async_result.result,
        )

    # FAILURE, REVOKED, or any unknown terminal state
    error_msg = str(async_result.result) if async_result.result else async_result.state
    return ResultResponse(task_id=task_id, status="FAILED", error=error_msg)


# ── Health check ───────────────────────────────────────────────────────────────


@app.get("/health", tags=["Ops"], summary="Service health check")
def health() -> dict[str, str]:
    """Returns 200 OK when the API server is reachable."""
    return {"status": "ok"}
