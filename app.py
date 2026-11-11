"""Optional FastAPI wrapper for SyntheticForge.

FastAPI is imported defensively so the core CLI and library can be used in a
minimal environment.  Install the API extras to run this module as a server:

    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload
"""

from __future__ import annotations

import io

import pandas as pd

from evaluate import evaluate_synthetic
from generate import generate_synthetic

try:
    from fastapi import FastAPI, File, Form, UploadFile
except ImportError:  # pragma: no cover - exercised only without FastAPI installed
    FastAPI = None  # type: ignore[assignment]
    File = Form = UploadFile = None  # type: ignore[assignment]


if FastAPI is not None:
    app = FastAPI(title="SyntheticForge", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(
        file: UploadFile = File(...),
        rows: int | None = Form(default=None),
        method: str = Form(default="copula"),
        seed: int = Form(default=42),
    ) -> dict:
        contents = await file.read()
        real = pd.read_csv(io.BytesIO(contents))
        synthetic = generate_synthetic(real, n_rows=rows, method=method, seed=seed)  # type: ignore[arg-type]
        report = evaluate_synthetic(real, synthetic)
        return {
            "records": synthetic.where(pd.notnull(synthetic), None).to_dict(orient="records"),
            "report": report,
        }
else:  # pragma: no cover
    app = None


if __name__ == "__main__" and app is None:
    raise SystemExit("FastAPI is not installed. Install fastapi, uvicorn and python-multipart to run the API.")
