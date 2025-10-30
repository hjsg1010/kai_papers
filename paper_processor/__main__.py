"""Run the FastAPI app with ``python -m paper_processor``."""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("paper_processor.api:app", host="0.0.0.0", port=7070, reload=False, workers=2)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    main()
