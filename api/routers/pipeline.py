"""
/api/v1/pipeline endpoints — pipeline status and manual trigger.
"""
from fastapi import APIRouter, BackgroundTasks

from api.database import pipeline_runs_col

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status")
async def pipeline_status():
    """Return the most recent pipeline run document."""
    col = pipeline_runs_col()
    doc = await col.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    if not doc:
        return {"status": "never_run", "message": "No pipeline runs recorded yet"}
    return doc


@router.get("/history")
async def pipeline_history(limit: int = 10):
    """Return the last N pipeline run records."""
    col = pipeline_runs_col()
    cursor = col.find({}, {"_id": 0}, sort=[("started_at", -1)], limit=limit)
    docs = [d async for d in cursor]
    return {"runs": docs}


@router.post("/trigger")
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
):
    """Manually trigger the pipeline (runs in background)."""
    from api.scheduler import run_pipeline
    background_tasks.add_task(run_pipeline)
    return {"status": "triggered", "message": "Pipeline started in background"}
