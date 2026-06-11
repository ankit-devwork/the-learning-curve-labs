"""
Health check endpoint.

Used for:
- Deployment validation
- Monitoring
- Load balancer checks
"""

from fastapi import APIRouter
from pycorekit.logging.logger import get_logger

router = APIRouter(tags=["Health"])
log = get_logger("health")

@router.get(
    "/health",
    summary="Health check endpoint",
    description=(
        "Returns the operational status of the API. "
        "Useful for monitoring, uptime checks, and deployment validation."
    ),
    operation_id="healthCheck"
)
async def health_check():
    log.info("Health check invoked")
    return {"status": "ok", "message": "API is running"}

