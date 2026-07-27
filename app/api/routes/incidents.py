"""Incident endpoints: create/inspect incidents, approve/reject remediations.

POST endpoints return 202: orchestration is asynchronous by design — the API
only records intent and enqueues a Celery task; a worker drives the agents.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.approval import ApprovalRequest
from app.observability.cost import compute_cost_summary
from app.models.incident import Incident
from app.schemas.incidents import (
    ApprovalDecisionIn,
    ApprovalRequestOut,
    IncidentAccepted,
    IncidentCreate,
    IncidentDetailOut,
    IncidentListOut,
    IncidentOut,
)
from app.tasks import orchestrate_incident_task, resume_incident_task

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=IncidentAccepted)
async def create_incident(
    body: IncidentCreate, db: AsyncSession = Depends(get_db)
) -> IncidentAccepted:
    incident = Incident(
        title=body.title,
        description=body.description,
        source=body.source,
        severity=body.severity,
        status="open",
        raw_payload=body.raw_payload,
    )
    db.add(incident)
    # Commit BEFORE enqueueing: the worker may pick the task up instantly and
    # must be able to see the row from its own session.
    await db.commit()
    orchestrate_incident_task.delay(str(incident.id))
    return IncidentAccepted(id=incident.id, status=incident.status)


@router.get("", response_model=IncidentListOut)
async def list_incidents(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> IncidentListOut:
    base = select(Incident)
    if status_filter is not None:
        base = base.where(Incident.status == status_filter)

    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        (
            await db.execute(
                base.order_by(Incident.created_at.desc()).limit(limit).offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return IncidentListOut(
        items=[IncidentOut.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{incident_id}", response_model=IncidentDetailOut)
async def get_incident(
    incident_id: UUID, db: AsyncSession = Depends(get_db)
) -> IncidentDetailOut:
    incident = (
        await db.execute(
            select(Incident)
            .where(Incident.id == incident_id)
            # messages relationship is ordered by created_at (model definition)
            .options(selectinload(Incident.messages))
        )
    ).scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    detail = IncidentDetailOut.model_validate(incident)
    detail.cost_summary = compute_cost_summary(incident.messages)
    return detail


@router.get("/{incident_id}/approval", response_model=ApprovalRequestOut)
async def get_pending_approval(
    incident_id: UUID, db: AsyncSession = Depends(get_db)
) -> ApprovalRequestOut:
    approval = (
        await db.execute(
            select(ApprovalRequest)
            .where(
                ApprovalRequest.incident_id == incident_id,
                ApprovalRequest.status == "pending",
            )
            .order_by(ApprovalRequest.requested_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if approval is None:
        raise HTTPException(
            status_code=404, detail="No pending approval request for this incident"
        )
    return ApprovalRequestOut.model_validate(approval)


@router.post(
    "/{incident_id}/approve",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IncidentAccepted,
)
async def approve_incident(
    incident_id: UUID,
    body: ApprovalDecisionIn,
    db: AsyncSession = Depends(get_db),
) -> IncidentAccepted:
    incident = await db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Incident is {incident.status!r}, not 'awaiting_approval'",
        )
    pending = (
        await db.execute(
            select(ApprovalRequest.id)
            .where(
                ApprovalRequest.incident_id == incident_id,
                ApprovalRequest.status == "pending",
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if pending is None:
        raise HTTPException(
            status_code=409, detail="No pending approval request to resolve"
        )

    resume_incident_task.delay(str(incident_id), body.approved, body.resolved_by)
    return IncidentAccepted(id=incident_id, status=incident.status)
