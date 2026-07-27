import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models._columns import JSONDict, UUIDType, created_at_col, uuid_pk

if TYPE_CHECKING:
    from app.models.incident import Incident


class ApprovalRequest(Base):
    """Human-in-the-loop gate: a proposed remediation awaiting a decision.

    status: pending / approved / rejected / auto_denied
    (auto_denied = guardrail rejected it before a human ever saw it).
    """

    __tablename__ = "approval_requests"

    id: Mapped[uuid.UUID] = uuid_pk()
    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType,
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    proposed_action: Mapped[dict[str, Any]] = mapped_column(JSONDict, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )

    requested_at: Mapped[datetime] = created_at_col()
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_by: Mapped[str | None] = mapped_column(String(200), nullable=True)

    incident: Mapped["Incident"] = relationship(back_populates="approval_requests")

    def __repr__(self) -> str:
        return f"<ApprovalRequest {self.id} [{self.status}] risk={self.risk_level}>"
