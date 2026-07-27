import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models._columns import JSONDict, created_at_col, updated_at_col, uuid_pk

if TYPE_CHECKING:
    from app.models.agent_message import AgentMessage
    from app.models.approval import ApprovalRequest
    from app.models.memory import MemoryRecord


class Incident(Base):
    """The unit of work: one alert/event the agent swarm is responding to.

    status lifecycle: open -> triaging -> diagnosing -> awaiting_approval
                      -> remediating -> resolved -> closed
    severity: low / medium / high / critical
    Both are plain strings by design (no DB enum) so values can evolve
    without migrations.
    """

    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = uuid_pk()
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="synthetic")
    severity: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="open", index=True
    )
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONDict, nullable=True)

    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="AgentMessage.created_at",
    )
    approval_requests: Mapped[list["ApprovalRequest"]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ApprovalRequest.requested_at",
    )
    memory_records: Mapped[list["MemoryRecord"]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Incident {self.id} [{self.severity}/{self.status}] {self.title!r}>"
