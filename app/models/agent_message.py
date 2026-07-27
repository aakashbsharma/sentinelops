import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models._columns import JSONDict, UUIDType, created_at_col, uuid_pk

if TYPE_CHECKING:
    from app.models.incident import Incident


class AgentMessage(Base):
    """One typed message emitted by an agent during an incident.

    `content` stores the serialized Pydantic payload (Stage 4 defines the
    schemas); `confidence` and `needs_more_data` are first-class columns so
    the orchestrator can route on them without parsing JSON.
    """

    __tablename__ = "agent_messages"

    id: Mapped[uuid.UUID] = uuid_pk()
    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType,
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # role: triage / diagnostician / planner / executor / orchestrator
    role: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    # Correlation id (Stage 7): which orchestrator run produced this message.
    # An escalated-then-resumed incident has multiple runs; run_id is what
    # separates them in a trace. Nullable: pre-Stage-7 rows have no run.
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSONDict, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    needs_more_data: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    created_at: Mapped[datetime] = created_at_col()

    incident: Mapped["Incident"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<AgentMessage {self.id} {self.role}:{self.agent_name}>"
