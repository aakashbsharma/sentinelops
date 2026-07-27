import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.core.database import Base
from app.models._columns import JSONDict, UUIDType, created_at_col, uuid_pk

if TYPE_CHECKING:
    from app.models.incident import Incident

# all-MiniLM-L6-v2 output size. Changing embedding models to a different
# dimension means one Alembic migration (resize vector column + reindex).
EMBEDDING_DIM = 384

# pgvector Vector on Postgres; plain JSON list-of-floats on other dialects
# (keeps the SQLite unit-test path working — similarity search there falls
# back to Python-side cosine, see app/memory/_retrieval.py).
EmbeddingType = JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql")


class MemoryRecord(Base):
    """A row in long-term memory.

    memory_type: "episodic" (past incidents + outcomes, linked to an incident)
                 or "semantic" (runbooks/docs, incident_id is NULL).
    """

    __tablename__ = "memory_records"

    id: Mapped[uuid.UUID] = uuid_pk()
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        UUIDType,
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType, nullable=True)
    # `metadata` is reserved by SQLAlchemy's Declarative API, hence metadata_.
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONDict, nullable=True
    )

    created_at: Mapped[datetime] = created_at_col()

    incident: Mapped["Incident | None"] = relationship(
        back_populates="memory_records"
    )

    def __repr__(self) -> str:
        return f"<MemoryRecord {self.id} [{self.memory_type}]>"
