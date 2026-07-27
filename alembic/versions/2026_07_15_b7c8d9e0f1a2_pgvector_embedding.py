"""enable pgvector; memory_records.embedding JSONB -> vector(384) + HNSW index

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-07-15

The Stage 1 placeholder column was never populated, so we drop/re-add rather
than write a JSONB->vector USING cast for data that doesn't exist.

Index choice — HNSW over IVFFlat:
  - HNSW gives better recall at this corpus size and needs no training step;
    IVFFlat requires representative data BEFORE index build (its list
    centroids are computed from existing rows), which is awkward for a table
    that starts empty and grows incident-by-incident.
  - HNSW costs more RAM and slower writes — irrelevant at episodic-memory
    write rates (a few rows per resolved incident).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "b7c8d9e0f1a2"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.drop_column("memory_records", "embedding")
    op.add_column(
        "memory_records",
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    )

    # vector_cosine_ops matches the `<=>` operator used by retrieval.
    op.execute(
        "CREATE INDEX ix_memory_records_embedding_hnsw "
        "ON memory_records USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memory_records_embedding_hnsw")
    op.drop_column("memory_records", "embedding")
    op.add_column(
        "memory_records",
        sa.Column("embedding", sa.dialects.postgresql.JSONB(), nullable=True),
    )
    # extension left installed on purpose: other tables may adopt vectors,
    # and DROP EXTENSION would fail if they have.
