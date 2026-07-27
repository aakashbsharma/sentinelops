"""agent_messages.run_id — correlation id for one orchestrator run (Stage 7)

Revision ID: c3d4e5f6a7b8
Revises: b7c8d9e0f1a2
Create Date: 2026-07-16

Nullable on purpose: rows written before Stage 7 have no run to belong to,
and backfilling a fabricated run_id would be worse than an honest NULL.
Indexed because the trace panel and eval harness both query by it.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b7c8d9e0f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_messages",
        sa.Column("run_id", sa.String(length=36), nullable=True),
    )
    op.create_index(
        op.f("ix_agent_messages_run_id"), "agent_messages", ["run_id"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_messages_run_id"), table_name="agent_messages")
    op.drop_column("agent_messages", "run_id")
