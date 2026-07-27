"""Shared column type helpers.

Postgres gets native UUID / JSONB; other dialects (e.g. SQLite in unit tests)
fall back to portable representations. Statuses/severities are deliberately
plain strings, not DB enums, so new values never require a migration.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

# JSONB on Postgres, generic JSON elsewhere.
JSONDict = JSON().with_variant(JSONB(), "postgresql")

# Native UUID on Postgres, CHAR(32) elsewhere.
UUIDType = Uuid(as_uuid=True)


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUIDType, primary_key=True, default=uuid.uuid4)


def created_at_col() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def updated_at_col() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
