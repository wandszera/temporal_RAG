"""Camada de persistência bitemporal (SQLite e PostgreSQL)."""

import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    create_engine,
    select,
    delete,
    MetaData,
)
from sqlalchemy.orm import declarative_base

from temporal_rag.config import settings
from temporal_rag.models.schema import Customer, EventType, TimelineEvent

metadata = MetaData()

customers_table = Table(
    "customers",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(255), nullable=False),
    Column("email", String(255), nullable=False),
    Column("phone", String(64)),
    Column("company", String(255)),
    Column("document_cpf", String(64)),
    Column("created_at", DateTime, nullable=False),
    Column("initial_plan", String(64), nullable=False),
    Column("current_plan", String(64)),
    Column("address", Text),
)

timeline_events_table = Table(
    "timeline_events",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("customer_id", String(64), ForeignKey("customers.id"), nullable=False, index=True),
    Column("event_type", String(64), nullable=False, index=True),
    Column("title", String(255), nullable=False),
    Column("content", Text, nullable=False),
    Column("metadata_json", Text, default="{}"),
    # Bitemporal indices
    Column("valid_from", DateTime, nullable=False, index=True),
    Column("valid_to", DateTime, nullable=True, index=True),
    Column("recorded_at", DateTime, nullable=False, index=True),
)

# Composite indices for bitemporal queries
Index("idx_bitemporal_lookup", timeline_events_table.c.customer_id, timeline_events_table.c.valid_from, timeline_events_table.c.valid_to)


class Database:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self.engine = create_engine(self.db_url, echo=False)
        self.init_db()

    def init_db(self):
        metadata.create_all(self.engine)

    def clear_all(self):
        with self.engine.begin() as conn:
            conn.execute(delete(timeline_events_table))
            conn.execute(delete(customers_table))

    def save_customer(self, customer: Customer):
        with self.engine.begin() as conn:
            conn.execute(
                customers_table.insert().values(
                    id=customer.id,
                    name=customer.name,
                    email=customer.email,
                    phone=customer.phone,
                    company=customer.company,
                    document_cpf=customer.document_cpf,
                    created_at=customer.created_at,
                    initial_plan=customer.initial_plan,
                    current_plan=customer.current_plan,
                    address=customer.address,
                )
            )

    def save_events(self, events: List[TimelineEvent]):
        if not events:
            return
        rows = [
            {
                "id": evt.id,
                "customer_id": evt.customer_id,
                "event_type": evt.event_type.value,
                "title": evt.title,
                "content": evt.content,
                "metadata_json": json.dumps(evt.metadata, ensure_ascii=False),
                "valid_from": evt.valid_from,
                "valid_to": evt.valid_to,
                "recorded_at": evt.recorded_at,
            }
            for evt in events
        ]
        with self.engine.begin() as conn:
            conn.execute(timeline_events_table.insert(), rows)

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        with self.engine.connect() as conn:
            stmt = select(customers_table).where(customers_table.c.id == customer_id)
            row = conn.execute(stmt).mappings().first()
            if not row:
                return None
            return Customer(
                id=row["id"],
                name=row["name"],
                email=row["email"],
                phone=row["phone"],
                company=row["company"],
                document_cpf=row["document_cpf"],
                created_at=row["created_at"],
                initial_plan=row["initial_plan"],
                current_plan=row["current_plan"],
                address=row["address"],
            )

    def list_customers(self, limit: int = 100) -> List[Customer]:
        with self.engine.connect() as conn:
            stmt = select(customers_table).limit(limit)
            rows = conn.execute(stmt).mappings().all()
            return [
                Customer(
                    id=row["id"],
                    name=row["name"],
                    email=row["email"],
                    phone=row["phone"],
                    company=row["company"],
                    document_cpf=row["document_cpf"],
                    created_at=row["created_at"],
                    initial_plan=row["initial_plan"],
                    current_plan=row["current_plan"],
                    address=row["address"],
                )
                for row in rows
            ]

    def get_all_events(self, customer_id: str) -> List[TimelineEvent]:
        with self.engine.connect() as conn:
            stmt = (
                select(timeline_events_table)
                .where(timeline_events_table.c.customer_id == customer_id)
                .order_by(timeline_events_table.c.valid_from.asc())
            )
            rows = conn.execute(stmt).mappings().all()
            return self._rows_to_events(rows)

    def get_bitemporal_events(
        self,
        customer_id: str,
        as_of_valid_time: Optional[datetime] = None,
        as_of_recorded_time: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
    ) -> List[TimelineEvent]:
        """Consulta eventos respeitando as dimensões bitemporais (valid_time e recorded_time)."""
        stmt = select(timeline_events_table).where(
            timeline_events_table.c.customer_id == customer_id
        )

        if as_of_valid_time is not None:
            # Evento começou antes ou no momento as_of e terminou depois (ou não terminou)
            stmt = stmt.where(
                timeline_events_table.c.valid_from <= as_of_valid_time
            ).where(
                (timeline_events_table.c.valid_to == None)  # noqa: E711
                | (timeline_events_table.c.valid_to > as_of_valid_time)
            )

        if as_of_recorded_time is not None:
            # Evita vazamento de dados gravados no futuro
            stmt = stmt.where(
                timeline_events_table.c.recorded_at <= as_of_recorded_time
            )

        if event_type is not None:
            stmt = stmt.where(timeline_events_table.c.event_type == event_type.value)

        stmt = stmt.order_by(timeline_events_table.c.valid_from.asc())

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
            return self._rows_to_events(rows)

    def _rows_to_events(self, rows) -> List[TimelineEvent]:
        events = []
        for row in rows:
            meta = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception:
                    meta = {}
            events.append(
                TimelineEvent(
                    id=row["id"],
                    customer_id=row["customer_id"],
                    event_type=EventType(row["event_type"]),
                    title=row["title"],
                    content=row["content"],
                    metadata=meta,
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    recorded_at=row["recorded_at"],
                )
            )
        return events
