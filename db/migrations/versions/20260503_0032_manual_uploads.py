"""Track dashboard manual file uploads.

Original upload bytes are staged in R2 only long enough for the normalizer to
persist extracted text into documents/chunks. The worker deletes the staged
object before marking the queue row done and records original_deleted_at here.

Revision ID: 0032_manual_uploads
Revises: 0031_codex_device_source_bkfl
Create Date: 2026-05-03
"""
from alembic import op

# revision identifiers, used by Alembic. Keep IDs <= 32 chars.
revision = "0032_manual_uploads"
down_revision = "0031_codex_device_source_bkfl"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_uploads (
            upload_id           TEXT PRIMARY KEY,
            customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            filename            TEXT NOT NULL,
            content_type        TEXT NOT NULL DEFAULT 'application/octet-stream',
            file_size_bytes     BIGINT NOT NULL DEFAULT 0,
            file_sha256         TEXT NOT NULL,
            staging_object_key  TEXT,
            payload_object_key  TEXT,
            uploaded_by         TEXT,
            uploaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status              TEXT NOT NULL,
            parse_engine        TEXT,
            parse_error         TEXT,
            extracted_chars     INT NOT NULL DEFAULT 0,
            doc_id              TEXT,
            indexed_at          TIMESTAMPTZ,
            original_deleted_at TIMESTAMPTZ,
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT manual_uploads_status_check CHECK (
                status IN ('queued', 'indexed', 'failed_parse', 'failed_ingest')
            )
        );

        CREATE INDEX IF NOT EXISTS idx_manual_uploads_customer_uploaded
            ON manual_uploads (customer_id, uploaded_at DESC);
        CREATE INDEX IF NOT EXISTS idx_manual_uploads_customer_status
            ON manual_uploads (customer_id, status, uploaded_at DESC);
        CREATE INDEX IF NOT EXISTS idx_manual_uploads_doc
            ON manual_uploads (customer_id, doc_id)
            WHERE doc_id IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_manual_uploads_doc;
        DROP INDEX IF EXISTS idx_manual_uploads_customer_status;
        DROP INDEX IF EXISTS idx_manual_uploads_customer_uploaded;
        DROP TABLE IF EXISTS manual_uploads;
        """
    )
