-- Migration: Add information sufficiency + clarification fields to report_analysis
-- and create the report_clarifications audit table.
-- Run against your PostgreSQL database: psql -d precursorai -f this_file.sql

BEGIN;

-- 1. New columns on report_analysis
ALTER TABLE report_analysis
    ADD COLUMN IF NOT EXISTS person_in_proximity        BOOLEAN,
    ADD COLUMN IF NOT EXISTS information_sufficiency    INTEGER,
    ADD COLUMN IF NOT EXISTS information_sufficiency_reason TEXT,
    ADD COLUMN IF NOT EXISTS followup_reason            TEXT;

-- 2. New audit table for clarifications
CREATE TABLE IF NOT EXISTS report_clarifications (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id   UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    answer      TEXT,          -- NULL until user responds
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_clarifications_report_id
    ON report_clarifications (report_id);

COMMIT;
