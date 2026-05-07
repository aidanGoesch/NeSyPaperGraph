CREATE TABLE IF NOT EXISTS storage_documents (
    document_key VARCHAR(64) PRIMARY KEY,
    payload JSONB NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id VARCHAR(128) PRIMARY KEY,
    paper_identity_key VARCHAR(255) NOT NULL UNIQUE,
    paper_title TEXT NOT NULL,
    semantic_scholar_paper_id VARCHAR(255),
    normalized_title TEXT NOT NULL DEFAULT '',
    canonical_source_url TEXT NOT NULL DEFAULT '',
    selected_primary_source TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'unread',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_papers_semantic_scholar
ON papers (semantic_scholar_paper_id)
WHERE semantic_scholar_paper_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS paper_links (
    id BIGSERIAL PRIMARY KEY,
    paper_id VARCHAR(128) NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    raw_url TEXT NOT NULL,
    normalized_url TEXT NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_links_normalized_url
ON paper_links (normalized_url);

CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_links_single_primary
ON paper_links (paper_id)
WHERE is_primary = TRUE;

CREATE TABLE IF NOT EXISTS paper_notes (
    paper_id VARCHAR(128) PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    notes_markdown TEXT NOT NULL DEFAULT '',
    topic_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'unread',
    updated_at TIMESTAMPTZ NOT NULL
);
