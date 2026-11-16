-- 003_add_specific_title.sql
-- Preserve entity-specific headline (`specific_title`) from Lisuns News API alongside
-- the main story `title`, and index `url` to group multi-company articles from the same story.

ALTER TABLE articles ADD COLUMN IF NOT EXISTS specific_title TEXT;
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url);

