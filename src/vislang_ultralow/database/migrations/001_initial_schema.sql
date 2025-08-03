-- Initial database schema for VisLang-UltraLow-Resource
-- This file contains the SQL statements to create the initial database schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url VARCHAR(2048) NOT NULL UNIQUE,
    title VARCHAR(1024) NOT NULL,
    source VARCHAR(50) NOT NULL,
    language VARCHAR(10) NOT NULL,
    content TEXT,
    content_type VARCHAR(50) DEFAULT 'html',
    word_count INTEGER DEFAULT 0 CHECK (word_count >= 0),
    publication_date TIMESTAMP,
    scraped_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW(),
    quality_score REAL DEFAULT 0.0 CHECK (quality_score >= 0 AND quality_score <= 1),
    processing_status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    src VARCHAR(2048),
    alt_text TEXT,
    width INTEGER CHECK (width >= 0),
    height INTEGER CHECK (height >= 0),
    page_number INTEGER,
    image_type VARCHAR(50) DEFAULT 'infographic',
    confidence_score REAL DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    ocr_text TEXT,
    ocr_confidence REAL DEFAULT 0.0 CHECK (ocr_confidence >= 0 AND ocr_confidence <= 1),
    ocr_engines_used JSONB DEFAULT '[]'::jsonb,
    ocr_results JSONB DEFAULT '{}'::jsonb,
    processed_at TIMESTAMP,
    processing_duration REAL,
    storage_path VARCHAR(1024),
    file_size INTEGER,
    file_hash VARCHAR(64)
);

-- Dataset items table
CREATE TABLE IF NOT EXISTS dataset_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    instruction TEXT NOT NULL,
    response TEXT NOT NULL,
    instruction_type VARCHAR(50) DEFAULT 'description',
    quality_score REAL NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
    human_validated BOOLEAN DEFAULT FALSE,
    validation_score REAL CHECK (validation_score IS NULL OR (validation_score >= 0 AND validation_score <= 1)),
    split VARCHAR(20) DEFAULT 'train' CHECK (split IN ('train', 'validation', 'test')),
    dataset_version VARCHAR(50) DEFAULT 'v1.0',
    source_language VARCHAR(10) DEFAULT 'en',
    target_language VARCHAR(10) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    generation_model VARCHAR(100),
    generation_params JSONB DEFAULT '{}'::jsonb,
    context JSONB DEFAULT '{}'::jsonb,
    UNIQUE(document_id, image_id, instruction, target_language)
);

-- Training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    base_model VARCHAR(200) NOT NULL,
    model_path VARCHAR(1024),
    processor_path VARCHAR(1024),
    languages JSONB NOT NULL,
    dataset_version VARCHAR(50) NOT NULL,
    training_config JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'stopped')),
    total_steps INTEGER DEFAULT 0 CHECK (total_steps >= 0),
    current_step INTEGER DEFAULT 0 CHECK (current_step >= 0 AND current_step <= total_steps),
    train_loss REAL,
    eval_loss REAL,
    eval_bleu REAL,
    eval_rouge REAL,
    eval_bertscore REAL,
    metrics_history JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    gpu_hours REAL,
    peak_memory_gb REAL,
    wandb_run_id VARCHAR(100),
    wandb_project VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for performance

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_source_language ON documents(source, language);
CREATE INDEX IF NOT EXISTS idx_documents_scraped_at ON documents(scraped_at);
CREATE INDEX IF NOT EXISTS idx_documents_quality_score ON documents(quality_score);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_publication_date ON documents(publication_date);

-- Full-text search index on documents
CREATE INDEX IF NOT EXISTS idx_documents_title_gin ON documents USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_documents_content_gin ON documents USING gin(to_tsvector('english', content));

-- Images indexes
CREATE INDEX IF NOT EXISTS idx_images_document_id ON images(document_id);
CREATE INDEX IF NOT EXISTS idx_images_type ON images(image_type);
CREATE INDEX IF NOT EXISTS idx_images_confidence ON images(confidence_score);
CREATE INDEX IF NOT EXISTS idx_images_ocr_confidence ON images(ocr_confidence);
CREATE INDEX IF NOT EXISTS idx_images_processed_at ON images(processed_at);
CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);

-- Dataset items indexes
CREATE INDEX IF NOT EXISTS idx_dataset_items_split ON dataset_items(split);
CREATE INDEX IF NOT EXISTS idx_dataset_items_quality ON dataset_items(quality_score);
CREATE INDEX IF NOT EXISTS idx_dataset_items_language ON dataset_items(target_language);
CREATE INDEX IF NOT EXISTS idx_dataset_items_version ON dataset_items(dataset_version);
CREATE INDEX IF NOT EXISTS idx_dataset_items_validated ON dataset_items(human_validated);
CREATE INDEX IF NOT EXISTS idx_dataset_items_created_at ON dataset_items(created_at);
CREATE INDEX IF NOT EXISTS idx_dataset_items_instruction_type ON dataset_items(instruction_type);

-- Training runs indexes
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_dataset_version ON training_runs(dataset_version);
CREATE INDEX IF NOT EXISTS idx_training_runs_started_at ON training_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_training_runs_eval_loss ON training_runs(eval_loss);
CREATE INDEX IF NOT EXISTS idx_training_runs_created_at ON training_runs(created_at);

-- Create update trigger for last_updated column
CREATE OR REPLACE FUNCTION update_last_updated_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_last_updated 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_last_updated_column();

-- Insert some initial data for development/testing
INSERT INTO documents (
    url, title, source, language, content, quality_score, processing_status
) VALUES 
    ('https://example.unhcr.org/sample-report', 'Sample UNHCR Report', 'unhcr', 'en', 
     'This is a sample humanitarian report for testing purposes.', 0.85, 'completed'),
    ('https://example.who.int/health-report', 'WHO Health Report 2024', 'who', 'en',
     'Sample WHO report with health statistics and recommendations.', 0.92, 'completed')
ON CONFLICT (url) DO NOTHING;

-- Create views for common queries

-- High-quality documents view
CREATE OR REPLACE VIEW high_quality_documents AS
SELECT 
    d.*,
    COUNT(i.id) as image_count,
    AVG(i.ocr_confidence) as avg_ocr_confidence
FROM documents d
LEFT JOIN images i ON d.id = i.document_id
WHERE d.quality_score >= 0.8
GROUP BY d.id;

-- Dataset statistics view
CREATE OR REPLACE VIEW dataset_statistics AS
SELECT 
    split,
    target_language,
    COUNT(*) as item_count,
    AVG(quality_score) as avg_quality,
    COUNT(CASE WHEN human_validated THEN 1 END) as validated_count
FROM dataset_items
GROUP BY split, target_language;

-- Training run summary view
CREATE OR REPLACE VIEW training_run_summary AS
SELECT 
    id,
    name,
    status,
    dataset_version,
    languages,
    eval_loss,
    eval_bleu,
    started_at,
    completed_at,
    CASE 
        WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (completed_at - started_at))/3600.0 
        ELSE NULL 
    END as duration_hours
FROM training_runs
ORDER BY created_at DESC;