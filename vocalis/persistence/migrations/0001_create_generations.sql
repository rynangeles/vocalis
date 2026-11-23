-- Generation history. Paths are stored as text. Audio bytes stay on disk.

CREATE TABLE generations (
    id TEXT PRIMARY KEY,
    generation_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_text TEXT NOT NULL,
    voice_description TEXT,
    reference_audio_path TEXT,
    reference_transcript TEXT,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    output_audio_path TEXT,
    sample_rate INTEGER,
    channels INTEGER,
    sample_width_bytes INTEGER,
    frame_count INTEGER,
    duration_seconds REAL,
    audio_format TEXT,
    simulated INTEGER,
    audio_disclaimer TEXT,
    generation_seconds REAL,
    error_message TEXT,
    CHECK (generation_type IN ('voice_design', 'voice_cloning')),
    CHECK (provider IN ('mock', 'voxcpm')),
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    CHECK (simulated IS NULL OR simulated IN (0, 1))
);

CREATE INDEX idx_generations_created_at ON generations (created_at, id);

CREATE INDEX idx_generations_status ON generations (status);

CREATE INDEX idx_generations_type ON generations (generation_type);

CREATE INDEX idx_generations_provider ON generations (provider);
