CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE code_embeddings (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code       TEXT NOT NULL,
    path       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    embedding  VECTOR(768) NOT NULL
);
