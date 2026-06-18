import sqlite3
from pathlib import Path


def connect_db(sqlite_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(sqlite_path: str) -> None:
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_db(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                department TEXT NOT NULL,
                category TEXT NOT NULL,
                security_level TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_document(conn, document: dict) -> None:
    conn.execute(
        """
        INSERT INTO documents (
            id, source_path, title, doc_type, department, category,
            security_level, created_at
        )
        VALUES (
            :id, :source_path, :title, :doc_type, :department, :category,
            :security_level, :created_at
        )
        ON CONFLICT(id) DO UPDATE SET
            source_path = excluded.source_path,
            title = excluded.title,
            doc_type = excluded.doc_type,
            department = excluded.department,
            category = excluded.category,
            security_level = excluded.security_level,
            created_at = excluded.created_at
        """,
        document,
    )


def upsert_chunk(conn, chunk: dict) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            id, document_id, chunk_index, text, token_count
        )
        VALUES (
            :id, :document_id, :chunk_index, :text, :token_count
        )
        ON CONFLICT(id) DO UPDATE SET
            document_id = excluded.document_id,
            chunk_index = excluded.chunk_index,
            text = excluded.text,
            token_count = excluded.token_count
        """,
        chunk,
    )


def reset_db(conn) -> None:
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")


def find_candidate_chunk_ids(
    conn,
    doc_type: str | None = None,
    department: str | None = None,
    category: str | None = None,
    security_level: str | None = None,
    source_path: str | None = None,
) -> list[str]:
    filters = [
        ("documents.doc_type", doc_type),
        ("documents.department", department),
        ("documents.category", category),
        ("documents.security_level", security_level),
        ("documents.source_path", source_path),
    ]
    where_clauses = []
    params = []

    for column, value in filters:
        if value is not None:
            where_clauses.append(f"{column} = ?")
            params.append(value)

    sql = """
        SELECT chunks.id
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
    """
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " ORDER BY documents.source_path, chunks.chunk_index, chunks.id"

    rows = conn.execute(sql, params).fetchall()
    return [row[0] for row in rows]
