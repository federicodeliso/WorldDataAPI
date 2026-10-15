
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3

app = FastAPI(
    title="WorldData API",
    description="API for the WorldData economic, demographic and fiscal database",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://world-data-frontend.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set WORLD_DATA_DB_PATH to the absolute path of worlddata.db.
# Default: worlddata.db in the same directory as this file.
DB_PATH = Path(
    os.getenv(
        "WORLD_DATA_DB_PATH",
        str(Path(__file__).resolve().parent / "worlddata.db"),
    )
).resolve()


@contextmanager
def get_connection():
    if not DB_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "SQLite database file not found. "
                "Configure WORLD_DATA_DB_PATH and ensure the file "
                "is available to the running service."
            ),
        )

    conn = None
    try:
        conn = sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        )
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SQLite database error: {exc}",
        )
    finally:
        if conn is not None:
            conn.close()


def rows_as_dicts(rows):
    return [dict(row) for row in rows]


@app.get("/")
def root():
    return {
        "name": "WorldData API",
        "version": "1.1.0",
        "status": "online",
    }


@app.get("/health")
def health():
    with get_connection() as conn:
        conn.execute("SELECT 1").fetchone()
    return {
        "status": "healthy",
        "database": "connected",
        "database_type": "SQLite",
    }


@app.get("/schema")
def schema():
    """List the actual tables in the SQLite database."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

    return {"database": DB_PATH.name, "tables": rows_as_dicts(rows)}


@app.get("/schema/{table}")
def table_schema(table: str):
    """Show columns for a table, useful for mapping your actual schema."""
    with get_connection() as conn:
        exists = conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type IN ('table', 'view') AND name = ?
            """,
            (table,),
        ).fetchone()

        if not exists:
            raise HTTPException(status_code=404, detail="Table not found.")

        # PRAGMA cannot bind a table name; verify it exists above and quote it.
        safe_table = '"' + table.replace('"', '""') + '"'
        rows = conn.execute(
            f"PRAGMA table_info({safe_table})"
        ).fetchall()

    return {"table": table, "columns": rows_as_dicts(rows)}


@app.get("/stats")
def stats():
    # These table names must exist in the SQLite file.
    with get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute("""
                SELECT name FROM sqlite_master WHERE type = 'table'
            """).fetchall()
        }

        required = {"entities", "indicators", "observations"}
        missing = required - tables
        if missing:
            raise HTTPException(
                status_code=501,
                detail=(
                    "The SQLite schema differs from the expected API schema. "
                    f"Missing tables: {', '.join(sorted(missing))}. "
                    "Use /schema to inspect the database."
                ),
            )

        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        indicators = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
        observations = conn.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        years = conn.execute(
            "SELECT MIN(year), MAX(year) FROM observations"
        ).fetchone()

    return {
        "entities": entities,
        "indicators": indicators,
        "observations": observations,
        "min_year": years[0],
        "max_year": years[1],
    }


@app.get("/countries")
def countries(limit: int = Query(50, ge=1, le=500)):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT entity_id, name
            FROM entities
            ORDER BY name
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return rows_as_dicts(rows)


@app.get("/indicators")
def indicators(limit: int = Query(100, ge=1, le=2000)):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT indicator_id, code, name, description,
                   unit, frequency, category
            FROM indicators
            ORDER BY code
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return rows_as_dicts(rows)


@app.get("/indicators/{code}")
def indicator_detail(code: str):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT i.indicator_id, i.code, i.name, i.description,
                   i.unit, i.frequency, i.category,
                   s.name AS source_name,
                   s.organization AS source_organization,
                   s.url AS source_url
            FROM indicators i
            LEFT JOIN sources s ON i.source_id = s.source_id
            WHERE i.code = ?
            """,
            (code,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Indicator not found.")

    result = dict(row)
    result["source"] = {
        "name": result.pop("source_name"),
        "organization": result.pop("source_organization"),
        "url": result.pop("source_url"),
    }
    return result


@app.get("/countries/{country}")
def country_detail(country: str):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT entity_id, name, iso2, iso3, region,
                   subregion, income_group
            FROM entities
            WHERE name = ?
            """,
            (country,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Country/entity not found.")

    return dict(row)


@app.get("/categories")
def categories():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT category, COUNT(*) AS indicator_count
            FROM indicators
            WHERE category IS NOT NULL AND category <> ''
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()

    return rows_as_dicts(rows)


@app.get("/data/{country}/{indicator}")
def data(
    country: str,
    indicator: str,
    start_year: int | None = None,
    end_year: int | None = None,
):
    query = """
        SELECT e.name AS country, i.code AS indicator, o.year, o.value
        FROM observations o
        JOIN entities e ON o.entity_id = e.entity_id
        JOIN indicators i ON o.indicator_id = i.indicator_id
        WHERE e.name = ? AND i.code = ?
    """
    params = [country, indicator]

    if start_year is not None:
        query += " AND o.year >= ?"
        params.append(start_year)
    if end_year is not None:
        query += " AND o.year <= ?"
        params.append(end_year)

    query += " ORDER BY o.year"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No data found for this country and indicator.",
        )

    return rows_as_dicts(rows)


@app.get("/rankings/{indicator}")
def rankings(
    indicator: str,
    year: int,
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(20, ge=1, le=500),
):
    sort_order = "DESC" if order == "desc" else "ASC"

    query = f"""
        SELECT e.name AS country, e.iso3, o.year, o.value
        FROM observations o
        JOIN entities e ON o.entity_id = e.entity_id
        JOIN indicators i ON o.indicator_id = i.indicator_id
        WHERE i.code = ? AND o.year = ? AND o.value IS NOT NULL
        ORDER BY o.value {sort_order}
        LIMIT ?
    """

    with get_connection() as conn:
        rows = conn.execute(query, (indicator, year, limit)).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for indicator '{indicator}' in {year}.",
        )

    return {
        "indicator": indicator,
        "year": year,
        "order": order,
        "count": len(rows),
        "results": [
            {"rank": index + 1, **dict(row)}
            for index, row in enumerate(rows)
        ],
    }


@app.get("/latest/{indicator}")
def latest(indicator: str, limit: int = Query(200, ge=1, le=500)):
    query = """
        SELECT e.name AS country, e.iso3, o.year, o.value
        FROM observations o
        JOIN entities e ON o.entity_id = e.entity_id
        JOIN indicators i ON o.indicator_id = i.indicator_id
        WHERE i.code = ?
          AND o.value IS NOT NULL
          AND o.year = (
              SELECT MAX(o2.year)
              FROM observations o2
              WHERE o2.entity_id = o.entity_id
                AND o2.indicator_id = o.indicator_id
                AND o2.value IS NOT NULL
          )
        ORDER BY e.name
        LIMIT ?
    """

    with get_connection() as conn:
        rows = conn.execute(query, (indicator, limit)).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for indicator '{indicator}'.",
        )

    results = rows_as_dicts(rows)
    return {
        "indicator": indicator,
        "year": max(row["year"] for row in results),
        "count": len(results),
        "results": results,
    }


@app.get("/compare/{indicator}")
def compare(
    indicator: str,
    countries: str,
    start_year: int | None = None,
    end_year: int | None = None,
):
    country_list = [c.strip() for c in countries.split(",") if c.strip()]
    if not country_list:
        raise HTTPException(
            status_code=400,
            detail="At least one country/entity must be provided.",
        )

    placeholders = ",".join("?" for _ in country_list)
    query = f"""
        SELECT e.name AS country, e.iso3, o.year, o.value
        FROM observations o
        JOIN entities e ON o.entity_id = e.entity_id
        JOIN indicators i ON o.indicator_id = i.indicator_id
        WHERE i.code = ? AND e.name IN ({placeholders})
    """
    params = [indicator, *country_list]

    if start_year is not None
