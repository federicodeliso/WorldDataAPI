from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import sqlite3
from pathlib import Path
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Expected location:
# C:\Users\Utente\Documents\PROJECTS\WorldData\worlddata.db
#
# For now, keep the database path configurable through an
# environment variable so the backend can later be moved
# without changing the code.

DB_PATH = Path(
    # Windows path for your current local project.
    r"C:\Users\Utente\Documents\PROJECTS\WorldData\worlddata.db"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="WorldData API",
    description=(
        "API for the WorldData economic, demographic "
        "and fiscal database"
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    """
    Open a read-only SQLite connection.

    The API only performs SELECT queries. It never modifies
    worlddata.db.
    """

    if not DB_PATH.exists():
        raise RuntimeError(
            f"WorldData database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    # Tell SQLite this connection is intended for reading only.
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


def database_error_message(error: Exception) -> str:
    return f"Database error: {str(error)}"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "WorldData API",
        "version": "2.0.0",
        "status": "online",
        "database": "SQLite",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 AS ok"
            ).fetchone()

        return {
            "status": "healthy",
            "database": "connected",
            "database_type": "SQLite",
            "database_path": str(DB_PATH),
            "test": row["ok"] == 1,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# STATS
# ============================================================

@app.get("/stats")
def stats():

    try:
        with get_connection() as conn:

            dataset_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM datasets
                """
            ).fetchone()[0]

            observation_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM dataset_values
                """
            ).fetchone()[0]

            country_count = conn.execute(
                """
                SELECT COUNT(DISTINCT country_name)
                FROM dataset_values
                """
            ).fetchone()[0]

            min_year, max_year = conn.execute(
                """
                SELECT
                    MIN(year),
                    MAX(year)
                FROM dataset_values
                """
            ).fetchone()

        return {
            # These names preserve the existing frontend contract.
            "entities": country_count,
            "indicators": dataset_count,
            "observations": observation_count,
            "min_year": min_year,
            "max_year": max_year,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# COUNTRIES
# ============================================================

@app.get("/countries")
def countries(
    limit: int = Query(
        50,
        ge=1,
        le=500,
    )
):

    try:
        with get_connection() as conn:

            rows = conn.execute(
                """
                SELECT DISTINCT
                    country_name
                FROM dataset_values
                WHERE country_name IS NOT NULL
                  AND country_name <> ''
                ORDER BY country_name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "entity_id": index + 1,
                "name": row["country_name"],
            }
            for index, row in enumerate(rows)
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# INDICATORS / DATASETS
# ============================================================

@app.get("/indicators")
def indicators(
    limit: int = Query(
        100,
        ge=1,
        le=500,
    )
):

    try:
        with get_connection() as conn:

            rows = conn.execute(
                """
                SELECT
                    dataset_name,
                    dataset_type,
                    description,
                    unit,
                    source
                FROM datasets
                ORDER BY dataset_name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                # Synthetic ID for compatibility with the
                # existing React application.
                "indicator_id": index + 1,

                # This is the value React uses in /data/{country}/{indicator}.
                "code": row["dataset_name"],

                "name": row["dataset_name"],

                "description": row["description"],

                "unit": row["unit"],

                "frequency": "Annual",

                # The SQLite schema does not have a dedicated
                # category column, so dataset_type is exposed
                # here for API compatibility.
                "category": row["dataset_type"],
            }
            for index, row in enumerate(rows)
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# INDICATOR DETAIL
# ============================================================

@app.get("/indicators/{code}")
def indicator_detail(code: str):

    try:
        with get_connection() as conn:

            row = conn.execute(
                """
                SELECT
                    dataset_name,
                    dataset_type,
                    description,
                    unit,
                    source
                FROM datasets
                WHERE dataset_name = ?
                """,
                (code,),
            ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Indicator '{code}' not found.",
            )

        return {
            "indicator_id": 1,
            "code": row["dataset_name"],
            "name": row["dataset_name"],
            "description": row["description"],
            "unit": row["unit"],
            "frequency": "Annual",
            "category": row["dataset_type"],
            "source": {
                "name": row["source"],
                "organization": row["source"],
                "url": None,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# COUNTRY DETAIL
# ============================================================

@app.get("/countries/{country}")
def country_detail(country: str):

    try:
        with get_connection() as conn:

            row = conn.execute(
                """
                SELECT
                    country_name
                FROM dataset_values
                WHERE country_name = ?
                LIMIT 1
                """,
                (country,),
            ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Country or entity '{country}' not found.",
            )

        return {
            "entity_id": 1,
            "name": row["country_name"],
            "iso2": None,
            "iso3": None,
            "region": None,
            "subregion": None,
            "income_group": None,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# CATEGORIES
# ============================================================

@app.get("/categories")
def categories():

    try:
        with get_connection() as conn:

            rows = conn.execute(
                """
                SELECT
                    dataset_type AS category,
                    COUNT(*) AS indicator_count
                FROM datasets
                WHERE dataset_type IS NOT NULL
                  AND dataset_type <> ''
                GROUP BY dataset_type
                ORDER BY dataset_type
                """
            ).fetchall()

        return [
            {
                "category": row["category"],
                "indicator_count": row["indicator_count"],
            }
            for row in rows
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# DATA
# ============================================================

@app.get("/data/{country}/{indicator}")
def data(
    country: str,
    indicator: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
):

    query = """
        SELECT
            country_name AS country,
            dataset_name AS indicator,
            year,
            value
        FROM dataset_values
        WHERE country_name = ?
          AND dataset_name = ?
    """

    params = [
        country,
        indicator,
    ]

    if start_year is not None:
        query += """
            AND year >= ?
        """
        params.append(start_year)

    if end_year is not None:
        query += """
            AND year <= ?
        """
        params.append(end_year)

    query += """
        ORDER BY year
    """

    try:
        with get_connection() as conn:

            rows = conn.execute(
                query,
                params,
            ).fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No data found for this "
                    "country and indicator."
                ),
            )

        return [
            {
                "country": row["country"],
                "indicator": row["indicator"],
                "year": row["year"],
                "value": row["value"],
            }
            for row in rows
        ]

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# RANKINGS
# ============================================================

@app.get("/rankings/{indicator}")
def rankings(
    indicator: str,
    year: int,
    order: str = Query(
        "desc",
        pattern="^(asc|desc)$",
    ),
    limit: int = Query(
        20,
        ge=1,
        le=500,
    ),
):

    sort_order = "DESC" if order == "desc" else "ASC"

    query = f"""
        SELECT
            country_name AS country,
            year,
            value
        FROM dataset_values
        WHERE dataset_name = ?
          AND year = ?
          AND value IS NOT NULL
        ORDER BY value {sort_order}
        LIMIT ?
    """

    try:
        with get_connection() as conn:

            rows = conn.execute(
                query,
                (
                    indicator,
                    year,
                    limit,
                ),
            ).fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No data found for indicator "
                    f"'{indicator}' in {year}."
                ),
            )

        return {
            "indicator": indicator,
            "year": year,
            "order": order,
            "count": len(rows),
            "results": [
                {
                    "rank": index + 1,
                    "country": row["country"],
                    "iso3": None,
                    "year": row["year"],
                    "value": row["value"],
                }
                for index, row in enumerate(rows)
            ],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# LATEST
# ============================================================

@app.get("/latest/{indicator}")
def latest(
    indicator: str,
    limit: int = Query(
        200,
        ge=1,
        le=500,
    ),
):

    query = """
        SELECT
            dv.country_name AS country,
            dv.year,
            dv.value
        FROM dataset_values dv
        WHERE dv.dataset_name = ?
          AND dv.value IS NOT NULL
          AND dv.year = (
              SELECT MAX(dv2.year)
              FROM dataset_values dv2
              WHERE dv2.country_name = dv.country_name
                AND dv2.dataset_name = dv.dataset_name
                AND dv2.value IS NOT NULL
          )
        ORDER BY dv.country_name
        LIMIT ?
    """

    try:
        with get_connection() as conn:

            rows = conn.execute(
                query,
                (
                    indicator,
                    limit,
                ),
            ).fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No data found for "
                    f"indicator '{indicator}'."
                ),
            )

        latest_year = max(
            row["year"]
            for row in rows
        )

        return {
            "indicator": indicator,
            "year": latest_year,
            "count": len(rows),
            "results": [
                {
                    "country": row["country"],
                    "iso3": None,
                    "year": row["year"],
                    "value": row["value"],
                }
                for row in rows
            ],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# COMPARE
# ============================================================

@app.get("/compare/{indicator}")
def compare(
    indicator: str,
    countries: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
):

    country_list = [
        country.strip()
        for country in countries.split(",")
        if country.strip()
    ]

    if not country_list:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one country/entity "
                "must be provided."
            ),
        )

    placeholders = ",".join(
        ["?"] * len(country_list)
    )

    query = f"""
        SELECT
            country_name AS country,
            year,
            value
        FROM dataset_values
        WHERE dataset_name = ?
          AND country_name IN ({placeholders})
    """

    params = [
        indicator,
        *country_list,
    ]

    if start_year is not None:
        query += """
            AND year >= ?
        """
        params.append(start_year)

    if end_year is not None:
        query += """
            AND year <= ?
        """
        params.append(end_year)

    query += """
        ORDER BY country_name, year
    """

    try:
        with get_connection() as conn:

            rows = conn.execute(
                query,
                params,
            ).fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No data found for the "
                    "requested comparison."
                ),
            )

        return {
            "indicator": indicator,
            "countries_requested": country_list,
            "start_year": start_year,
            "end_year": end_year,
            "count": len(rows),
            "results": [
                {
                    "country": row["country"],
                    "iso3": None,
                    "year": row["year"],
                    "value": row["value"],
                }
                for row in rows
            ],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=database_error_message(e),
        )


# ============================================================
# START LOCAL SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
