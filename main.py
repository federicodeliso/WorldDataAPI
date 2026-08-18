from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="WorldData API",
    description="API for the WorldData economic, demographic and fiscal database",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.getenv("WORLD_DATA_DB_HOST"),
    "port": int(os.getenv("WORLD_DATA_DB_PORT", "5432")),
    "dbname": os.getenv("WORLD_DATA_DB_NAME"),
    "user": os.getenv("WORLD_DATA_DB_USER"),
    "password": os.getenv("WORLD_DATA_DB_PASSWORD"),
    "sslmode": os.getenv("WORLD_DATA_DB_SSLMODE", "require")
}


pool = ConnectionPool(
    conninfo=(
        f"host={DB_CONFIG['host']} "
        f"port={DB_CONFIG['port']} "
        f"dbname={DB_CONFIG['dbname']} "
        f"user={DB_CONFIG['user']} "
        f"password={DB_CONFIG['password']} "
        f"sslmode={DB_CONFIG['sslmode']}"
    ),
    min_size=1,
    max_size=10,
    open=True
)


def get_connection():
    return pool.connection()


@app.get("/")
def root():
    return {
        "name": "WorldData API",
        "version": "1.0.0",
        "status": "online"
    }



@app.get("/health")
def health():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

        return {
            "status": "healthy",
            "database": "connected"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )


@app.get("/stats")
def stats():
    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute("SELECT COUNT(*) FROM entities")
            entities = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM indicators")
            indicators = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM observations")
            observations = cur.fetchone()[0]

            cur.execute("SELECT MIN(year), MAX(year) FROM observations")
            min_year, max_year = cur.fetchone()

    return {
        "entities": entities,
        "indicators": indicators,
        "observations": observations,
        "min_year": min_year,
        "max_year": max_year
    }


@app.get("/countries")
def countries(
    limit: int = Query(50, ge=1, le=500)
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, name
                FROM entities
                ORDER BY name
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()

    return [
        {
            "entity_id": row[0],
            "name": row[1]
        }
        for row in rows
    ]


@app.get("/indicators")
def indicators(
    limit: int = Query(100, ge=1, le=500)
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    indicator_id,
                    code,
                    name,
                    description,
                    unit,
                    frequency,
                    category
                FROM indicators
                ORDER BY code
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()

    return [
        {
            "indicator_id": row[0],
            "code": row[1],
            "name": row[2],
            "description": row[3],
            "unit": row[4],
            "frequency": row[5],
            "category": row[6]
        }
        for row in rows
    ]

@app.get("/indicators/{code}")
def indicator_detail(code: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    i.indicator_id,
                    i.code,
                    i.name,
                    i.description,
                    i.unit,
                    i.frequency,
                    i.category,
                    s.name AS source_name,
                    s.organization AS source_organization,
                    s.url AS source_url
                FROM indicators i
                LEFT JOIN sources s
                    ON i.source_id = s.source_id
                WHERE i.code = %s
            """, (code,))

            row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Indicator '{code}' not found."
        )

    return {
        "indicator_id": row[0],
        "code": row[1],
        "name": row[2],
        "description": row[3],
        "unit": row[4],
        "frequency": row[5],
        "category": row[6],
        "source": {
            "name": row[7],
            "organization": row[8],
            "url": row[9]
        }
    }


@app.get("/countries/{country}")
def country_detail(country: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    e.entity_id,
                    e.name,
                    e.iso2,
                    e.iso3,
                    e.region,
                    e.subregion,
                    e.income_group
                FROM entities e
                WHERE e.name = %s
            """, (country,))

            row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Country or entity '{country}' not found."
        )

    return {
        "entity_id": row[0],
        "name": row[1],
        "iso2": row[2],
        "iso3": row[3],
        "region": row[4],
        "subregion": row[5],
        "income_group": row[6]
    }


@app.get("/categories")
def categories():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    category,
                    COUNT(*) AS indicator_count
                FROM indicators
                WHERE category IS NOT NULL
                  AND category <> ''
                GROUP BY category
                ORDER BY category
            """)

            rows = cur.fetchall()

    return [
        {
            "category": row[0],
            "indicator_count": row[1]
        }
        for row in rows
    ]



@app.get("/data/{country}/{indicator}")
def data(
    country: str,
    indicator: str,
    start_year: int | None = None,
    end_year: int | None = None
):
    query = """
        SELECT
            e.name AS country,
            i.code AS indicator,
            o.year,
            o.value
        FROM observations o
        JOIN entities e
            ON o.entity_id = e.entity_id
        JOIN indicators i
            ON o.indicator_id = i.indicator_id
        WHERE e.name = %s
          AND i.code = %s
    """

    params = [country, indicator]

    if start_year is not None:
        query += " AND o.year >= %s"
        params.append(start_year)

    if end_year is not None:
        query += " AND o.year <= %s"
        params.append(end_year)

    query += " ORDER BY o.year"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No data found for this country and indicator."
        )

    return [
        {
            "country": row[0],
            "indicator": row[1],
            "year": row[2],
            "value": row[3]
        }
        for row in rows
    ]

@app.get("/rankings/{indicator}")
def rankings(
    indicator: str,
    year: int,
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(20, ge=1, le=500)
):
    sort_order = "DESC" if order == "desc" else "ASC"

    query = f"""
        SELECT
            e.name AS country,
            e.iso3,
            o.year,
            o.value
        FROM observations o
        JOIN entities e
            ON o.entity_id = e.entity_id
        JOIN indicators i
            ON o.indicator_id = i.indicator_id
        WHERE i.code = %s
          AND o.year = %s
          AND o.value IS NOT NULL
        ORDER BY o.value {sort_order}
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (indicator, year, limit))
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for indicator '{indicator}' in {year}."
        )

    return {
        "indicator": indicator,
        "year": year,
        "order": order,
        "count": len(rows),
        "results": [
            {
                "rank": index + 1,
                "country": row[0],
                "iso3": row[1],
                "year": row[2],
                "value": row[3]
            }
            for index, row in enumerate(rows)
        ]
    }

@app.get("/latest/{indicator}")
def latest(
    indicator: str,
    limit: int = Query(200, ge=1, le=500)
):
    query = """
        SELECT
            e.name AS country,
            e.iso3,
            o.year,
            o.value
        FROM observations o
        JOIN entities e
            ON o.entity_id = e.entity_id
        JOIN indicators i
            ON o.indicator_id = i.indicator_id
        WHERE i.code = %s
          AND o.value IS NOT NULL
          AND o.year = (
              SELECT MAX(o2.year)
              FROM observations o2
              WHERE o2.entity_id = o.entity_id
                AND o2.indicator_id = o.indicator_id
                AND o2.value IS NOT NULL
          )
        ORDER BY e.name
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (indicator, limit))
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for indicator '{indicator}'."
        )

    latest_year = max(row[2] for row in rows)

    return {
        "indicator": indicator,
        "year": latest_year,
        "count": len(rows),
        "results": [
            {
                "country": row[0],
                "iso3": row[1],
                "year": row[2],
                "value": row[3]
            }
            for row in rows
        ]
    }


@app.get("/compare/{indicator}")
def compare(
    indicator: str,
    countries: str,
    start_year: int | None = None,
    end_year: int | None = None
):
    country_list = [
        country.strip()
        for country in countries.split(",")
        if country.strip()
    ]

    if not country_list:
        raise HTTPException(
            status_code=400,
            detail="At least one country/entity must be provided."
        )

    query = """
        SELECT
            e.name AS country,
            e.iso3,
            o.year,
            o.value
        FROM observations o
        JOIN entities e
            ON o.entity_id = e.entity_id
        JOIN indicators i
            ON o.indicator_id = i.indicator_id
        WHERE i.code = %s
          AND e.name = ANY(%s)
    """

    params = [indicator, country_list]

    if start_year is not None:
        query += " AND o.year >= %s"
        params.append(start_year)

    if end_year is not None:
        query += " AND o.year <= %s"
        params.append(end_year)

    query += " ORDER BY e.name, o.year"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No data found for the requested comparison."
        )

    return {
        "indicator": indicator,
        "countries_requested": country_list,
        "start_year": start_year,
        "end_year": end_year,
        "count": len(rows),
        "results": [
            {
                "country": row[0],
                "iso3": row[1],
                "year": row[2],
                "value": row[3]
            }
            for row in rows
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)