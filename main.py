from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import os
from contextlib import contextmanager
from typing import Optional, List

import turso_serverless

import numpy as np
import pandas as pd

from pydantic import BaseModel, Field

from linearmodels.panel import (
    PooledOLS,
    PanelOLS,
    RandomEffects,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Database credentials are configured as environment variables by the host.

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="WorldData API",
    description=(
        "API for the WorldData economic, demographic "
        "and fiscal database"
    ),
    version="2.1.0",
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

class _HybridRow:
    """A row that supports both row["column"] and row[0] access."""

    def __init__(self, values, description):
        self._values = tuple(values)
        self._columns = {
            column[0]: index
            for index, column in enumerate(description or ())
        }

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._columns[key]]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _CursorAdapter:
    """Keep the existing API code compatible with DB-API cursor rows."""

    def __init__(self, cursor):
        self._cursor = cursor

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return _HybridRow(row, self._cursor.description)

    def fetchall(self):
        return [
            _HybridRow(row, self._cursor.description)
            for row in self._cursor.fetchall()
        ]

    def __iter__(self):
        for row in self._cursor:
            yield _HybridRow(row, self._cursor.description)


class _ConnectionAdapter:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql, parameters=()):
        return _CursorAdapter(self._connection.execute(sql, parameters))


@contextmanager
def get_connection():
    """
    Open a connection to the remote Turso database.

    Set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in the hosting
    environment. No database file or token is stored in this repository.
    """
    database_url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if not database_url or not auth_token:
        raise RuntimeError(
            "Missing TURSO_DATABASE_URL or TURSO_AUTH_TOKEN environment variable."
        )

    conn = None
    try:
        conn = turso_serverless.connect(
            database_url,
            auth_token=auth_token,
        )
        yield _ConnectionAdapter(conn)
    finally:
        if conn is not None:
            conn.close()


def database_error_message(error: Exception) -> str:
    return f"Database error: {str(error)}"


# ============================================================
# PANEL DATA — REQUEST MODEL
# ============================================================

class PanelRegressionRequest(BaseModel):
    countries: List[str] = Field(
        ...,
        min_length=2,
        max_length=100,
    )

    dependent: str = Field(
        ...,
        min_length=1,
    )

    independents: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
    )

    start_year: Optional[int] = None
    end_year: Optional[int] = None


# ============================================================
# PANEL DATA — HELPERS
# ============================================================

def clean_numeric(value):
    """
    Convert a value to a JSON-safe float.

    Returns None for missing, invalid, infinite,
    or otherwise non-finite values.
    """

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def adjusted_r_squared(
    r_squared,
    observations,
    parameters,
):
    """
    Calculate adjusted R-squared.

    parameters excludes the intercept.
    """

    if (
        r_squared is None
        or observations is None
        or parameters is None
    ):
        return None

    observations = int(observations)
    parameters = int(parameters)

    denominator = observations - parameters - 1

    if observations <= parameters + 1:
        return None

    return clean_numeric(
        1
        - (
            (1 - r_squared)
            * (observations - 1)
            / denominator
        )
    )


def coefficient_table(
    result,
    variables,
):
    """
    Convert a linearmodels result into a JSON-safe
    coefficient table.

    The field names here MUST match the frontend
    PanelData.tsx types:

        coefficient
        std_error
        t_stat
        p_value
        ci_lower
        ci_upper
    """

    output = []

    params = result.params
    std_errors = result.std_errors
    tstats = result.tstats
    pvalues = result.pvalues
    conf_int = result.conf_int()

    for variable in variables:

        if variable not in params.index:
            continue

        coefficient = clean_numeric(
            params[variable]
        )

        std_error = clean_numeric(
            std_errors[variable]
        )

        t_statistic = clean_numeric(
            tstats[variable]
        )

        p_value = clean_numeric(
            pvalues[variable]
        )

        lower = clean_numeric(
            conf_int.loc[variable].iloc[0]
        )

        upper = clean_numeric(
            conf_int.loc[variable].iloc[1]
        )

        output.append(
            {
                "variable": variable,
                "coefficient": coefficient,
                "std_error": std_error,

                # Frontend expects "t_stat"
                "t_stat": t_statistic,

                "p_value": p_value,

                # Frontend expects these two fields
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    return output


def model_statistics(
    result,
    number_of_parameters,
):
    """
    Extract common statistics from a linearmodels
    regression result.
    """

    observations = int(
        getattr(result, "nobs", 0)
    )

    r_squared = clean_numeric(
        getattr(result, "rsquared", None)
    )

    return {
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared(
            r_squared,
            observations,
            number_of_parameters,
        ),
        "observations": observations,
    }


def hausman_test(
    fixed_result,
    random_result,
    variables,
):
    """
    Hausman specification test.

    H0:
        Random Effects is consistent.

    H1:
        Fixed Effects is preferred.

    A p-value below 0.05 leads to Fixed Effects.
    """

    try:

        beta_fe = fixed_result.params[
            variables
        ]

        beta_re = random_result.params[
            variables
        ]

        covariance_fe = fixed_result.cov.loc[
            variables,
            variables,
        ]

        covariance_re = random_result.cov.loc[
            variables,
            variables,
        ]

        difference = (
            beta_fe - beta_re
        )

        covariance_difference = (
            covariance_fe - covariance_re
        )

        # Ensure symmetry
        covariance_difference = (
            covariance_difference
            + covariance_difference.T
        ) / 2

        try:

            inverse = np.linalg.inv(
                covariance_difference.values
            )

        except np.linalg.LinAlgError:

            # Moore-Penrose pseudoinverse if
            # covariance difference is singular
            inverse = np.linalg.pinv(
                covariance_difference.values
            )

        statistic = float(
            difference.values.T
            @ inverse
            @ difference.values
        )

        # Numerical rounding can theoretically
        # produce a tiny negative value.
        statistic = max(
            0.0,
            statistic,
        )

        degrees_of_freedom = len(
            variables
        )

        from scipy.stats import chi2

        p_value = float(
            1
            - chi2.cdf(
                statistic,
                degrees_of_freedom,
            )
        )

        if p_value < 0.05:

            preferred_model = (
                "fixed_effects"
            )

        else:

            preferred_model = (
                "random_effects"
            )

        return {
            "statistic": clean_numeric(
                statistic
            ),
            "degrees_of_freedom": (
                degrees_of_freedom
            ),
            "p_value": clean_numeric(
                p_value
            ),
            "preferred_model": (
                preferred_model
            ),
            "significance_level": 0.05,
            "null_hypothesis": (
                "Random Effects is consistent."
            ),
            "alternative_hypothesis": (
                "Fixed Effects is preferred."
            ),
        }

    except Exception as error:

        return {
            "statistic": None,
            "degrees_of_freedom": len(
                variables
            ),
            "p_value": None,
            "preferred_model": None,
            "significance_level": 0.05,
            "null_hypothesis": (
                "Random Effects is consistent."
            ),
            "alternative_hypothesis": (
                "Fixed Effects is preferred."
            ),
            "error": str(error),
        }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "WorldData API",
        "version": "2.1.0",
        "status": "online",
        "database": "Turso",
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
            "database_type": "Turso",
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
        500,
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
        le=2000,
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
                "indicator_id": index + 1,
                "code": row["dataset_name"],
                "name": row["dataset_name"],
                "description": row["description"],
                "unit": row["unit"],
                "frequency": "Annual",
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
# PIT — PERSONAL INCOME TAX BRACKETS
# ============================================================

@app.get("/pit")
def pit(
    country: Optional[str] = None,
    currency: Optional[str] = Query(
        None,
        pattern="^(ORIGINAL|USD|EUR)$",
    ),
    year: Optional[int] = None,
):

    query = """
        SELECT
            country_name,
            year,
            bracket,
            lower_bound,
            upper_bound,
            tax_rate,
            currency,
            original_currency
        FROM pit_brackets
        WHERE 1 = 1
    """

    params = []

    if country is not None:
        query += """
            AND country_name = ?
        """
        params.append(country)

    if currency is not None:
        query += """
            AND currency = ?
        """
        params.append(currency)

    if year is not None:
        query += """
            AND year = ?
        """
        params.append(year)

    query += """
        ORDER BY
            country_name,
            year,
            currency,
            bracket
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
                detail="No PIT data found for the requested filters.",
            )

        return {
            "count": len(rows),
            "results": [
                {
                    "country": row["country_name"],
                    "year": row["year"],
                    "bracket": row["bracket"],
                    "lower_bound": row["lower_bound"],
                    "upper_bound": row["upper_bound"],
                    "tax_rate": row["tax_rate"],
                    "currency": row["currency"],
                    "original_currency": row["original_currency"],
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
# PIT — SINGLE COUNTRY
# ============================================================

@app.get("/pit/{country}")
def pit_country(
    country: str,
    currency: str = Query(
        "ORIGINAL",
        pattern="^(ORIGINAL|USD|EUR)$",
    ),
    year: int = Query(
        2024,
        ge=1900,
        le=2100,
    ),
):

    try:
        with get_connection() as conn:

            rows = conn.execute(
                """
                SELECT
                    country_name,
                    year,
                    bracket,
                    lower_bound,
                    upper_bound,
                    tax_rate,
                    currency,
                    original_currency
                FROM pit_brackets
                WHERE country_name = ?
                  AND currency = ?
                  AND year = ?
                ORDER BY bracket
                """,
                (
                    country,
                    currency,
                    year,
                ),
            ).fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No PIT data found for "
                    f"'{country}' in {year} "
                    f"with currency '{currency}'."
                ),
            )

        return {
            "country": country,
            "year": year,
            "currency": currency,
            "original_currency": rows[0]["original_currency"],
            "bracket_count": len(rows),
            "brackets": [
                {
                    "bracket": row["bracket"],
                    "lower_bound": row["lower_bound"],
                    "upper_bound": row["upper_bound"],
                    "tax_rate": row["tax_rate"],
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
# PCR — SOCIAL CONTRIBUTION RATES
# ============================================================

@app.get("/pcr")
def pcr(
    country: Optional[str] = None,
    contributor: Optional[str] = Query(
        None,
        pattern="^(Employer|Employee)$",
    ),
    year: Optional[int] = None,
):

    query = """
        SELECT
            country_name,
            year,
            contributor,
            contribution_type,
            rate
        FROM social_contributions
        WHERE 1 = 1
    """

    params = []

    if country is not None:
        query += """
            AND country_name = ?
        """
        params.append(country)

    if contributor is not None:
        query += """
            AND contributor = ?
        """
        params.append(contributor)

    if year is not None:
        query += """
            AND year = ?
        """
        params.append(year)

    query += """
        ORDER BY
            country_name,
            year,
            contributor,
            contribution_type
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
                detail="No PCR data found for the requested filters.",
            )

        return {
            "count": len(rows),
            "results": [
                {
                    "country": row["country_name"],
                    "year": row["year"],
                    "contributor": row["contributor"],
                    "contribution_type": row["contribution_type"],
                    "rate": row["rate"],
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
# PCR — SINGLE COUNTRY
# ============================================================

@app.get("/pcr/{country}")
def pcr_country(
    country: str,
    contributor: Optional[str] = Query(
        None,
        pattern="^(Employer|Employee)$",
    ),
    year: int = Query(
        2024,
        ge=1900,
        le=2100,
    ),
):

    query = """
        SELECT
            country_name,
            year,
            contributor,
            contribution_type,
            rate
        FROM social_contributions
        WHERE country_name = ?
          AND year = ?
    """

    params = [
        country,
        year,
    ]

    if contributor is not None:
        query += """
            AND contributor = ?
        """
        params.append(contributor)

    query += """
        ORDER BY
            contributor,
            contribution_type
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
                    f"No PCR data found for "
                    f"'{country}' in {year}."
                ),
            )

        return {
            "country": country,
            "year": year,
            "contributor": contributor,
            "count": len(rows),
            "contributions": [
                {
                    "contributor": row["contributor"],
                    "contribution_type": row["contribution_type"],
                    "rate": row["rate"],
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
# PANEL DATA — REGRESSION
# ============================================================

@app.post("/panel/regression")
def panel_regression(
    request: PanelRegressionRequest,
):
    """
    Estimate a panel-data regression using:

        1. Pooled OLS
        2. Fixed Effects
        3. Random Effects
        4. Hausman specification test

    The panel is constructed from dataset_values using:

        country_name + year

    as the panel index.

    Standard errors are clustered by country.
    """

    # --------------------------------------------------------
    # VALIDATE REQUEST
    # --------------------------------------------------------

    countries = [
        country.strip()
        for country in request.countries
        if country.strip()
    ]

    dependent = request.dependent.strip()

    independents = [
        indicator.strip()
        for indicator in request.independents
        if indicator.strip()
    ]

    if len(countries) < 2:

        raise HTTPException(
            status_code=400,
            detail=(
                "Panel regression requires at least "
                "two countries."
            ),
        )

    if not dependent:

        raise HTTPException(
            status_code=400,
            detail=(
                "A dependent variable is required."
            ),
        )

    if not independents:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one independent variable "
                "is required."
            ),
        )

    if dependent in independents:

        raise HTTPException(
            status_code=400,
            detail=(
                "The dependent variable cannot also "
                "be an independent variable."
            ),
        )

    if (
        request.start_year is not None
        and request.end_year is not None
        and request.start_year > request.end_year
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "start_year cannot be greater "
                "than end_year."
            ),
        )

    # Remove duplicate countries while preserving order.
    countries = list(
        dict.fromkeys(countries)
    )

    # Remove duplicate independent variables.
    independents = list(
        dict.fromkeys(independents)
    )

    indicators = [
        dependent,
        *independents,
    ]

    # --------------------------------------------------------
    # BUILD SQL QUERY
    # --------------------------------------------------------

    country_placeholders = ",".join(
        ["?"] * len(countries)
    )

    indicator_placeholders = ",".join(
        ["?"] * len(indicators)
    )

    query = f"""
        SELECT
            country_name,
            year,
            dataset_name,
            value
        FROM dataset_values
        WHERE country_name IN (
            {country_placeholders}
        )
        AND dataset_name IN (
            {indicator_placeholders}
        )
        AND value IS NOT NULL
    """

    params = [
        *countries,
        *indicators,
    ]

    if request.start_year is not None:

        query += """
            AND year >= ?
        """

        params.append(
            request.start_year
        )

    if request.end_year is not None:

        query += """
            AND year <= ?
        """

        params.append(
            request.end_year
        )

    query += """
        ORDER BY
            country_name,
            year,
            dataset_name
    """

    # --------------------------------------------------------
    # READ SQLITE
    # --------------------------------------------------------

    try:

        with get_connection() as conn:

            rows = conn.execute(
                query,
                params,
            ).fetchall()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=database_error_message(
                error
            ),
        )

    if not rows:

        raise HTTPException(
            status_code=404,
            detail=(
                "No data found for the requested "
                "countries, indicators and years."
            ),
        )

    # --------------------------------------------------------
    # LONG DATAFRAME
    # --------------------------------------------------------

    raw_data = [
        {
            "country": row["country_name"],
            "year": int(row["year"]),
            "indicator": row["dataset_name"],
            "value": row["value"],
        }
        for row in rows
    ]

    df_long = pd.DataFrame(
        raw_data
    )

    df_long["value"] = pd.to_numeric(
        df_long["value"],
        errors="coerce",
    )

    df_long = df_long.dropna(
        subset=["value"]
    )

    if df_long.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "The selected data contains no "
                "valid numeric observations."
            ),
        )

    # --------------------------------------------------------
    # CONVERT LONG DATA TO PANEL-WIDE DATA
    # --------------------------------------------------------

    df = df_long.pivot_table(
        index=[
            "country",
            "year",
        ],
        columns="indicator",
        values="value",
        aggfunc="mean",
    ).reset_index()

    df.columns.name = None

    required_columns = [
        dependent,
        *independents,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    "Some requested indicators "
                    "were not found."
                ),
                "missing_indicators": (
                    missing_columns
                ),
            },
        )

    # --------------------------------------------------------
    # COMPLETE CASES
    # --------------------------------------------------------

    model_df = df[
        [
            "country",
            "year",
            *required_columns,
        ]
    ].dropna()

    if model_df.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                "There are no complete country-year "
                "observations containing the dependent "
                "variable and all independent variables."
            ),
        )

    # --------------------------------------------------------
    # SORT PANEL
    # --------------------------------------------------------

    model_df = model_df.sort_values(
        [
            "country",
            "year",
        ]
    )

    # --------------------------------------------------------
    # REMOVE COUNTRIES WITH TOO FEW OBSERVATIONS
    # --------------------------------------------------------

    country_observation_counts = (
        model_df
        .groupby("country")
        .size()
    )

    countries_with_insufficient_data = (
        country_observation_counts[
            country_observation_counts < 2
        ]
        .index
        .tolist()
    )

    if countries_with_insufficient_data:

        model_df = model_df[
            ~model_df["country"].isin(
                countries_with_insufficient_data
            )
        ]

    if (
        model_df["country"].nunique()
        < 2
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "After removing countries with "
                "insufficient observations, fewer "
                "than two countries remain."
            ),
        )

    # --------------------------------------------------------
    # PANEL INDEX
    # --------------------------------------------------------

    panel_df = model_df.set_index(
        [
            "country",
            "year",
        ]
    )

    y = panel_df[
        dependent
    ]

    X = panel_df[
        independents
    ]

    # --------------------------------------------------------
    # BASIC PANEL INFORMATION
    # --------------------------------------------------------

    countries_used = (
        model_df["country"]
        .drop_duplicates()
        .tolist()
    )

    number_of_countries = int(
        len(countries_used)
    )

    number_of_years = int(
        model_df["year"].nunique()
    )

    number_of_observations = int(
        len(model_df)
    )

    # Check whether every country has exactly
    # the same set of years.
    country_year_sets = (
        model_df
        .groupby("country")["year"]
        .apply(
            lambda s: set(
                s.tolist()
            )
        )
    )

    balanced = (
        len(country_year_sets) > 0
        and len(
            {
                tuple(sorted(years))
                for years in country_year_sets
            }
        ) == 1
    )

    actual_start_year = int(
        model_df["year"].min()
    )

    actual_end_year = int(
        model_df["year"].max()
    )

    # --------------------------------------------------------
    # POOLED OLS
    # --------------------------------------------------------

    try:

        # Pooled OLS gets an intercept.
        X_pooled = X.copy()

        X_pooled.insert(
            0,
            "const",
            1.0,
        )

        pooled_model = PooledOLS(
            y,
            X_pooled,
        )

        pooled_result = pooled_model.fit(
            cov_type="clustered",
            cluster_entity=True,
        )

        pooled_statistics = model_statistics(
            pooled_result,
            len(independents),
        )

        pooled_coefficients = (
            coefficient_table(
                pooled_result,
                independents,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Pooled OLS failed."
                ),
                "error": str(error),
            },
        )

    # --------------------------------------------------------
    # FIXED EFFECTS
    # --------------------------------------------------------

    try:

        fixed_model = PanelOLS(
            y,
            X,
            entity_effects=True,
        )

        fixed_result = fixed_model.fit(
            cov_type="clustered",
            cluster_entity=True,
        )

        fixed_statistics = model_statistics(
            fixed_result,
            len(independents),
        )

        fixed_coefficients = (
            coefficient_table(
                fixed_result,
                independents,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Fixed Effects failed."
                ),
                "error": str(error),
            },
        )

    # --------------------------------------------------------
    # RANDOM EFFECTS
    # --------------------------------------------------------

    try:

        # Random Effects gets an intercept.
        X_random = X.copy()

        X_random.insert(
            0,
            "const",
            1.0,
        )

        random_model = RandomEffects(
            y,
            X_random,
        )

        random_result = random_model.fit(
            cov_type="clustered",
            cluster_entity=True,
        )

        random_statistics = model_statistics(
            random_result,
            len(independents),
        )

        random_coefficients = (
            coefficient_table(
                random_result,
                independents,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Random Effects failed."
                ),
                "error": str(error),
            },
        )

    # --------------------------------------------------------
    # HAUSMAN TEST
    # --------------------------------------------------------

    hausman = hausman_test(
        fixed_result,
        random_result,
        independents,
    )

    # --------------------------------------------------------
    # MODEL RECOMMENDATION
    # --------------------------------------------------------

    preferred_model = (
        hausman.get(
            "preferred_model"
        )
    )

    if preferred_model == "fixed_effects":

        recommendation = (
            "Fixed Effects is preferred "
            "according to the Hausman test."
        )

    elif preferred_model == "random_effects":

        recommendation = (
            "Random Effects is preferred "
            "according to the Hausman test."
        )

    else:

        recommendation = (
            "The Hausman test could not "
            "determine a preferred model."
        )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "panel": {
            "countries": number_of_countries,
            "observations": number_of_observations,
            "years": number_of_years,

            "countries_requested": countries,
            "countries_used": countries_used,

            "start_year": actual_start_year,
            "end_year": actual_end_year,

            "balanced": balanced,

            "dependent": dependent,
            "independents": independents,
        },

        "pooled_ols": {
            **pooled_statistics,
            "coefficients": (
                pooled_coefficients
            ),
        },

        "fixed_effects": {
            **fixed_statistics,
            "coefficients": (
                fixed_coefficients
            ),
        },

        "random_effects": {
            **random_statistics,
            "coefficients": (
                random_coefficients
            ),
        },

        "hausman_test": hausman,

        "recommendation": {
            "preferred_model": (
                preferred_model
            ),
            "text": recommendation,
        },
    }


# ============================================================
# START LOCAL SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
