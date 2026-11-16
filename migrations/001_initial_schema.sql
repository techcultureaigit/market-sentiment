-- 001_initial_schema.sql
-- Market Sentiment Agent database schema (intraday regime, direction & entry conditions).
-- Conventions: TIMESTAMPTZ everywhere (stored UTC); IST only for session logic.
-- Point-in-time rule: every row that can feed a model carries `available_at`, the earliest
-- moment we could have known it. A prediction with as_of = C reads only rows with
-- available_at <= C.

-- ================================================================ news
CREATE TABLE IF NOT EXISTS raw_articles (
    article_id BIGINT PRIMARY KEY,
    payload    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS articles (
    article_id       BIGINT PRIMARY KEY,
    title            TEXT,
    summary          TEXT,
    body_text        TEXT,
    url              TEXT,
    source           TEXT,
    article_type     TEXT,
    subcategory      TEXT,
    sector           TEXT,
    company          TEXT,
    symbol           TEXT,
    isin             TEXT,
    scrip_code       TEXT,
    vendor_sentiment TEXT,
    published_at     TIMESTAMPTZ NOT NULL,
    ingested_at      TIMESTAMPTZ NOT NULL,
    available_at     TIMESTAMPTZ NOT NULL,
    session_bucket   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_available_at   ON articles (available_at);
CREATE INDEX IF NOT EXISTS idx_articles_published_at   ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_symbol         ON articles (symbol);
CREATE INDEX IF NOT EXISTS idx_articles_session_bucket ON articles (session_bucket);

CREATE TABLE IF NOT EXISTS article_scores (
    article_id      BIGINT REFERENCES articles(article_id) ON DELETE CASCADE,
    scorer_version  TEXT NOT NULL,
    prompt_version  TEXT,
    sentiment_score REAL NOT NULL,          -- -1 .. 1
    confidence      REAL,                   -- 0 .. 1
    nifty_relevance REAL,                   -- 0 .. 1
    sectors         TEXT[],
    macro_theme     TEXT,
    is_event        BOOLEAN NOT NULL DEFAULT FALSE,
    event_type      TEXT,                   -- policy, results, geopolitical, ...
    event_severity  REAL,                   -- 0 .. 1
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (article_id, scorer_version)
);
CREATE INDEX IF NOT EXISTS idx_article_scores_event ON article_scores (is_event) WHERE is_event;

-- ================================================================ calendar
CREATE TABLE IF NOT EXISTS nse_holidays (
    holiday_date DATE PRIMARY KEY,
    description  TEXT
);

CREATE TABLE IF NOT EXISTS expiry_calendar (
    underlying  TEXT NOT NULL,
    expiry_date DATE NOT NULL,
    expiry_kind TEXT NOT NULL CHECK (expiry_kind IN ('weekly', 'monthly')),
    PRIMARY KEY (underlying, expiry_date)
);

-- ================================================================ market data
CREATE TABLE IF NOT EXISTS instruments (
    symbol          TEXT PRIMARY KEY,       -- internal name, e.g. 'NIFTY50'
    vendor_ticker   TEXT NOT NULL,          -- vendor-specific code
    source          TEXT NOT NULL,          -- 'truedata', 'gdfl', 'yfinance', ...
    name            TEXT,
    exchange        TEXT,                   -- NSE, NYSE, ICE, FX
    asset_class     TEXT,                   -- index, equity, volatility, commodity, fx
    timezone        TEXT NOT NULL,          -- IANA tz of the trading session
    session_open    TIME,
    session_close   TIME,
    publish_lag_sec INTEGER NOT NULL DEFAULT 0,
    is_heavyweight  BOOLEAN NOT NULL DEFAULT FALSE,
    index_weight    NUMERIC(8, 5)
);

CREATE TABLE IF NOT EXISTS intraday_bars (
    symbol       TEXT NOT NULL REFERENCES instruments(symbol),
    ts           TIMESTAMPTZ NOT NULL,
    interval_sec INTEGER NOT NULL DEFAULT 60,
    open         NUMERIC(18, 6),
    high         NUMERIC(18, 6),
    low          NUMERIC(18, 6),
    close        NUMERIC(18, 6),
    volume       BIGINT,
    oi           BIGINT,
    available_at TIMESTAMPTZ NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, interval_sec, ts)
);
CREATE INDEX IF NOT EXISTS idx_intraday_bars_available_at ON intraday_bars (available_at);

CREATE TABLE IF NOT EXISTS daily_bars (
    symbol       TEXT NOT NULL REFERENCES instruments(symbol),
    trade_date   DATE NOT NULL,
    open         NUMERIC(18, 6),
    high         NUMERIC(18, 6),
    low          NUMERIC(18, 6),
    close         NUMERIC(18, 6),
    adj_close    NUMERIC(18, 6),
    volume       BIGINT,
    available_at TIMESTAMPTZ NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_bars_available_at ON daily_bars (available_at);

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    underlying    TEXT NOT NULL,
    snapshot_ts   TIMESTAMPTZ NOT NULL,
    expiry_date   DATE NOT NULL,
    strike        NUMERIC(12, 2) NOT NULL,
    option_type   CHAR(2) NOT NULL CHECK (option_type IN ('CE', 'PE')),
    ltp           NUMERIC(14, 4),
    bid           NUMERIC(14, 4),
    ask           NUMERIC(14, 4),
    volume        BIGINT,
    oi            BIGINT,
    oi_change     BIGINT,
    iv            NUMERIC(8, 4),
    underlying_px NUMERIC(14, 4),
    available_at  TIMESTAMPTZ NOT NULL,
    source        TEXT,
    PRIMARY KEY (underlying, snapshot_ts, expiry_date, strike, option_type)
);
CREATE INDEX IF NOT EXISTS idx_option_chain_available_at ON option_chain_snapshots (available_at);

CREATE TABLE IF NOT EXISTS market_breadth (
    universe     TEXT NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    advances     INTEGER,
    declines     INTEGER,
    unchanged    INTEGER,
    available_at TIMESTAMPTZ NOT NULL,
    source       TEXT,
    PRIMARY KEY (universe, ts)
);

CREATE TABLE IF NOT EXISTS daily_series (
    series       TEXT NOT NULL,
    obs_date     DATE NOT NULL,
    vintage      TEXT NOT NULL DEFAULT 'final' CHECK (vintage IN ('provisional', 'final')),
    value        NUMERIC(20, 6),
    source       TEXT,
    available_at TIMESTAMPTZ NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series, obs_date, vintage)
);

-- ================================================================ predictions / outputs
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id      BIGSERIAL PRIMARY KEY,
    instrument         TEXT NOT NULL DEFAULT 'NIFTY50' REFERENCES instruments(symbol),
    run_type           TEXT NOT NULL CHECK (run_type IN ('pre_open', 'scheduled', 'event')),
    as_of              TIMESTAMPTZ NOT NULL,
    generated_at       TIMESTAMPTZ NOT NULL,
    window_minutes     INTEGER NOT NULL CHECK (window_minutes IN (15, 30, 60)),
    horizon_start      TIMESTAMPTZ NOT NULL,
    horizon_end        TIMESTAMPTZ NOT NULL,
    trigger_article_id BIGINT REFERENCES articles(article_id),
    model_version      TEXT NOT NULL,
    label_version      TEXT NOT NULL,
    regime             TEXT CHECK (regime IN ('bullish', 'bearish', 'sideways')),
    prob_bullish       NUMERIC(5, 4),
    prob_bearish       NUMERIC(5, 4),
    prob_sideways      NUMERIC(5, 4),
    prob_up            NUMERIC(5, 4),
    entry_condition    TEXT CHECK (entry_condition IN ('favourable', 'cautious', 'avoid')),
    entry_score        NUMERIC(5, 4),
    confidence         NUMERIC(5, 4),
    invalidation       TEXT,
    top_drivers        JSONB,
    rationale          TEXT,
    features_used      JSONB,
    data_freshness     JSONB,
    status             TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'degraded', 'suppressed')),
    template_version   TEXT,
    rendered_text      TEXT,
    published          BOOLEAN NOT NULL DEFAULT FALSE,
    realized_return    NUMERIC(9, 6),
    realized_regime    TEXT CHECK (realized_regime IN ('bullish', 'bearish', 'sideways')),
    max_adverse_move   NUMERIC(9, 6),
    settled_at         TIMESTAMPTZ,
    UNIQUE (instrument, run_type, as_of, window_minutes)
);
CREATE INDEX IF NOT EXISTS idx_predictions_as_of ON predictions (as_of);

