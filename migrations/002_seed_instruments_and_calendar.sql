-- 002_seed_instruments_and_calendar.sql
-- Seed core instruments (indices, volatility, global cues, and Nifty 50 heavyweights)
-- and NSE trading holidays.

INSERT INTO instruments (
    symbol, vendor_ticker, source, name, exchange, asset_class,
    timezone, session_open, session_close, publish_lag_sec, is_heavyweight, index_weight
) VALUES
    -- Core Indian Indices & Volatility
    ('NIFTY50',    '^NSEI',       'yfinance', 'Nifty 50 Index',          'NSE',  'index',      'Asia/Kolkata',    '09:15', '15:30', 60,  FALSE, NULL),
    ('BANKNIFTY',  '^NSEBANK',    'yfinance', 'Nifty Bank Index',        'NSE',  'index',      'Asia/Kolkata',    '09:15', '15:30', 60,  FALSE, NULL),
    ('INDIAVIX',   '^INDIAVIX',   'yfinance', 'India VIX',               'NSE',  'volatility', 'Asia/Kolkata',    '09:15', '15:30', 60,  FALSE, NULL),

    -- Global Cues & Macro
    ('SP500',      '^GSPC',       'yfinance', 'S&P 500 Index',           'NYSE', 'index',      'America/New_York','09:30', '16:00', 60,  FALSE, NULL),
    ('NASDAQ',     '^IXIC',       'yfinance', 'NASDAQ Composite',        'NASDAQ','index',     'America/New_York','09:30', '16:00', 60,  FALSE, NULL),
    ('BRENT',      'BZ=F',        'yfinance', 'Brent Crude Oil Futures', 'ICE',  'commodity',  'Europe/London',   '01:00', '23:00', 60,  FALSE, NULL),
    ('USDINR',     'INR=X',       'yfinance', 'USD/INR Spot Rate',       'FX',   'fx',         'Asia/Kolkata',    '09:00', '17:00', 60,  FALSE, NULL),

    -- Top 10 Nifty 50 Heavyweights (for participation & breadth features)
    ('HDFCBANK',   'HDFCBANK.NS', 'yfinance', 'HDFC Bank Ltd',           'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,  11.90),
    ('RELIANCE',   'RELIANCE.NS', 'yfinance', 'Reliance Industries Ltd', 'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   8.45),
    ('ICICIBANK',  'ICICIBANK.NS','yfinance', 'ICICI Bank Ltd',          'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   8.10),
    ('INFY',       'INFY.NS',     'yfinance', 'Infosys Ltd',             'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   5.95),
    ('ITC',        'ITC.NS',      'yfinance', 'ITC Ltd',                 'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   4.15),
    ('TCS',        'TCS.NS',      'yfinance', 'Tata Consultancy Services','NSE', 'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   3.95),
    ('LT',         'LT.NS',       'yfinance', 'Larsen & Toubro Ltd',     'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   3.85),
    ('BHARTIARTL', 'BHARTIARTL.NS','yfinance','Bharti Airtel Ltd',       'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   3.75),
    ('AXISBANK',   'AXISBANK.NS', 'yfinance', 'Axis Bank Ltd',           'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   3.10),
    ('SBIN',       'SBIN.NS',     'yfinance', 'State Bank of India',     'NSE',  'equity',     'Asia/Kolkata',    '09:15', '15:30', 60,  TRUE,   2.95)
ON CONFLICT (symbol) DO NOTHING;

-- Seed NSE trading holidays (2025-2026)
INSERT INTO nse_holidays (holiday_date, description) VALUES
    ('2026-01-26', 'Republic Day'),
    ('2026-03-03', 'Holi'),
    ('2026-03-26', 'Shri Ram Navami'),
    ('2026-03-31', 'Id-Ul-Fitr (Ramzan Id)'),
    ('2026-04-03', 'Good Friday'),
    ('2026-04-14', 'Dr. Baba Saheb Ambedkar Jayanti'),
    ('2026-05-01', 'Maharashtra Day'),
    ('2026-08-15', 'Independence Day'),
    ('2026-09-14', 'Ganesh Chaturthi'),
    ('2026-10-02', 'Mahatma Gandhi Jayanti'),
    ('2026-10-20', 'Dussehra'),
    ('2026-11-09', 'Diwali-Balipratipada'),
    ('2026-11-24', 'Gurunanak Jayanti'),
    ('2026-12-25', 'Christmas')
ON CONFLICT (holiday_date) DO NOTHING;

