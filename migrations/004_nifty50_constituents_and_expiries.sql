-- 004_nifty50_constituents_and_expiries.sql
-- Seed all 50 Nifty 50 constituents with approximate index weights and
-- populate the 2026 Nifty 50 weekly/monthly F&O expiry calendar.

INSERT INTO instruments (
    symbol, vendor_ticker, source, name, exchange, asset_class,
    timezone, session_open, session_close, publish_lag_sec, is_heavyweight, index_weight
) VALUES
    ('HDFCBANK',   'HDFCBANK.NS',   'yfinance', 'HDFC Bank Ltd',                          'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,  11.90),
    ('RELIANCE',   'RELIANCE.NS',   'yfinance', 'Reliance Industries Ltd',                'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   8.45),
    ('ICICIBANK',  'ICICIBANK.NS',  'yfinance', 'ICICI Bank Ltd',                         'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   8.10),
    ('INFY',       'INFY.NS',       'yfinance', 'Infosys Ltd',                            'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   5.95),
    ('ITC',        'ITC.NS',        'yfinance', 'ITC Ltd',                                'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   4.15),
    ('TCS',        'TCS.NS',        'yfinance', 'Tata Consultancy Services Ltd',          'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   3.95),
    ('LT',         'LT.NS',         'yfinance', 'Larsen & Toubro Ltd',                    'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   3.85),
    ('BHARTIARTL', 'BHARTIARTL.NS', 'yfinance', 'Bharti Airtel Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   3.75),
    ('AXISBANK',   'AXISBANK.NS',   'yfinance', 'Axis Bank Ltd',                          'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   3.10),
    ('SBIN',       'SBIN.NS',       'yfinance', 'State Bank of India',                    'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, TRUE,   2.95),
    ('M&M',        'M&M.NS',        'yfinance', 'Mahindra & Mahindra Ltd',                'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  2.45),
    ('KOTAKBANK',  'KOTAKBANK.NS',  'yfinance', 'Kotak Mahindra Bank Ltd',                'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  2.40),
    ('HINDUNILVR', 'HINDUNILVR.NS', 'yfinance', 'Hindustan Unilever Ltd',                 'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  2.10),
    ('SUNPHARMA',  'SUNPHARMA.NS',  'yfinance', 'Sun Pharmaceutical Industries Ltd',      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.85),
    ('NTPC',       'NTPC.NS',       'yfinance', 'NTPC Ltd',                               'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.65),
    ('TATAMOTORS', 'TATAMOTORS.NS', 'yfinance', 'Tata Motors Ltd',                        'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.60),
    ('HCLTECH',    'HCLTECH.NS',    'yfinance', 'HCL Technologies Ltd',                   'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.55),
    ('MARUTI',     'MARUTI.NS',     'yfinance', 'Maruti Suzuki India Ltd',                'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.45),
    ('TRENT',      'TRENT.NS',      'yfinance', 'Trent Ltd',                              'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.40),
    ('POWERGRID',  'POWERGRID.NS',  'yfinance', 'Power Grid Corporation of India Ltd',    'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.35),
    ('TITAN',      'TITAN.NS',      'yfinance', 'Titan Company Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.30),
    ('ASIANPAINT', 'ASIANPAINT.NS', 'yfinance', 'Asian Paints Ltd',                       'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.25),
    ('TATASTEEL',  'TATASTEEL.NS',  'yfinance', 'Tata Steel Ltd',                         'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.20),
    ('BAJFINANCE', 'BAJFINANCE.NS', 'yfinance', 'Bajaj Finance Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.80),
    ('ULTRACEMCO', 'ULTRACEMCO.NS', 'yfinance', 'UltraTech Cement Ltd',                   'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.15),
    ('COALINDIA',  'COALINDIA.NS',  'yfinance', 'Coal India Ltd',                         'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.05),
    ('ONGC',       'ONGC.NS',       'yfinance', 'Oil & Natural Gas Corporation Ltd',      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  1.00),
    ('BAJAJ-AUTO', 'BAJAJ-AUTO.NS', 'yfinance', 'Bajaj Auto Ltd',                         'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.98),
    ('ADANIPORTS', 'ADANIPORTS.NS', 'yfinance', 'Adani Ports and Special Economic Zone',  'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.95),
    ('HINDALCO',   'HINDALCO.NS',   'yfinance', 'Hindalco Industries Ltd',                'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.92),
    ('GRASIM',     'GRASIM.NS',     'yfinance', 'Grasim Industries Ltd',                  'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.90),
    ('TECHM',      'TECHM.NS',      'yfinance', 'Tech Mahindra Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.88),
    ('BAJAJFINSV', 'BAJAJFINSV.NS', 'yfinance', 'Bajaj Finserv Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.86),
    ('BEL',        'BEL.NS',        'yfinance', 'Bharat Electronics Ltd',                 'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.85),
    ('JSWSTEEL',   'JSWSTEEL.NS',   'yfinance', 'JSW Steel Ltd',                          'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.83),
    ('NESTLEIND',  'NESTLEIND.NS',  'yfinance', 'Nestle India Ltd',                       'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.82),
    ('SHRIRAMFIN', 'SHRIRAMFIN.NS', 'yfinance', 'Shriram Finance Ltd',                    'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.80),
    ('CIPLA',      'CIPLA.NS',      'yfinance', 'Cipla Ltd',                              'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.78),
    ('WIPRO',      'WIPRO.NS',      'yfinance', 'Wipro Ltd',                              'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.76),
    ('DRREDDY',    'DRREDDY.NS',    'yfinance', 'Dr. Reddys Laboratories Ltd',            'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.74),
    ('SBILIFE',    'SBILIFE.NS',    'yfinance', 'SBI Life Insurance Company Ltd',         'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.70),
    ('EICHERMOT',  'EICHERMOT.NS',  'yfinance', 'Eicher Motors Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.68),
    ('HDFCLIFE',   'HDFCLIFE.NS',   'yfinance', 'HDFC Life Insurance Company Ltd',        'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.66),
    ('BPCL',       'BPCL.NS',       'yfinance', 'Bharat Petroleum Corporation Ltd',       'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.64),
    ('TATACONSUM', 'TATACONSUM.NS', 'yfinance', 'Tata Consumer Products Ltd',             'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.62),
    ('APOLLOHOSP', 'APOLLOHOSP.NS', 'yfinance', 'Apollo Hospitals Enterprise Ltd',        'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.60),
    ('HEROMOTOCO', 'HEROMOTOCO.NS', 'yfinance', 'Hero MotoCorp Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.58),
    ('BRITANNIA',  'BRITANNIA.NS',  'yfinance', 'Britannia Industries Ltd',               'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.56),
    ('INDUSINDBK', 'INDUSINDBK.NS', 'yfinance', 'IndusInd Bank Ltd',                      'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.54),
    ('ADANIENT',   'ADANIENT.NS',   'yfinance', 'Adani Enterprises Ltd',                  'NSE', 'equity', 'Asia/Kolkata', '09:15', '15:30', 60, FALSE,  0.52)
ON CONFLICT (symbol) DO UPDATE SET
    index_weight = EXCLUDED.index_weight,
    is_heavyweight = EXCLUDED.is_heavyweight;

-- Populate Nifty 50 Thursday F&O expiry dates for 2026 (shifted to Wednesday if Thursday is an NSE holiday).
WITH thursdays AS (
    SELECT d::date AS raw_dt,
           CASE WHEN EXISTS (SELECT 1 FROM nse_holidays h WHERE h.holiday_date = d::date)
                THEN (d::date - INTERVAL '1 day')::date
                ELSE d::date
           END AS expiry_dt
    FROM generate_series('2026-01-01'::date, '2026-12-31'::date, '1 day'::interval) AS g(d)
    WHERE EXTRACT(DOW FROM d) = 4
),
classified AS (
    SELECT expiry_dt,
           CASE WHEN LEAD( EXTRACT(MONTH FROM raw_dt) ) OVER (ORDER BY raw_dt) IS DISTINCT FROM EXTRACT(MONTH FROM raw_dt)
                THEN 'monthly'
                ELSE 'weekly'
           END AS expiry_kind
    FROM thursdays
)
INSERT INTO expiry_calendar (underlying, expiry_date, expiry_kind)
SELECT 'NIFTY50', expiry_dt, expiry_kind FROM classified
ON CONFLICT (underlying, expiry_date) DO NOTHING;

