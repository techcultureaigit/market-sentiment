-- Allow 1-minute and 5-minute intraday prediction windows in predictions table
ALTER TABLE predictions DROP CONSTRAINT IF EXISTS predictions_window_minutes_check;
ALTER TABLE predictions ADD CONSTRAINT predictions_window_minutes_check CHECK (window_minutes IN (1, 5, 15, 30, 60));

