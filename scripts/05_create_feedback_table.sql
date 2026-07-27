-- Create the shoreline_feedback table.
-- Run once against the target database using the credentials in configuration.txt.
-- Idempotent: safe to re-run.
--
-- Privacy note: client_ip is captured server-side from nginx headers only
-- (X-Forwarded-For / X-Real-IP / remote_addr). No client-side IP lookup services
-- are used. Define a retention policy and review access controls before deploying
-- to production.

CREATE TABLE IF NOT EXISTS public.shoreline_feedback (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    reliability         TEXT NOT NULL,
    details             TEXT[],
    other_text          TEXT,
    transect_id         TEXT,
    profileid           BIGINT,
    center_lon          DOUBLE PRECISION,
    center_lat          DOUBLE PRECISION,
    country             TEXT,
    continent           TEXT,
    shore_type          TEXT,
    change_rate         DOUBLE PRECISION,
    change_rate_std_err DOUBLE PRECISION,
    classification      TEXT,
    r_squared           DOUBLE PRECISION,
    client_ip           INET,
    user_agent          TEXT
);

COMMENT ON TABLE public.shoreline_feedback IS
    'Profile reliability feedback submitted via the shoreline monitor front-end. Client IP is captured server-side; no third-party geolocation data is stored.';
