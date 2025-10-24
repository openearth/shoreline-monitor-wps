-- Drop if it exists (optional)
DROP TABLE IF EXISTS gctr_shifted;

-- Create the derived table
CREATE TABLE gctr_shifted AS
WITH base AS (
  SELECT
    g.*,
    ST_LineMerge(g.geom_3857) AS geom_line
  FROM gctr g
),
mid_ref AS (
  SELECT
    *,
    -- midpoint as the new start
    ST_LineInterpolatePoint(geom_line, 0.5) AS mid_pt,

    -- direction endpoint chosen by sign (using interpolate 0/1 so it works for MultiLineString too)
    CASE
      WHEN sds_change_rate >= 0 THEN ST_LineInterpolatePoint(geom_line, 1.0)  -- end
      ELSE ST_LineInterpolatePoint(geom_line, 0.0)                            -- start
    END AS ref_pt,

    -- magnitude as a fraction in [0,1]
    LEAST(GREATEST(ABS(sds_change_rate), 0.0), 1.0) AS pct
  FROM base
),
tmp AS (
  SELECT
    *,
    ST_MakeLine(mid_pt, ref_pt) AS half_line
  FROM mid_ref
),
final AS (
  SELECT
    *,
    -- final point located pct along the half-line
    ST_LineInterpolatePoint(half_line, pct) AS end_pt
  FROM tmp
)
SELECT
  ROW_NUMBER() OVER ()::bigint AS gid,                 -- synthetic key; swap with your id if you have one
  sds_change_rate,
  ST_MakeLine(mid_pt, end_pt)::geometry(LineString, 3857) AS new_line_3857
FROM final;

ALTER TABLE gctr_shifted ADD PRIMARY KEY (gid);
CREATE INDEX gix_gctr_shifted_new_line ON gctr_shifted USING GIST (new_line_3857);
