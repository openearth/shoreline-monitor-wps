-- it is recommende to use GIN with textindices
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- gctr optimisations
create table country (idccn serial PRIMARY key, common_country_name text);
INSERT into country (common_country_name) select distinct common_country_name from gctr;

create table continent (idcnt serial PRIMARY key, continent text);
INSERT into continent (continent) select distinct continent from gctr;

select count(*) from gctr where continent is null --530 profiles without name
select count(*) from gctr where common_country_name is null --530 profiles without name


alter table gctr add column idccn integer;
update gctr g set idccn = country.idccn
from country
where g.common_country_name = country.common_country_name;

alter table gctr add column idcnt integer;
update gctr g set idcnt = continent.idcnt
from continent
where g.continent = continent.continent;

-- test -- distinct idcnt (Identifier continent) should give exact the number of continents = 8 (incl. Null)
select distinct(idcnt) from gctr -- Nulls are not filled with 1's
select distinct(idcnt, continent) from gctr where idcnt = 2

select distinct(common_country_name) from gctr --should yield 221
select distinct(idccn) from gctr  --should yield 221 :)

-- adding sandy = true 
alter table public.gctr add column sandy BOOLEAN;
update public.gctr set sandy = False;
update public.gctr set sandy = True where class_shore_type like '%sandy%';


-- everything ok
-- before deletion of common_country_name and continent check the size of the table
select
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))),
  pg_total_relation_size(quote_ident(table_name))
from information_schema.tables
where table_schema = 'public' and table_name = 'gctr';
-- yields 5746 MB
-- after removal of continent and common_country_name and vacuum (full) size is 2635 MB

alter table gctr drop column common_country_name;
alter table gctr drop column continent;

-- if used in the queries.... add INDEXES to common_country_name etc.

-- shorelinemonitor_series optimization part
-- first set GIN index on transect_id for both the table gctr and shorelinemonitor_series
CREATE INDEX idx_trnsct_id_gctr ON gctr USING GIN (transect_id gin_trgm_ops);
CREATE INDEX idx_trnsct_id_sms ON shorelinemonitor_series USING GIN (transect_id gin_trgm_ops);


alter table shorelinemonitor_series add gctr_id integer;
update shorelinemonitor_series s set gctr_id = g.index
from gctr g
where s.transect_id = g.transect_id;

-- before deletion of transect_id check the size of the table
select
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))),
  pg_total_relation_size(quote_ident(table_name))
from information_schema.tables
where table_schema = 'public' and table_name = 'shorelinemonitor_series';
-- yields 63 GB


-- various cleaning
CREATE INDEX IF NOT EXISTS idx_shorelinemonitor_series_gctr_id ON public.shorelinemonitor_series (gctr_id);
CREATE INDEX IF NOT EXISTS idx_shorelinemonitor_series_datetime ON public.shorelinemonitor_series (datetime);

ANALYZE public.gctr;
ANALYZE public.country;
ANALYZE public.continent;
ANALYZE public.shorelinemonitor_series;

VACUUM (FULL) public.gctr;
VACUUM (FULL) public.country;
VACUUM (FULL) public.continent;
VACUUM (FULL) public.shorelinemonitor_series;

