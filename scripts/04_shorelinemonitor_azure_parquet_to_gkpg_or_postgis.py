# this script retrieves Floris' CoastMonitor ShorelineMonitor parquet datasets (GCTS, GCTR, shorelines, raw-series & series) from Azure and builds GKPG files in the cloud or a PostGIS database
# Developer: Etiënne Kras, 06-08-2025
# Env: Coastal

# notes:
# 100 MB parquet writes to 2.3 GB GPKG; performance is not sufficient for large datasets
# we need a PostGIS database to store the data and serve it to the frontend. ICT provides that
# can improve by: Parallelize uploads, Add more metadata columns/indexes, Use a schema inside the PostGIS DB
# see chatGPT "read parquet from Azure" in Etienne's account

# %% load packages

import os
import geopandas as gpd
import pandas as pd
import pystac
import tempfile
from shapely import wkb
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from azure.storage.blob import BlobServiceClient

load_dotenv()

# %% configure cloud settings and postGIS connection
postGIS = True  # set to False if you want to write to GPKG files instead of PostGIS
account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
container_name = "shorelinemonitor-series"  # "gctr", "shorelinemonitor-shorelines", 
                         #"shorelinemonitor-series"
# maybe later: "shorelinemonitor-raw-series", "gcts"

# PostgreSQL connection (adjust as needed)
pg_user = os.getenv("PG_USER")
pg_pass = os.getenv("PG_PASS")
pg_host = os.getenv("PG_HOST")
pg_db = os.getenv("PG_DB")
pg_user = 'postgres'
pg_pass = 'thingy'
pg_host = 'localhost'
pg_port = "5432"

engine = create_engine(
    f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
)

# def create_connection_db():
#     """Creates a connection to the database
#     Returns:
#         connection object
#     """
#     cf = read_config() # or through os.environ..
#     user = cf.get("PostGIS", "USER")
#     password = cf.get("POSTGIS", "PASSWORD")
#     host = cf.get("PostGIS", "HOST")
#     port = cf.get("PostGIS", "PORT")
#     database = cf.get("PostGIS", "DATABASE")
#     engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
#     return engine

# %% catalog load
coclico_catalog = pystac.Catalog.from_file(
    "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
)
collection = coclico_catalog.get_child(container_name)

# %% blob service client and helper funcs

account_url = f"https://{account_name}.blob.core.windows.net"
blob_service_client = BlobServiceClient(
    account_url=account_url, credential=os.environ["AZURE_STORAGE_SAS_TOKEN"]
)

container_client = blob_service_client.get_container_client(container_name)

# some DBASE Functions
# function to create table based on the columnname
Base = declarative_base()
inspector = inspect(engine)

def create_table(columname):
    if not inspector.has_table(columname):
        strsql = f"""create table {columname} ({columname}id serial, {columname} text)""" 
        with engine.begin() as conn:
            conn.execute(text(strsql))
    return

# setup classes that define tables to connect to 
class CommonCountryName(Base):
    __tablename__ = 'common_country_name'
    common_country_nameid = Column(Integer, primary_key=True)
    common_country_name = Column(String, unique=True)

class Transect(Base):
    __tablename__ = 'transect'
    transectid = Column(Integer, primary_key=True)
    transect = Column(String, unique=True)

# function to insert country name and or find already inserted name
def insert_common_country_name(engine, common_country_name):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        existing_country = session.query(CommonCountryName).filter_by(common_country_name=common_country_name).first()
        if existing_country:
            return existing_country.common_country_nameid
        else:
            new_country = CommonCountryName(common_country_name=common_country_name)
            session.add(new_country)
            session.commit()
            return new_country.common_country_nameid
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def insert_transect(engine, transectid):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        id = session.query(Transect).filter_by(transect=transectid).first()
        if id:
            return id.transectid
        else:
            id = Transect(transect=transectid)
            session.add(id)
            session.commit()
            return id.transectid
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# helper func
def parse_wkb(val):
    if isinstance(val, bytes):
        try:
            return wkb.loads(val)
        except Exception as e:
            print(f"Failed to parse WKB: {e}")
            return None
    return None


# %% looping over files
item_blobs = collection.get_all_items()
for idx, item in enumerate(item_blobs):
    
    # set properties
    item_href = item.assets["data"].href.split(f"{container_name}/")[-1]

    print(f"Processing {idx}: {item.id}")

    parquet_url = (
        f"https://{account_name}.blob.core.windows.net/"
        f"{container_name}/{item_href}"
        f"{os.environ["AZURE_STORAGE_SAS_TOKEN"]}"
    )

    # read paruqet from the cloud
    try:
        df = pd.read_parquet(parquet_url, engine="pyarrow", storage_options={})
    except Exception as e:
        print(f"Failed to read parquet for {item.id}: {e}")
        continue

    # create geodataframe
    if "geometry" in df.columns:
        try:
            gdf = gpd.GeoDataFrame(
                df, geometry=df["geometry"].apply(parse_wkb)
            )  # default is "OGC:CRS84"
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf.columns = gdf.columns.str.replace(
                ":", "_", regex=False
            )  # needed for PostGIS compatibility in GCTR
        except Exception as e:
            print(f"Failed to convert to GeoDataFrame for {item.id}: {e}")
            continue
    else:
        print(f"No geometry column found in {item.id}, skipping...")
        continue

    if postGIS == True:
        # check if table exists and drop it
        table_name = container_name.lower().replace("-", "_")
        if idx == 0:
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))

            print(f"Earlier table is dropped: {table_name}")

        # small test to put in the database
        # gdf_sm = gdf.head(10)

        # only select first X columns in the databse
        # gdf_sm = gdf.iloc[:, :26]  # adjust as needed, e.g., first 10 columns


        # only select relevant columns
        if container_name == "gctr":  # for GCTR
            # some data normalisation based on the columns below
            create_table('common_country_name')
            cntr_ids = {}
            for country in df['common_country_name'].unique():
                id = insert_common_country_name(engine,country)
                cntr_ids[country]=id

            create_table('transect')
            transects = {}
            for trid in df['transect_id']:
                id = insert_transect(engine,trid)
                transects[trid]=id

            gdf['country_id'] = gdf['common_country_name'].map(cntr_ids)
            gdf['transectid'] = gdf['transect_id'].map(transects)

            gdf = gdf[
                [
                    "transectid",
                    "geometry",
                    "continent",
                    "country_id",
                    "sds_start_datetime",
                    "sds_end_datetime",
                    "sds_change_rate",
                    "sds_change_intercept",
                    "sds_change_rate_std_err",
                    "sds_r_squared",
                    "class_shore_type",
                    "class_coastal_type",
                ]
            ]
        if container_name == "shorelinemonitor-series":  # for series

            # here a piece of code that gets the unique transect_ids
            # retrieves them from the gctr transect_ids, bear in mind, here you 
            # can have multiple records with the same transect_id! Hence the use of unique()
            # where it is supposed not to happen with gctr data
            transects = {}
            for trid in df['transect_id'].unique():
                id = insert_transect(engine,trid)
                transects[trid]=id 
            gdf['transectid'] = gdf['transect_id'].map(transects)

            gdf = gdf[
                [
                    "transectid",
                    "obs_id",
                    "datetime",
                    "geometry",
                    "shoreline_position",
                    "chainage",
                    "obs_is_primary",
                    "obs_is_outlier",
                ]
            ]

        # write to PostgreSQL with PostGIS database (appending the table)
        try:
            print(f"Uploading to PostGIS table: {table_name}")
            gdf.to_postgis(  # gdf_sm to test
                name=table_name,
                con=engine,
                if_exists="append",  # or 'append' if needed, "replace", "fail"
                index=False,
                chunksize=10000,  # boosts performance on large inserts
            )

        except Exception as e:
            print(f"Failed to upload {item.id}: {e}")

    if postGIS == False:
        # Write to temporary local file (GKPG) and then to cloud bucket
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg_path = os.path.join(tmpdir, f"{item.id}.gpkg")

            try:
                gdf.to_file(gpkg_path, driver="GPKG", layer="layer")
            except Exception as e:
                print(f"Failed to write GPKG file for {item.id}: {e}")
                continue

            # TODO: test the below to write to cloud storage, probably need to adjust folder names!
            # Upload to Azure Blob Storage
            output_blob_path = f"converted-gpkg/{item.id}.gpkg"
            blob_client = container_client.get_blob_client(output_blob_path)

            try:
                with open(gpkg_path, "rb") as f:
                    blob_client.upload_blob(f, overwrite=True)
                print(f"Uploaded GeoPackage: {output_blob_path}")
            except Exception as e:
                print(f"Failed to upload GeoPackage for {item.id}: {e}")

# %% set index and primary keys to tables in the database

# set index on the complete table
if postGIS == True:
    with engine.begin() as conn:
        # Create spatial index
        print(f"Creating spatial index on {table_name}...")
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {table_name}_geom_idx ON {table_name} USING GIST (geometry);"
            )
        )

        print(f"")
        conn.execute(
            text(
                """
            ALTER TABLE public.transects
            ADD CONSTRAINT transects_pkey PRIMARY KEY (transect_id);
        """
            )
        )
        conn.execute(
            text(
                """
            ALTER TABLE public.transect_points
            ADD COLUMN id bigserial PRIMARY KEY;
        """
            )
        )

    print(f"Done with setting index {table_name}")
