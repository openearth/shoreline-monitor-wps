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
from sqlalchemy import create_engine, text
from azure.storage.blob import BlobServiceClient

load_dotenv()

# %% configure cloud settings and postGIS connection
postGIS = True  # set to False if you want to write to GPKG files instead of PostGIS
account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
container_name = "shorelinemonitor-series"  # "gctr", "shorelinemonitor-shorelines", "shorelinemonitor-series"
# maybe later: "shorelinemonitor-raw-series", "gcts"

# PostgreSQL connection (adjust as needed)
pg_user = os.getenv("PG_USER")
pg_pass = os.getenv("PG_PASS")
pg_host = os.getenv("PG_HOST")
pg_db = os.getenv("PG_DB")
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
            gdf = gdf[
                [
                    "transect_id",
                    "geometry",
                    "continent",
                    "common_country_name",
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
            gdf = gdf[
                [
                    "transect_id",
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
