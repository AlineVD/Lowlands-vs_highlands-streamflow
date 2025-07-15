import os
import csv
import json
import paths
import urllib

import nrfa_api, download_shapefiles

from tqdm import tqdm

QUALIFYING_IDS_PATH = paths.DATA + "/qualifying_station_ids.json"
LOG_IDS_PATH = paths.DATA + "/station_download_log.json"


def get_qualifying_station_ids(all_station_ids, base_url):
    """
    if os.path.exists(QUALIFYING_IDS_PATH):
        print("📂 Loading cached qualifying station IDs...")
        with open(QUALIFYING_IDS_PATH, "r") as f:
            return json.load(f)
            """

    print("🔍 Filtering qualifying station IDs...")
    qualifying = []
    for sid in tqdm(all_station_ids, desc="Filtering stations"):
        try:
            query = f"station={sid}&data-type=gdf&format=json-object"
            url = f"{base_url}/time-series?{query}"
            response = urllib.request.urlopen(url, timeout=60).read()
            stream = json.loads(response)['data-stream']
            if len(stream) < 2:
                continue
            start_year = int(stream[0][:4])
            end_year = int(stream[-2][:4])
            if start_year <= 1980 and end_year >= 2022:
                qualifying.append(sid)
        except Exception as e:
            print(f"⚠️ Skipping {sid} due to error: {e}")
            continue

    with open(QUALIFYING_IDS_PATH, "w") as f:
        json.dump(qualifying, f)
        print(f"✅ Saved {len(qualifying)} qualifying IDs to {QUALIFYING_IDS_PATH}")

    return qualifying


def download_csv(station_id):
    print(station_id)
    query = f"station={station_id}&data-type=gdf&format=json-object"
    url = f"{nrfa_api.BASE_URL}/time-series?{query}"
    response = urllib.request.urlopen(url).read()
    response = json.loads(response)
    stream = response['data-stream']
    if not stream:
        return False
    csv_path = paths.CATCHMENT_BASINS + f"/{str(station_id)}/{str(station_id)}.csv"
    nrfa_api.save_nrfa_api_response_to_csv(response, csv_path)

    return True

def load_processed_stations(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return set(row['station_id'] for row in reader)

def download_all_station_data(
    ids_path=QUALIFYING_IDS_PATH,
    log_path=LOG_IDS_PATH,
):
    # Load qualifying station IDs
    with open(ids_path, "r") as f:
        station_ids = json.load(f)

    # Load or initialize status log
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = {}

    # Process each station
    for sid in tqdm(station_ids, desc="📥 Downloading station data"):
        str_sid = str(sid)
        if str_sid in log and log[str_sid].get("status") == "success":
            print(str_sid, 'already downloaded')
            continue  # Already done

        try:
            print("downloading streamflow")
            flow_success = download_csv(sid)
            print(flow_success)
            print("downloading shapefile")
            shape_success = download_shapefiles.download_shapefile_auto(sid)
            if flow_success and shape_success:
                status = "success"
            elif flow_success or shape_success:
                status = "partial"
            else:
                status = "failed"
        except Exception as e:
            flow_success = shape_success = False
            status = f"failed: {str(e)}"

        log[str_sid] = {
            "flow_success": flow_success,
            "shape_success": shape_success,
            "status": status
        }

        # Update log after each station
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    print("\n✅ Download process complete.")

# === Main entry point ===
def main():

    if not os.path.exists(QUALIFYING_IDS_PATH):

        station_ids = nrfa_api.get_all_stations_id()
        get_qualifying_station_ids(station_ids, nrfa_api.BASE_URL)

    download_all_station_data()

if __name__ == "__main__":
    main()