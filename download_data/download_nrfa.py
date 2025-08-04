import os
import csv
import json
import paths
import urllib
import shutil

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
            if start_year <= 1980 and end_year >= 2020:
                qualifying.append(sid)
        except Exception as e:
            print(f"⚠️ Skipping {sid} due to error: {e}")
            continue

    with open(QUALIFYING_IDS_PATH, "w") as f:
        json.dump(qualifying, f)
        print(f"✅ Saved {len(qualifying)} qualifying IDs to {QUALIFYING_IDS_PATH}")

    return qualifying


def download_csv(station_id, data_type='gdf', csv_path=None):
    print(station_id)
    query = f"station={station_id}&data-type={data_type}&format=json-object"
    url = f"{nrfa_api.BASE_URL}/time-series?{query}"
    print(url)
    response = urllib.request.urlopen(url).read()
    response = json.loads(response)
    stream = response['data-stream']
    if not stream:
        return False
    if data_type == 'cdr':
        ext = '_nrfa'
    else:
        ext = ''
    csv_path = paths.CATCHMENT_BASINS + f"/{str(station_id)}/{str(station_id)}{ext}.csv"
    nrfa_api.save_nrfa_api_response_to_csv(response, csv_path)

    return True


def load_log(log_path):
    """
    Load the log from JSON file or return an empty dict if file doesn't exist.
    """
    if not os.path.exists(log_path):
        return {}
    with open(log_path, "r") as f:
        return json.load(f)


def update_log_with_qualifying_ids(log_path, ids_path):
    """
    Load the log file, add missing qualifying IDs, and save the updated log.
    """
    log = load_log(log_path)

    # Load qualifying station IDs
    with open(ids_path, "r") as f:
        qualifying_ids = json.load(f)

    for sid in qualifying_ids:
        str_sid = str(sid)
        if str_sid not in log:
            log[str_sid] = {
                "flow_success": False,
                "rain_success": False,
                "shape_success": False,
                "status": "not_attempted"
            }

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"✅ Log updated with {len(qualifying_ids)} qualifying station IDs.")
    return log


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

        # Load previous station log if it exists, otherwise initialize
        station_log = log.get(str_sid, {
            "flow_success": False,
            "rain_success": False,
            "shape_success": False,
            "status": "not_attempted"
        })

        # Skip if already fully successful
        if station_log.get("status") == "success":
            print(str_sid, 'already downloaded')
            continue

        try:
            # Only download what's missing
            if not station_log.get("flow_success", False):
                print("downloading streamflow")
                station_log["flow_success"] = download_csv(sid, 'gdf')
                print(station_log["flow_success"])

            if not station_log.get("rain_success", False):
                print("downloading rainfall")
                station_log["rain_success"] = download_csv(sid, 'cdr')
                print(station_log["rain_success"])

                # If still failed, move to unsuccessful folder
                if not station_log["rain_success"]:
                    unsuccessful_dir = paths.CATCHMENT_BASINS + f"/unsuccessful/{sid}"
                    os.makedirs(os.path.dirname(unsuccessful_dir), exist_ok=True)

                    try:
                        shutil.move(paths.CATCHMENT_BASINS + f"/{sid}", unsuccessful_dir)
                        print(f"❌ Moved {sid} to 'unsuccessful'")
                    except FileExistsError:
                        print(f"⚠️ Folder {sid} already exists in 'unsuccessful' — skipping or handling manually")

            if not station_log.get("shape_success", False):
                print("downloading shapefile")
                station_log["shape_success"] = download_shapefiles.download_shapefile_auto(sid)
                print(station_log["shape_success"])

            # Determine updated status
            if station_log["flow_success"] and station_log["rain_success"] and station_log["shape_success"]:
                station_log["status"] = "success"
            elif station_log["flow_success"] or station_log["rain_success"] or station_log["shape_success"]:
                station_log["status"] = "partial"
            else:
                station_log["status"] = "failed"

        except Exception as e:
            print(f"Exception while processing station {sid}: {e}")
            station_log["status"] = f"failed: {str(e)}"
            # Do not overwrite previous partial successes

        # Save the updated log entry
        log[str_sid] = station_log
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    print("\n✅ Download process complete.")


def main():

    if not os.path.exists(QUALIFYING_IDS_PATH):

        station_ids = nrfa_api.get_all_stations_id()
        get_qualifying_station_ids(station_ids, nrfa_api.BASE_URL)

    download_all_station_data()

if __name__ == "__main__":
    main()