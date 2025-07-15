import os
import csv
import json
import urllib.request

BASE_URL = "https://nrfaapps.ceh.ac.uk/nrfa/ws"

def get_all_stations_id():
    station_ids_url = f"{BASE_URL}/station-ids?format=json-object"
    response = urllib.request.urlopen(station_ids_url).read()
    return json.loads(response)['station-ids']

def save_nrfa_api_response_to_csv(response: dict, out_path: str):
    """Converts NRFA API JSON response to a CSV file with metadata and timeseries."""

    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write metadata
        writer.writerow(["file", "timestamp", response.get("timestamp", "")])
        writer.writerow(["database", "id", "nrfa-public-30"])
        writer.writerow(["database", "name", "UK National River Flow Archive"])

        station = response.get("station", {})
        writer.writerow(["station", "id", station.get("id", "")])
        writer.writerow(["station", "name", station.get("name", "")])
        writer.writerow(["station", "gridReference", f"{station.get('easting' ,'')}, {station.get('northing' ,'')}"])

        data_type = response.get("data-type", {})
        writer.writerow(["dataType", "id", data_type.get("id", "")])
        writer.writerow(["dataType", "name", data_type.get("name", "")])
        writer.writerow(["dataType", "parameter", data_type.get("parameter", "")])
        writer.writerow(["dataType", "units", data_type.get("units", "")])
        writer.writerow(["dataType", "period", data_type.get("period", "")])
        writer.writerow(["dataType", "measurementType", data_type.get("measurement-type", "")])

        # Extract and format time series
        data_stream = response.get("data-stream", [])
        if len(data_stream) % 2 != 0:
            raise ValueError("❌ Uneven data-stream list length: expected alternating date/value.")

        dates = data_stream[::2]
        values = data_stream[1::2]

        writer.writerow(["data", "first", dates[0]])
        writer.writerow(["data", "last", dates[-1]])
        writer.writerow([])  # Empty line
        writer.writerow(["date", "flow (m3/s)"])

        for date, value in zip(dates, values):
            writer.writerow([date, value])