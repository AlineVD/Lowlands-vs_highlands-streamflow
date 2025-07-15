import os
import time
import glob
import paths
import shutil
import zipfile

from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def prepare_output_dir(station_id):
    target_dir = paths.CATCHMENT_BASINS + f"/{str(station_id)}"
    os.makedirs(target_dir, exist_ok=True)
    print(target_dir)
    return target_dir


def rename_shapefiles_to_station_id(folder_path, station_id):
    base_names = ["shp", "dbf", "prj", "shx", "cst", "csv"]
    for ext in base_names:
        files = glob.glob(os.path.join(folder_path, f"*.{ext}"))
        for f in files:
            new_name = os.path.join(folder_path, f"{station_id}.{ext}")
            print(f"🔄 Renaming {os.path.basename(f)} → {station_id}.{ext}")
            os.rename(f, new_name)

def download_shapefile_auto(station_id, download_dir="downloads"):
    os.makedirs(download_dir, exist_ok=True)
    url = f"https://nrfa.ceh.ac.uk/data/station/spatial_download/{station_id}"

    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "directory_upgrade": True
    })
    # Optional: show browser for debugging
    # chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    wait = WebDriverWait(driver, 15)

    print(f"\n🚀 Opening station {station_id} download page...")
    driver.get(url)

    # Accept cookies
    try:
        cookie_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".agree-button")))
        driver.execute_script("arguments[0].click();", cookie_btn)
        print("✅ Accepted cookies.")
    except:
        print("ℹ️ No cookie popup found.")

    try:
        # Native JS injection using the correct ID: 'download-who'
        driver.execute_script("""
        const select = document.getElementById('download-who');
        for (const option of select.options) {
            if (option.text.includes('University student')) {
                select.value = option.value;
                const event = new Event('change', { bubbles: true });
                select.dispatchEvent(event);
                break;
                }
            }
        """)
        print("✅ Selected 'University student' via native JS.")
    except Exception as e:
        print("❌ Failed to set sector via native JS:", e)

    # Tick all checkboxes
    checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
    for cb in checkboxes:
        driver.execute_script("arguments[0].click();", cb)
    print(f"✅ Ticked {len(checkboxes)} checkboxes.")

    # ✅ Click the final <button id="goDownload"> once all fields are filled
    try:
        go_button = wait.until(
            EC.element_to_be_clickable((By.ID, "goDownload"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", go_button)
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", go_button)
        print("📦 Clicked final 'Download' button (goDownload).")
    except Exception as e:
        print("❌ Failed to click final download button:", e)
        driver.quit()
        return False

    print("⏳ Waiting for downloaded folder or ZIP file to appear...")
    downloaded_path = None
    start_time = time.time()
    timeout = 60  # seconds



    while time.time() - start_time < timeout:
        # Check for .zip file
        zip_files = glob.glob(os.path.join(download_dir, "*.zip"))
        if zip_files:
            zip_path = zip_files[0]
            target_dir = prepare_output_dir(station_id)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(zip_path)
            rename_shapefiles_to_station_id(target_dir, station_id)
            print(f"✅ Unzipped to: {target_dir}")
            driver.quit()
            return True

        # Check for raw folder download
        folders = [
            f for f in os.listdir(download_dir)
            if os.path.isdir(os.path.join(download_dir, f)) and f.lower().startswith("catchment_boundary")
        ]
        if folders:
            folder_path = os.path.join(download_dir, folders[0])
            target_dir = prepare_output_dir(station_id)
            shutil.move(folder_path, target_dir)
            print(f"✅ Moved folder to: {target_dir}")
            driver.quit()
            return True

        time.sleep(0.5)

    driver.quit()
    print("❌ No shapefile folder or ZIP file downloaded.")
    return False


