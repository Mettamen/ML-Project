"""Download the original Kaggle archive and extract only NIFTY50_all.csv."""
from pathlib import Path
import shutil
import urllib.request
import zipfile

SOURCE = "https://www.kaggle.com/api/v1/datasets/download/rohanrao/nifty50-stock-market-data"
DESTINATION = Path(__file__).resolve().parent / "data"


def main():
    DESTINATION.mkdir(exist_ok=True)
    target = DESTINATION / "NIFTY50_all.csv"
    if target.exists():
        print(f"Already downloaded: {target}")
        return
    archive = DESTINATION / "nifty50.zip"
    print("Downloading the original NIFTY-50 dataset from Kaggle...")
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "NIFTY50EducationalDemo/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as bundle:
            candidates = [name for name in bundle.namelist() if Path(name).name == "NIFTY50_all.csv"]
            if len(candidates) != 1:
                raise ValueError("The archive must contain exactly one NIFTY50_all.csv.")
            # Write to a fixed local filename; never extract arbitrary archive paths.
            with bundle.open(candidates[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
        print(f"Saved: {target}")
    except Exception as error:
        raise SystemExit(
            f"Automatic download failed: {error}\n"
            "Download manually from https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data "
            "and upload NIFTY50_all.csv in the app."
        )


if __name__ == "__main__":
    main()
