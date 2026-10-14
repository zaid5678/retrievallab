import urllib.request

from src.retrievallab.nist import CATALOG_PATH, CATALOG_URL


def main():
    if CATALOG_PATH.exists():
        print(f"Catalog already present at {CATALOG_PATH}")
        return

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NIST SP 800-53 Rev. 5 catalog to {CATALOG_PATH} ...")
    urllib.request.urlretrieve(CATALOG_URL, CATALOG_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
