"""Downloads the shared examples/tutorials data bundle (meshes, textures,
...) from a public Google Drive file and unzips it locally -- pure Python
stdlib only (urllib/http.cookiejar/zipfile), no extra dependency.
"""

import http.cookiejar
import io
import urllib.request
import zipfile
from pathlib import Path

_EXAMPLES_DATA_FILE_ID = "1D1ozWMdlPRlaQYFft80-FMj9qLDET_98"
_DRIVE_DOWNLOAD_URL = "https://docs.google.com/uc?export=download"


def _drive_download(file_id: str) -> bytes:
    """Downloads `file_id`'s raw bytes from Google Drive, transparently
    following the "Google can't scan this file for viruses" interstitial
    (a `download_warning_...` cookie carrying a confirm token) that Drive
    serves for files it doesn't return directly.
    """
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    url = f"{_DRIVE_DOWNLOAD_URL}&id={file_id}"
    with opener.open(url) as response:
        data = response.read()

    token = next((c.value for c in cookie_jar if c.name.startswith("download_warning")), None)
    if token:
        with opener.open(f"{url}&confirm={token}") as response:
            data = response.read()

    return data


def download_examples_data(target_dir: str = "data", force: bool = False) -> str:
    """Downloads the examples/tutorials data bundle (a zip file hosted on
    Google Drive) and extracts it into `target_dir` (created if needed).

    :param target_dir: Local folder to extract the zip into. Relative
        paths resolve against the current working directory.
    :param force: Re-download/re-extract even if `target_dir` already
        exists and isn't empty. By default, an existing non-empty
        `target_dir` is left untouched.
    :return: `target_dir`, as a string.
    """
    path = Path(target_dir)
    if path.is_dir() and any(path.iterdir()) and not force:
        print(f"'{path}' already exists and is not empty; skipping download (force=True to re-download).")
        return str(path)

    path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading examples data into '{path}'...")
    zip_bytes = _drive_download(_EXAMPLES_DATA_FILE_ID)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(path)

    print(f"Done: examples data extracted into '{path}'.")
    return str(path)
