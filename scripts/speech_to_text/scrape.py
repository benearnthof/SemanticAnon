
from pathlib import Path
import requests
import time
import re
from http.cookiejar import MozillaCookieJar
from yt_dlp import YoutubeDL

# -------------------------
# Configuration
# -------------------------

COLLECTION_ID = "875184"
EXPECTED = 655

OUTPUT =  Path("data") / "audio"
COOKIE_FILE = Path("cookies.txt")
ARCHIVE_FILE = "downloaded.txt"

OUTPUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# Authentication
# -------------------------

if not COOKIE_FILE.exists():
    raise FileNotFoundError(f"Missing cookies: {COOKIE_FILE}")

cookies = MozillaCookieJar(str(COOKIE_FILE))
cookies.load(ignore_discard=True, ignore_expires=True)

session = requests.Session()
session.cookies = cookies

session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/vnd.api+json",
})

# -------------------------
# Collection metadata
# -------------------------

print("Retrieving collection metadata...")

url = f"https://www.patreon.com/api/collection/{COLLECTION_ID}"

response = session.get(url, timeout=30)
response.raise_for_status()

collection = response.json()["data"]["attributes"]

thumbnail_url = collection["thumbnail"]["url"]

match = re.search(r"/campaign/(\d+)", thumbnail_url)

if not match:
    raise RuntimeError("Could not determine campaign ID")

campaign_id = match.group(1)

print(f"Campaign ID: {campaign_id}")

# -------------------------
# Retrieve episodes
# -------------------------

url = (
    "https://www.patreon.com/api/posts"
    "?include=campaign"
    f"&filter[campaign_id]={campaign_id}"
    "&filter[contains_exclusive_posts]=true"
    "&filter[is_draft]=false"
    f"&filter[collection_id]={COLLECTION_ID}"
    "&filter[include_drops]=true"
    "&sort=collection_order"
    "&json-api-version=1.0"
)

episodes = []

print("Retrieving episode list...")

while url:
    response = session.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()

    for post in data.get("data", []):
        episodes.append(post["id"])

    url = data.get("links", {}).get("next")

    print(f"Retrieved {len(episodes)} episodes...")

print(f"\nFound {len(episodes)} episodes.")
print(f"Expected: {EXPECTED}")

if len(episodes) != EXPECTED:
    print("WARNING: Episode count differs from expectation.")

# -------------------------
# Download configuration
# -------------------------

options = {
    "cookiefile": str(COOKIE_FILE),

    "format": "bestaudio/best",

    "outtmpl": str(
        OUTPUT / "%(title)s [%(id)s].%(ext)s"
    ),

    "writeinfojson": True,

    "continuedl": True,
    "ignoreerrors": True,

    # Track successful downloads
    "download_archive": str(ARCHIVE_FILE),

    # Retry transient errors
    "retries": 10,
    "fragment_retries": 10,
    "extractor_retries": 5,

    # Avoid hammering Patreon
    "sleep_interval": .1,
    "max_sleep_interval": .5,

    "quiet": False,
    "no_warnings": False,
}

# -------------------------
# Download episodes
# -------------------------

print("\nStarting downloads...\n")

with YoutubeDL(options) as ydl:

    for i, episode_id in enumerate(episodes, 1):

        print(
            f"\n[{i}/{len(episodes)}] "
            f"Episode {episode_id}"
        )

        try:
            ydl.download([
                f"https://www.patreon.com/posts/{episode_id}"
            ])

        except Exception as e:
            print(f"Error downloading {episode_id}: {e}")

print("\nFinished processing collection.")
print(f"Audio directory: {OUTPUT}")
