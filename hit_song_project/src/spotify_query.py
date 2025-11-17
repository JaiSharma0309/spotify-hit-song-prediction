"""
@file spotify_query.py
@brief Enriches a local CSV of songs with extra Spotify metadata.

This script:
  - Reads `spotify.csv` (with at least `song_title` and `artist` columns).
  - Queries the Spotify Web API (via spotipy) for each row.
  - Adds track-level info (release date, album type, explicit flag, popularity).
  - Adds artist-level info (artist popularity, genres).
  - Writes out an enriched file `spotify_enriched.csv`.
  - No manual scaling or modeling here, just clean feature enrichment.
  - All ML happens later in `hit_song.py`.
"""

import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# -------------------------------------------------------------------
# 1. Entering Spotify API credentials
# -------------------------------------------------------------------

CLIENT_ID = ""
CLIENT_SECRET = ""


def get_spotify_client():
    """
    @brief Create an authenticated Spotify client using client credentials.

    This uses the application-level "Client Credentials" OAuth flow.
    There is no user login; the app can only access public Spotify data.

    @return spotipy.Spotify
        An authenticated Spotify client object that can be used to call
        search, artist, and other endpoints.
    """
    auth_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    return spotipy.Spotify(auth_manager=auth_manager)


# -------------------------------------------------------------------
# 2. Spotify Query Helpers
# -------------------------------------------------------------------

def get_track_info(sp, song, artist):
    """
    @brief Fetch track-level metadata for a (song, artist) pair.

    The function builds a search query of the form:
        "track:<song> artist:<artist>"

    It then takes the top search result (if any) and extracts:
      - Spotify track ID
      - Matched track title
      - Matched artist name
      - Album release date
      - Album type (album / single / compilation)
      - Explicit flag
      - Track popularity
      - Artist ID (for follow-up artist query)

    @param sp
        Authenticated `spotipy.Spotify` client.
    @param song
        Song title string from the CSV.
    @param artist
        Artist string from the CSV.

    @return dict or None
        A dictionary with keys:
            "spotify_id", "matched_title", "matched_artist",
            "release_date", "album_type", "explicit",
            "track_popularity", "artist_id"
        or None if no track is found.
    """
    query = f"track:{song} artist:{artist}"
    result = sp.search(q=query, type="track", limit=1)

    if not result["tracks"]["items"]:
        return None

    item = result["tracks"]["items"][0]

    return {
        "spotify_id": item["id"],
        "matched_title": item["name"],
        "matched_artist": item["artists"][0]["name"],
        "release_date": item["album"]["release_date"],
        "album_type": item["album"]["album_type"],
        "explicit": item["explicit"],
        "track_popularity": item["popularity"],
        "artist_id": item["artists"][0]["id"]
    }


def get_artist_info(sp, artist_id, cache):
    """
    @brief Fetch artist-level metadata (popularity + genres).

    To avoid hitting the Spotify API repeatedly for the same artist,
    this function maintains a simple in-memory cache (a dict) keyed by
    `artist_id`.

    @param sp
        Authenticated `spotipy.Spotify` client.
    @param artist_id
        Spotify artist ID string.
    @param cache
        Dict used as a memoization cache:
            - keys: artist IDs (str)
            - values: dict with "artist_popularity" and "genres"

    @return dict
        A dictionary with keys:
            - "artist_popularity": int or None
            - "genres": comma-separated string or None
    """
    # Check cache first
    if artist_id in cache:
        return cache[artist_id]

    try:
        artist = sp.artist(artist_id)
        info = {
            "artist_popularity": artist["popularity"],
            "genres": ", ".join(artist["genres"]) if artist["genres"] else None
        }
    except Exception as e:
        # If anything goes wrong (network, rate limit, etc.),
        # fall back to None values so the pipeline still runs.
        print(f"  [WARN] Error fetching artist {artist_id}: {e}")
        info = {
            "artist_popularity": None,
            "genres": None
        }

    # Save in cache for future lookups
    cache[artist_id] = info
    return info


# -------------------------------------------------------------------
# 3. Main script: enrich spotify.csv
# -------------------------------------------------------------------

def main():
    """
    @brief Enrich the local `spotify.csv` file with extra Spotify metadata.

    High-level steps:
      1. Load `spotify.csv`.
      2. Drop columns we don't want to keep (`Unnamed: 0`, `target`) if present.
      3. Make sure we have `song_title` and `artist` columns.
      4. For each row:
            - Query Spotify for track info.
            - Query Spotify for artist info (with caching).
            - Append all metadata to a list of dicts.
      5. Concatenate the original DataFrame with the new metadata.
      6. Save the enriched dataset as `spotify_enriched.csv`.
    """
    print("\nLoading original dataset...")
    df = pd.read_csv("spotify.csv")

    # Remove unnecessary columns (if they exist)
    for col in ["Unnamed: 0", "target"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure needed columns exist
    if not {"song_title", "artist"}.issubset(df.columns):
        raise ValueError("spotify.csv must have 'song_title' and 'artist' columns.")

    # Create Spotify client
    sp = get_spotify_client()

    # This will store one dict per row in df
    extra_data = []

    # Cache for artist lookups to avoid duplicate API calls
    artist_cache = {}

    n = len(df)
    print(f"Enriching {n} songs with Spotify metadata...\n")

    # ----------------------------------------------------------------
    # Loop over dataset and enrich each row with Spotify info
    # ----------------------------------------------------------------
    for idx, row in df.iterrows():
        song = row["song_title"]
        artist = row["artist"]

        print(f"[{idx + 1}/{n}] {song} — {artist}")

        # ------------------------------------------------------------
        # Track info (search by song + artist)
        # ------------------------------------------------------------
        try:
            track = get_track_info(sp, song, artist)
        except Exception as e:
            # If search fails completely, log and push a "missing" record
            print(f"  [WARN] Error fetching track: {e}")
            extra_data.append({
                "spotify_id": None,
                "matched_title": None,
                "matched_artist": None,
                "release_date": None,
                "album_type": None,
                "explicit": None,
                "track_popularity": None,
                "artist_popularity": None,
                "genres": None
            })
            time.sleep(0.3)
            continue

        # If no result found, store Nones so row counts stay aligned
        if track is None:
            print("  [INFO] Track not found.")
            extra_data.append({
                "spotify_id": None,
                "matched_title": None,
                "matched_artist": None,
                "release_date": None,
                "album_type": None,
                "explicit": None,
                "track_popularity": None,
                "artist_popularity": None,
                "genres": None
            })
            time.sleep(0.3)
            continue

        # Small sleep to be gentle with the API
        time.sleep(0.3)

        # ------------------------------------------------------------
        # Artist info (popularity + genres, with caching)
        # ------------------------------------------------------------
        artist_info = get_artist_info(sp, track["artist_id"], artist_cache)

        # Another small pause to avoid being too aggressive
        time.sleep(0.3)

        # ------------------------------------------------------------
        # Append final enriched row
        # ------------------------------------------------------------
        extra_data.append({
            "spotify_id": track["spotify_id"],
            "matched_title": track["matched_title"],
            "matched_artist": track["matched_artist"],
            "release_date": track["release_date"],
            "album_type": track["album_type"],
            "explicit": track["explicit"],
            "track_popularity": track["track_popularity"],
            "artist_popularity": artist_info["artist_popularity"],
            "genres": artist_info["genres"]
        })

    # ----------------------------------------------------------------
    # Combine original data with enrichment and save
    # ----------------------------------------------------------------
    extra_df = pd.DataFrame(extra_data)
    df_enriched = pd.concat([df.reset_index(drop=True), extra_df], axis=1)

    output_file = "spotify_enriched.csv"
    df_enriched.to_csv(output_file, index=False)
    print(f"\nDone! Enriched dataset written to '{output_file}'.\n")


if __name__ == "__main__":
    main()
