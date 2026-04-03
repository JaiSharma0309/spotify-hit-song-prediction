"""Enrich a local Spotify song dataset with track and artist metadata."""

import os
import time
from pathlib import Path

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INPUT_PATH = DATA_DIR / "spotify.csv"
OUTPUT_PATH = DATA_DIR / "spotify_enriched.csv"


def get_spotify_client():
    """Create an authenticated Spotify client using environment variables."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET before running this script."
        )

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def get_track_info(sp, song, artist):
    """Fetch track-level metadata for a `(song, artist)` pair."""
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
        "artist_id": item["artists"][0]["id"],
    }


def get_artist_info(sp, artist_id, cache):
    """Fetch artist-level metadata and reuse cached lookups when possible."""
    if artist_id in cache:
        return cache[artist_id]

    try:
        artist = sp.artist(artist_id)
        info = {
            "artist_popularity": artist["popularity"],
            "genres": ", ".join(artist["genres"]) if artist["genres"] else None,
        }
    except Exception as exc:
        print(f"  [WARN] Error fetching artist {artist_id}: {exc}")
        info = {
            "artist_popularity": None,
            "genres": None,
        }

    cache[artist_id] = info
    return info


def build_missing_record():
    """Return a placeholder record when Spotify data cannot be fetched."""
    return {
        "spotify_id": None,
        "matched_title": None,
        "matched_artist": None,
        "release_date": None,
        "album_type": None,
        "explicit": None,
        "track_popularity": None,
        "artist_popularity": None,
        "genres": None,
    }


def main():
    """Enrich `data/spotify.csv` with Spotify metadata and save the result."""
    print("\nLoading original dataset...")
    df = pd.read_csv(INPUT_PATH)

    for col in ["Unnamed: 0", "target"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if not {"song_title", "artist"}.issubset(df.columns):
        raise ValueError("data/spotify.csv must have 'song_title' and 'artist' columns.")

    sp = get_spotify_client()
    extra_data = []
    artist_cache = {}

    n_rows = len(df)
    print(f"Enriching {n_rows} songs with Spotify metadata...\n")

    for idx, row in df.iterrows():
        song = row["song_title"]
        artist = row["artist"]

        print(f"[{idx + 1}/{n_rows}] {song} — {artist}")

        try:
            track = get_track_info(sp, song, artist)
        except Exception as exc:
            print(f"  [WARN] Error fetching track: {exc}")
            extra_data.append(build_missing_record())
            time.sleep(0.3)
            continue

        if track is None:
            print("  [INFO] Track not found.")
            extra_data.append(build_missing_record())
            time.sleep(0.3)
            continue

        time.sleep(0.3)
        artist_info = get_artist_info(sp, track["artist_id"], artist_cache)
        time.sleep(0.3)

        extra_data.append({
            "spotify_id": track["spotify_id"],
            "matched_title": track["matched_title"],
            "matched_artist": track["matched_artist"],
            "release_date": track["release_date"],
            "album_type": track["album_type"],
            "explicit": track["explicit"],
            "track_popularity": track["track_popularity"],
            "artist_popularity": artist_info["artist_popularity"],
            "genres": artist_info["genres"],
        })

    extra_df = pd.DataFrame(extra_data)
    df_enriched = pd.concat([df.reset_index(drop=True), extra_df], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Enriched dataset written to '{OUTPUT_PATH}'.\n")


if __name__ == "__main__":
    main()

