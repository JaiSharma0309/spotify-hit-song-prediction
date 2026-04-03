from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def test_core_project_files_exist():
    expected = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "src" / "spotify_query.py",
        ROOT / "src" / "hit_song.py",
        DATA_DIR / "spotify.csv",
        DATA_DIR / "spotify_enriched.csv",
    ]
    for path in expected:
        assert path.exists(), f"Expected file missing: {path}"


def test_enriched_dataset_has_required_columns():
    df = pd.read_csv(DATA_DIR / "spotify_enriched.csv", nrows=5)
    required_columns = {
        "song_title",
        "artist",
        "track_popularity",
        "artist_popularity",
        "release_date",
        "album_type",
        "explicit",
    }
    assert required_columns.issubset(df.columns)


def test_saved_hit_threshold_matches_documented_run():
    df = pd.read_csv(DATA_DIR / "spotify_enriched.csv")
    usable = df.dropna(subset=["track_popularity"])
    hit_threshold = usable["track_popularity"].quantile(0.75)

    assert len(usable) == 1882
    assert hit_threshold == 62.0

