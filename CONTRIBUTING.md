# Contributing

Thanks for your interest in improving this project.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. If you want to run the data enrichment step, export Spotify API credentials:

```bash
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
```

## Project Workflow

1. Run `python src/spotify_query.py` to enrich the raw dataset.
2. Run `python src/hit_song.py` to train models and regenerate plots.
3. Run `pytest` to execute the lightweight test suite.

## Contribution Guidelines

- Keep file paths repo-relative so scripts work from the project root.
- Prefer small, focused commits.
- Update the README if you change the workflow or outputs.
- Do not commit secrets or personal API credentials.

