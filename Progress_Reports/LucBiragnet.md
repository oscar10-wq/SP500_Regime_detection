## [2026-02-14] Data Loading Setup for FRED API

### New Dependencies Installed
- `fredapi`: official Python wrapper for the St. Louis Fed API.
  - **Why?** - Much easier to use an existing wrapper than to build everything from scratch.
- `python-dotenv`: for securely managing environment variable (API keys).

### Configuration Changes
- Configured .gitignore to prevent leaking API keys.
- Added config directory within `data/scripts` to hold .env API key files.

### Data Pipeline Changes
- `BaseDataLoader` interface for fetching data from multiple sources, `FredDataLoader` implementation to fetch data from the Federal Reserve API.
  - **Why?** - Easily extendable for future data sources (e.g., `YahooDataLoader`).