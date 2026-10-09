# Comment Pulse

An interactive, CPU-only YouTube comment analyzer built with Streamlit. It downloads public comments through the official YouTube Data API, scores each comment locally with VADER, and provides a Gemini-powered LangChain chatbot grounded in BM25-retrieved comments.
## View Deployed version

https://youtubecommentanalysis-thisisatestbtw.streamlit.app
## Features

- Accepts standard YouTube, `youtu.be`, Shorts, Live, embed URLs, or a raw video ID
- Retrieves top-level comments and optional replies with pagination
- Shows sentiment mix, score distribution, audience mood over time, common terms, searchable comments, and CSV export
- Uses LangChain and Gemini to answer questions with `[C1]`-style comment citations
- Runs sentiment analysis and retrieval entirely on CPU; no CUDA, PyTorch, or TensorFlow packages

## Setup

Python 3.13 or newer is required. From the project directory, create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the project and its test dependency with pip:

```bash
python -m pip install --upgrade pip
```

Create a [YouTube Data API v3 key](https://console.cloud.google.com/apis/library/youtube.googleapis.com) and a free-tier [Gemini API key in Google AI Studio](https://aistudio.google.com/app/apikey). The Gemini key is only required for the chatbot. You can paste both keys into the app, set environment variables, or create `.streamlit/secrets.toml`:

```toml
YOUTUBE_API_KEY = "your-youtube-key"
GEMINI_API_KEY = "your-gemini-key"
```

The secrets file is ignored by Git.

## Run

With the virtual environment activated:

```bash
python -m streamlit run main.py
```

Open <http://localhost:8501>, paste a public video URL, and select **Analyze comments**.

## Notes

- The YouTube Data API key needs access to YouTube Data API v3. Private videos and videos with disabled comments cannot be analyzed.
- VADER is a fast lexicon-based model optimized for English social text. Scores for other languages may be less reliable.
- Chat answers are based on the analyzed sample, not every viewer or every comment when a maximum is configured.
- The default `gemini-3.5-flash-lite` model supports the Gemini API free tier. Free-tier availability and rate limits are controlled by Google and may vary by region and account.

## Test

```bash
python -m pytest
```
