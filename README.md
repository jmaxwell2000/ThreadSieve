# ThreadSieve

ThreadSieve is a local-first CLI for turning AI conversations, copied threads, and exported chats into durable, tagged, source-linked knowledge objects.

It is built to be boringly portable: plain Markdown notes, local source archives, JSONL, and SQLite FTS search. The MVP runs with zero required runtime dependencies beyond Python 3.11+.

## What Works Now

- `threadsieve init`
- `threadsieve ingest ./chat.json`
- `threadsieve extract --thread latest`
- `threadsieve extract --file ./thread.md`
- `threadsieve extract --clipboard`
- `threadsieve search "knowledge management"`
- `threadsieve open OBJECT_ID`
- ChatGPT export JSON importer
- Plain text / Markdown transcript importer
- Local canonical source archives
- Markdown notes with YAML-style frontmatter and message-span provenance
- SQLite full-text search index
- OpenAI-compatible extraction adapter
- Offline deterministic fallback extraction so the tool works before model setup

## Mac Quick Start

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
threadsieve init --workspace ~/ThreadSieve
threadsieve extract --file examples/thread.md
threadsieve search provenance
```

Zero-install from a checkout also works:

```bash
chmod +x bin/threadsieve
./bin/threadsieve init --workspace ~/ThreadSieve
./bin/threadsieve extract --file examples/thread.md
./bin/threadsieve search provenance
```

If you do not want a virtual environment, use `pipx`:

```bash
brew install pipx
pipx ensurepath
pipx install .
threadsieve init --workspace ~/ThreadSieve
```

## Use Your Clipboard

Copy a chat transcript, then run:

```bash
threadsieve extract --clipboard
```

ThreadSieve uses the native clipboard command when available:

- macOS: `pbpaste`
- Linux Wayland: `wl-paste`
- Linux X11: `xclip` or `xsel`
- Windows: PowerShell `Get-Clipboard`

No Python clipboard package is required. If no clipboard tool is available, paste the transcript into a text file and run:

```bash
threadsieve extract --file ./thread.md
```

## Cross-Platform File Opening

`threadsieve open OBJECT_ID` uses the native file opener:

- macOS: `open`
- Linux: `xdg-open`
- Windows: `os.startfile`

On any platform, this always works without a GUI:

```bash
threadsieve open OBJECT_ID --print
```

## Use a Local Model Through Ollama

ThreadSieve talks to OpenAI-compatible `/chat/completions` APIs. Ollama can expose that endpoint.

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:14b
```

In another terminal:

```bash
threadsieve init --workspace ~/ThreadSieve
```

Edit `~/.threadsieve/config.json`:

```json
{
  "workspace": "~/ThreadSieve",
  "confidence_threshold": 0.55,
  "models": {
    "extract": {
      "provider": "openai-compatible",
      "base_url": "http://localhost:11434/v1",
      "model": "qwen2.5:14b",
      "api_key_env": "THREADSIEVE_API_KEY"
    }
  },
  "redaction": {
    "enabled": false,
    "patterns": []
  }
}
```

Then run:

```bash
threadsieve extract --file examples/thread.md
```

## Use a Hosted OpenAI-Compatible API

Set your key:

```bash
export THREADSIEVE_API_KEY="paste-your-key-here"
```

Edit `~/.threadsieve/config.json` with your provider:

```json
{
  "workspace": "~/ThreadSieve",
  "confidence_threshold": 0.55,
  "models": {
    "extract": {
      "provider": "openai-compatible",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4.1-mini",
      "api_key_env": "THREADSIEVE_API_KEY"
    }
  }
}
```

## Output Layout

```text
~/ThreadSieve/
  Knowledge/
    Ideas/
    Decisions/
    Open Loops/
    Tasks/
    Drafts/
  Sources/
    chatgpt/
      THREAD_ID-title/
        thread.json
        thread.md
        manifest.json
  objects.jsonl
  index.sqlite
```

Every note includes:

- object ID
- type
- tags
- confidence
- source app
- source thread ID
- local archived thread path
- message IDs
- character spans

## Import Formats

ThreadSieve currently accepts:

- ChatGPT conversation export JSON objects with `mapping`
- normalized JSON with a `messages` array
- JSON lists of message objects
- Markdown or text transcripts with role prefixes like `User:` and `Assistant:`
- raw text as one source message

## Project Goals

ThreadSieve is not trying to become a notes app. It is the extraction, provenance, and recall layer between temporary conversations and the knowledge system you already use.

The near-term direction:

- richer extraction schemas
- better validation of cited spans
- conservative deduplication
- Obsidian-friendly links
- watch folders
- MCP server
- browser capture

## Development

Run tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

The code intentionally starts with the Python standard library. Add runtime dependencies only when they clearly improve portability or correctness.
