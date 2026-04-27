# ThreadSieve

ThreadSieve is a local-first CLI for turning AI conversations, copied threads, and exported chats into durable, tagged, source-linked knowledge objects.

It is built to be boringly portable: plain Markdown notes, local source archives, JSONL, and SQLite FTS search. The MVP runs with zero required runtime dependencies beyond Python 3.11+.

## What Works Now

- `threadsieve init`
- `threadsieve ingest ./chat.json`
- `threadsieve extract --thread latest`
- `threadsieve extract --file ./thread.md`
- `threadsieve extract --source ./chats --out ./knowledge`
- `threadsieve trace OBJECT_ID --knowledge ./knowledge`
- `threadsieve index ./knowledge`
- `threadsieve regression`
- `threadsieve eval`
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

The handoff-style pipeline command writes directly to a user-chosen knowledge folder:

```bash
./bin/threadsieve extract --source examples/thread.md --out ./knowledge
./bin/threadsieve trace OBJECT_ID --knowledge ./knowledge
```

Run the privacy-safe regression fixture tests from a source checkout:

```bash
./bin/threadsieve regression
```

Run a privacy-safe live model eval against synthetic fixtures only:

```bash
./bin/threadsieve eval
```

The default eval checks 5 synthetic fixtures against 3 current low-cost OpenRouter models. It estimates
30 model calls: semantic log plus extraction for each fixture/model pair. Use `--max-calls` to enforce
a hard budget.

Extraction now creates an intermediate semantic log by default. User messages are preserved verbatim as
`USER_STATEMENT`, while assistant messages are compressed into `AI_CONTEXT` metadata. The extractor uses
that log so user thought flow is treated as primary evidence and assistant ideas are only extracted when
the user reacts to or develops them.

When an assistant response contains a revisable artifact such as a prompt, schema, draft, code block, plan,
or requirements list, the semantic log can preserve the needed excerpt as `AI_ARTIFACT` instead of compressing
it away. This helps extraction merge refinement chains into one durable object rather than creating one note
per user edit.

Classic workspace mode writes logs here:

```text
~/ThreadSieve/Semantic Logs/
```

The `--source/--out` pipeline writes logs here:

```text
./knowledge/semantic_logs/
```

To bypass this stage:

```bash
threadsieve extract --file examples/thread.md --no-semantic-log
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
threadsieve configure-provider ollama --model qwen2.5:14b
threadsieve doctor
threadsieve test-provider
threadsieve extract --file examples/thread.md
```

Ollama keeps inference local, but the model still sees the thread text. Choose a model you trust for the sensitivity of your chats.

## Use OpenRouter

OpenRouter is a hosted provider router. When you use it, ThreadSieve sends the thread text you extract to OpenRouter and whichever upstream model handles the request.

Create an API key in OpenRouter, then type:

```bash
export OPENROUTER_API_KEY="paste-your-key-here"
threadsieve configure-provider openrouter --model openai/gpt-4o-mini
threadsieve doctor
threadsieve test-provider
threadsieve extract --file examples/thread.md
```

For a project-local pipeline config, copy `threadsieve.example.yaml` to `threadsieve.yaml`, edit the paths/model, then run:

```bash
threadsieve extract --config threadsieve.yaml
```

To make the key available in every new Terminal window, add it to your shell profile:

```bash
echo 'export OPENROUTER_API_KEY="paste-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

ThreadSieve stores only the environment variable name in config, not the key itself.

The generated config will look like this:

```json
{
  "workspace": "~/ThreadSieve",
  "confidence_threshold": 0.55,
  "models": {
    "extract": {
      "kind": "openai-compatible",
      "base_url": "https://openrouter.ai/api/v1",
      "model": "openai/gpt-4o-mini",
      "api_key_env": "OPENROUTER_API_KEY",
      "headers": {
        "HTTP-Referer": "https://github.com/jmaxwell2000/ThreadSieve",
        "X-Title": "ThreadSieve"
      },
      "provider": "openrouter"
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

## Use Another OpenAI-Compatible API

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
      "base_url": "https://api.example.com/v1",
      "model": "your-model-name",
      "api_key_env": "THREADSIEVE_API_KEY"
    }
  }
}
```

Useful provider commands:

```bash
threadsieve providers
threadsieve configure-provider offline
threadsieve configure-provider ollama --model qwen2.5:14b
threadsieve configure-provider openrouter --model openai/gpt-4o-mini
threadsieve configure-provider openai-compatible --base-url http://localhost:1234/v1 --model local-model
threadsieve doctor
threadsieve test-provider
```

`threadsieve doctor` does not send thread contents to any model. `threadsieve test-provider` sends only a tiny test prompt.

## Customize The Extraction Prompt

ThreadSieve creates editable prompts when you run `init`.

There are two prompt kinds:

- `extract`: controls idea, task, decision, product/project note extraction and the content that becomes Markdown notes.
- `semantic`: controls the intermediate semantic log, where user statements are preserved and assistant replies become context metadata.

Show the active prompt path and contents:

```bash
threadsieve show-prompt
threadsieve show-prompt --kind semantic
```

Show only the path:

```bash
threadsieve show-prompt --path-only
threadsieve show-prompt --kind semantic --path-only
```

On macOS, open the prompt in TextEdit:

```bash
open "$(threadsieve show-prompt --path-only)"
open "$(threadsieve show-prompt --kind semantic --path-only)"
```

Or edit it in Terminal:

```bash
nano "$(threadsieve show-prompt --path-only)"
nano "$(threadsieve show-prompt --kind semantic --path-only)"
```

The default prompt tells the model to return JSON only, extract fewer high-value objects, cite message IDs, include `exact_text` where possible, and avoid unsupported claims. After editing the prompt, run extraction normally:

```bash
threadsieve extract --clipboard
```

Restore the default prompt:

```bash
threadsieve reset-prompt --force
threadsieve reset-prompt --kind semantic --force
threadsieve reset-prompt --kind all --force
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

The `--source/--out` pipeline layout is flatter and object-type first:

```text
knowledge/
  ideas/
  tasks/
  decisions/
  questions/
  features/
  insights/
  requirements/
  risks/
  _needs_review/
  semantic_logs/
  _runs/
  .threadsieve/state.json
  index.jsonl
```

Every note includes:

- object ID
- type
- object role
- canonical statement, when the extractor can form one
- tags
- confidence
- source app
- source thread ID
- local archived thread path
- message IDs
- character spans
- typed source references, when available

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
