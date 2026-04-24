User: I want a tiny tool that turns long AI chats into durable notes. The notes should keep links back to exact messages so I can trust where an idea came from.

Assistant: That sounds like a conversation-to-knowledge distillation pipeline. The key product bet is provenance: every extracted object needs a source thread, message ID, and span reference.

User: Could this run locally first, then optionally call an OpenAI-compatible API or Ollama?

Assistant: Yes. A good MVP is a CLI with local archives, Markdown output, SQLite search, and a provider adapter for OpenAI-compatible chat completions.
