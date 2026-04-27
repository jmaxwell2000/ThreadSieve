# Sample Long Mixed Conversation
**Chat ID:** fixture-long-mixed-conversation
**Date:** 2026-01-05

### User (2026-01-05 08:00:00)
I want to sketch a product workflow for a local memory extractor. It should turn long conversations into small reusable notes, but I do not want it to become a notes app.

### Assistant (2026-01-05 08:00:03)
The product sounds like a pipeline between raw conversation files and downstream knowledge tools.

### User (2026-01-05 08:01:00)
Yes. Decision: source files should stay where the user keeps them. The tool should read them and write extracted objects somewhere else.

### Assistant (2026-01-05 08:01:03)
Decision captured: source ownership stays with the user, output is written separately.

### User (2026-01-05 08:02:00)
I also want a fake protocol fixture for testing, something safe and generic.

### Assistant (2026-01-05 08:02:05)
You can use a synthetic protocol with ordinary directives.

### User (2026-01-05 08:03:00)
Compact Review Protocol: REVIEW_NOTE_MODE

Directives:

Source First: Every extracted note must point back to the original message.

Small Objects: Split unrelated ideas into separate records.

Review Flag: Route uncertain objects to a review folder.

No Private Data: Fixtures must avoid real names, secrets, and private transcripts.

### Assistant (2026-01-05 08:03:05)
Protocol noted. It can test framework extraction without using private material.

### User (2026-01-05 08:04:00)
Give examples of how a user might use this review protocol.

### Assistant (2026-01-05 08:04:05)
Example One:
User: "Extract decisions from this planning chat."
AI: "Two decisions found. Both cite source messages."

Example Two:
User: "Find unresolved questions."
AI: "Three questions found. One low confidence."

Continue?

### User (2026-01-05 08:04:30)
Continue.

### Assistant (2026-01-05 08:04:35)
Example Three:
User: "Separate tasks from product ideas."
AI: "Created two tasks and one product concept."

Example Four:
User: "Show me why this note exists."
AI: "Rationale and source references attached."

### User (2026-01-05 08:05:00)
Draft a short extraction prompt for this product.

### Assistant (2026-01-05 08:05:05)
Prompt draft:

1. Summarize the whole conversation.
2. Extract durable ideas, decisions, tasks, and questions.
3. Prefer user-authored content.
4. Mention source messages when useful.
5. Add a friendly paragraph explaining every note.

### User (2026-01-05 08:06:00)
Revise that. Do not summarize the whole conversation first. Make source messages mandatory, and remove the friendly explanation.

### Assistant (2026-01-05 08:06:05)
Updated prompt:

Extract durable ideas, decisions, tasks, and questions. Prefer user-authored content. Cite source messages for every item.

### User (2026-01-05 08:07:00)
Open question: should low confidence notes be written directly to the main folder or routed somewhere separate?

### Assistant (2026-01-05 08:07:05)
Separate routing reduces clutter and makes review easier.

### User (2026-01-05 08:08:00)
Okay, task: add a review folder later, but don't build a complicated review UI yet.

### Assistant (2026-01-05 08:08:04)
Task noted: add review folder before building a complex UI.

### User (2026-01-05 08:09:00)
Wait, capture this messy part too. If I trail off or restart a sentence, the system should not invent the missing thought.

### Assistant (2026-01-05 08:09:04)
That is a requirement for conservative extraction from voice-like transcripts.

### User (2026-01-05 08:10:00)
Right, and the assistant's examples should help explain my next move, but they should not become my idea unless I adopt them.
