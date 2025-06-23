# 🔍 Multi-Agent Information Assistant

This is a multi-agent system powered by OpenAI Agents SDK that routes user queries to the most appropriate answering agent based on intent, sensitivity to recency, and safety constraints. It combines real-time search capabilities with LLM-based reasoning while enforcing guardrails to ensure relevance, safety, and integrity.

---

## 🧠 Agents Overview

### 🧾 Answering Agents

- **Brave Search Agent**

  - Performs live web searches using the Brave Search API.
  - Returns high-confidence answers if the top 3 results agree.
  - If the results conflict, it provides a nuanced comparison.
  - If no quality results are found, it gracefully declines to answer.

- **LLM Agent**
  - Answers general knowledge questions that **do not require recency**.
  - Assumes a knowledge cut-off of `2024-10-01`.

### 🔀 Routing & Guardrails

- **Triage Agent**

  - Determines which agent (LLM or Brave) should handle a given query.
  - Routes based on whether the question depends on recent information.

- **Relevance Agent**

  - Ensures the question is relevant to the assistant’s scope (e.g., knowledge-seeking or web-based).

- **Jailbreak Agent**

  - Detects prompt injection or attempts to exploit the system.

- **NSFW Agent**
  - Flags unsafe or inappropriate content.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/lionelgwk/search-ai-agents.git
cd search-ai-agents
```

### 2. (Optional) Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

## ⚙️ Configuration

Make sure to set your environment variables (e.g., OpenAI API key, Brave Search key) in a .env file:

```bash
BRAVE_API_KEY=BRAVE_API_KEY_HERE
OPENAI_API_KEY=OPENAI_API_KEY_HERE
```

You can find the example in the `.env.example` file.

You can get a Brave Search API key from [here](https://brave.com/search/api/).

You can get a OpenAI API key from [here](https://platform.openai.com/api-keys).

## 🙌 Credits

Built with ❤️ using the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).
