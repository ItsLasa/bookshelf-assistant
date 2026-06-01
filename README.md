

#  Bookshelf Assistant

A  AI assistant for your personal book collection. Ask natural questions like *"Which of my books are about AI?"* or *"Recommend a hopeful read"* and get smart, grounded answers.

This project combines three core technologies behind modern AI apps:

-  **GraphQL** — flexible, precise queries for structured book data
-  **RAG (Retrieval-Augmented Generation)** — searches your book notes to answer fuzzy questions

## Features

- Ask questions in plain English through a simple web UI
- Smart routing: structured questions use GraphQL-style filters, meaning-based questions use RAG
- GraphQL API you can query directly at `/graphql`
- Grounded answers powered by the OpenAI API

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Node.js + Express |
| API | Apollo Server (GraphQL) |
| AI / RAG | OpenAI API (embeddings + chat) |
| Data | JSON file |
| Frontend | HTML, CSS & JavaScript |

##  Getting Started

**Prerequisites:** Node.js v18+ and an OpenAI API key.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/bookshelf-assistant.git
cd bookshelf-assistant

# 2. Install dependencies
npm install

# 3. Add your OpenAI API key to a .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 4. Start the server
node server.js
```

Then open `http://localhost:4000` in your browser. 🎉

## Project Structure

```
bookshelf-assistant/
├── books.json        # Your book data (the ontology in practice)
├── server.js         # Backend: GraphQL + RAG + agent router
├── public/
│   └── index.html    # Simple web UI
├── .env              # Your secret API key (not committed)
└── package.json      # Project config
```

##  Usage

Type a question into the box and click **Ask**. Examples:

- "Which books are about AI?" → uses RAG
- "What have I finished?" → uses structured filtering
- "Recommend a hopeful book" → uses RAG

You can also query the GraphQL API directly at `http://localhost:4000/graphql`:

```graphql
{
  books(genre: "Sci-Fi") {
    title
    author { name }
    rating
  }
}
```

##  How It Works

1. You ask a question in the UI.
2. The **agent** decides whether it's a structured or fuzzy question.
3. Structured → filtered with **GraphQL**; fuzzy → relevant notes retrieved via **RAG**.
4. The **LLM** writes an answer grounded only in the retrieved book data.
