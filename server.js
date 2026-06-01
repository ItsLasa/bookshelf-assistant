import express from "express";
import { ApolloServer } from "@apollo/server";
import cors from "cors";
import fs from "fs";
import OpenAI from "openai";
import { pipeline } from "@xenova/transformers";
import "dotenv/config";

// Load our book data
const books = JSON.parse(fs.readFileSync("./books.json", "utf-8"));

// OpenRouter client (OpenAI-compatible)
const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});

// ---------- PART A: GraphQL (structured data) ----------
const typeDefs = `#graphql
  type Author { name: String, country: String }
  type Book {
    id: ID
    title: String
    author: Author
    genre: String
    year: Int
    rating: Float
    status: String
    notes: String
  }
  type Query {
    books(genre: String, status: String): [Book]
  }
`;

const resolvers = {
  Query: {
    books: (_, { genre, status }) =>
      books.filter(
        (b) =>
          (!genre || b.genre === genre) &&
          (!status || b.status === status)
      ),
  },
};

// ---------- PART B: RAG (local embeddings) ----------
let embedder = null;
async function getEmbedder() {
  if (!embedder) {
    console.log("Loading local embedding model (first run may take a minute)...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return embedder;
}

async function embed(text) {
  const model = await getEmbedder();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

let bookEmbeddings = [];
async function buildEmbeddings() {
  console.log("Building embeddings for your books...");
  for (const book of books) {
    const vector = await embed(`${book.title}: ${book.notes}`);
    bookEmbeddings.push({ book, vector });
  }
  console.log("Embeddings ready!");
}

async function retrieve(question, topK = 3) {
  const qVector = await embed(question);
  return bookEmbeddings
    .map((item) => ({
      book: item.book,
      score: cosineSimilarity(qVector, item.vector),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((item) => item.book);
}

// ---------- PART C: The Agent ----------
async function askAgent(question) {
  const lower = question.toLowerCase();
  let context = "";

  if (lower.includes("finished") || lower.includes("reading") || lower.includes("to-read")) {
    const status = lower.includes("finished")
      ? "Finished"
      : lower.includes("reading")
      ? "Reading"
      : "To-Read";
    const matches = books.filter((b) => b.status === status);
    context = matches.map((b) => `${b.title} by ${b.author.name}`).join("\n");
  } else {
    const relevant = await retrieve(question);
    context = relevant
      .map((b) => `${b.title} by ${b.author.name}: ${b.notes}`)
      .join("\n");
  }

  const completion = await openai.chat.completions.create({
model: "openrouter/auto", // free tier on OpenRouter
    messages: [
      {
        role: "system",
        content:
          "You are a helpful bookshelf assistant. Answer ONLY using the book context provided. If the answer isn't there, say so.",
      },
      { role: "user", content: `Books:\n${context}\n\nQuestion: ${question}` },
    ],
  });
  return completion.choices[0].message.content;
}

// ---------- Wire up the server ----------
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static("public"));

// GraphQL via executeOperation (works with Apollo Server v5)
const apollo = new ApolloServer({ typeDefs, resolvers });
await apollo.start();

app.post("/graphql", async (req, res) => {
  const result = await apollo.executeOperation({
    query: req.body.query,
    variables: req.body.variables ?? {},
  });
  res.json(result.body.singleResult);
});

app.post("/ask", async (req, res) => {
  try {
    const answer = await askAgent(req.body.question);
    res.json({ answer });
  } catch (e) {
    res.status(500).json({ answer: "Error: " + e.message });
  }
});

await buildEmbeddings();
app.listen(4000, () =>
  console.log("🚀 Open http://localhost:4000 in your browser")
);