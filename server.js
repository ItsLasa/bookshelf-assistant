import express from "express";
import { ApolloServer } from "@apollo/server";
import cors from "cors";
import fs from "fs";
import OpenAI from "openai";
import { pipeline } from "@xenova/transformers";
import "dotenv/config";

// Load our book data
const books = JSON.parse(fs.readFileSync("./books.json", "utf-8"));

// ---------- Book cover images (from OpenLibrary, cached locally) ----------
const COVER_CACHE_FILE = "./book-covers.json";
let bookCovers = {};
if (fs.existsSync(COVER_CACHE_FILE)) {
  bookCovers = JSON.parse(fs.readFileSync(COVER_CACHE_FILE, "utf-8"));
  console.log(`Loaded ${Object.keys(bookCovers).length} cached covers`);
}
async function fetchCover(id, title, author) {
  const url = `https://openlibrary.org/search.json?q=${encodeURIComponent(title + " " + author)}&fields=cover_i&limit=1`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (data.docs?.[0]?.cover_i) {
      return `https://covers.openlibrary.org/b/id/${data.docs[0].cover_i}-M.jpg`;
    }
  } catch {}
  return null;
}
async function buildCovers() {
  if (Object.keys(bookCovers).length === books.length) return;
  console.log("Fetching missing book covers from OpenLibrary...");
  const concurrency = 3;
  let fetched = 0;
  for (let i = 0; i < books.length; i += concurrency) {
    const batch = books.slice(i, i + concurrency);
    const results = await Promise.allSettled(
      batch.map((b) => fetchCover(b.id, b.title, b.author.name))
    );
    batch.forEach((b, j) => {
      if (results[j].status === "fulfilled" && results[j].value) {
        bookCovers[b.id] = results[j].value;
        fetched++;
      }
    });
    await new Promise((r) => setTimeout(r, 200));
  }
  console.log(`Fetched ${fetched} new covers`);
  fs.writeFileSync(COVER_CACHE_FILE, JSON.stringify(bookCovers, null, 2));
}
function bookToCard(b) {
  const cover = bookCovers[b.id] || null;
  return { id: b.id, title: b.title, author: b.author.name, genre: b.genre, year: b.year, rating: b.rating, status: b.status, cover };
}

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
const GENRE_KEYWORDS = {
  "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "science fiction": "Sci-Fi",
  "fantasy": "Fantasy",
  "dystopian": "Dystopian",
  "cyberpunk": "Cyberpunk",
  "sci-fi comedy": "Sci-Fi Comedy",
  "post-apocalyptic": "Post-Apocalyptic",
  "fantasy comedy": "Fantasy Comedy",
  "classic": "Classic",
  "magical realism": "Magical Realism",
  "philosophical fiction": "Philosophical Fiction",
  "historical fiction": "Historical Fiction",
  "literary fiction": "Literary Fiction",
  "mythological fiction": "Mythological Fiction",
  "thriller": "Thriller",
  "crime": "Crime",
  "mystery": "Mystery",
  "self-help": "Self-Help",
  "productivity": "Productivity",
  "finance": "Finance",
  "business": "Business",
  "psychology": "Psychology",
  "history": "History",
  "science": "Science",
  "memoir": "Memoir",
  "philosophy": "Philosophy",
};

function detectGenre(question) {
  const lower = question.toLowerCase();
  for (const [keyword, genre] of Object.entries(GENRE_KEYWORDS)) {
    if (lower.includes(keyword)) return genre;
  }
  return null;
}

async function askAgent(question) {
  const lower = question.toLowerCase();
  let context = "";
  let allResults = false;
  let matchedBooks = null;

  const genre = detectGenre(question);
  const status = lower.includes("finished")
    ? "Finished"
    : lower.includes("reading")
    ? "Reading"
    : lower.includes("to-read")
    ? "To-Read"
    : null;

  if (genre && status) {
    matchedBooks = books.filter((b) => b.genre === genre && b.status === status);
    allResults = matchedBooks.length > 3;
    context = matchedBooks.map((b) => `${b.title} by ${b.author.name}`).join("\n");
  } else if (genre) {
    matchedBooks = books.filter((b) => b.genre === genre);
    allResults = matchedBooks.length > 3;
    context = matchedBooks.map((b) => `${b.title} by ${b.author.name} (${b.status})`).join("\n");
  } else if (status) {
    matchedBooks = books.filter((b) => b.status === status);
    allResults = matchedBooks.length > 3;
    context = matchedBooks.map((b) => `${b.title} by ${b.author.name}`).join("\n");
  } else {
    const relevant = await retrieve(question);
    context = relevant
      .map((b) => `${b.title} by ${b.author.name}: ${b.notes}`)
      .join("\n");
  }

  const systemPrompt = allResults
    ? "You are a helpful bookshelf assistant. The context contains ALL matching books. List every single one of them in your answer. Do not skip any."
    : "You are a helpful bookshelf assistant. Answer ONLY using the book context provided. If the answer isn't there, say so.";

  const completion = await openai.chat.completions.create({
    model: "openrouter/auto",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: `Books:\n${context}\n\nQuestion: ${question}` },
    ],
  });
  return {
    answer: completion.choices[0].message.content,
    books: matchedBooks ? matchedBooks.map(bookToCard) : null,
  };
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
    const result = await askAgent(req.body.question);
    res.json({ answer: result.answer, books: result.books });
  } catch (e) {
    res.status(500).json({ answer: "Error: " + e.message, books: null });
  }
});

await Promise.all([buildEmbeddings(), buildCovers()]);
app.listen(4000, () =>
  console.log("🚀 Open http://localhost:4000 in your browser")
);