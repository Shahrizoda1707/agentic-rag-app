"use client";
import { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  steps?: string[];
  sources?: string[];
};

// Render'dagi backend manzili environment variable orqali beriladi.
// Vercel'da Settings -> Environment Variables ichida
// NEXT_PUBLIC_API_URL ni backend manzilingizga tenglashtiring.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

const STEP_LABELS: Record<string, string> = {
  retrieve: "🔍 Retrieve",
  grade_documents: "📊 Grade",
  web_search: "🌐 Web search",
  web_search_skipped: "🌐 Web search (skipped)",
  generate: "✍️ Generate",
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          steps: data.steps,
          sources: data.sources,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Xatolik yuz berdi: ${(err as Error).message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <h1 className="text-xl font-semibold mb-4">Agentic RAG Chat</h1>
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            <div
              className={`inline-block px-4 py-2 rounded-2xl max-w-[80%] ${
                m.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
              }`}
            >
              {m.content}
            </div>
            {m.role === "assistant" && m.steps && m.steps.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {m.steps.map((s, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-2 py-1 rounded-full bg-indigo-100 text-indigo-700"
                  >
                    {STEP_LABELS[s] || s}
                  </span>
                ))}
              </div>
            )}
            {m.role === "assistant" && m.sources && m.sources.length > 0 && (
              <div className="text-xs text-gray-500 mt-1">
                Manbalar:{" "}
                {m.sources.map((src, idx) => (
                  <span key={idx} className="underline mr-2">
                    [{idx + 1}] {src}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="text-gray-400 text-sm">Javob tayyorlanmoqda…</div>}
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 border rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Savolingizni yozing…"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-blue-600 text-white px-5 py-2 rounded-full disabled:opacity-50"
        >
          Yuborish
        </button>
      </div>
    </main>
  );
}
