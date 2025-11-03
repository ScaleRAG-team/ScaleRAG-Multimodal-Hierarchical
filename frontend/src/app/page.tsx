"use client";
import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [q, setQ] = useState("");
  const [ans, setAns] = useState("");
  const [loading, setLoading] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  const generate = () => {
    setAns(""); setLoading(true);
    esRef.current?.close();

    const es = new EventSource(`${API}/api/generate?query=${encodeURIComponent(q)}&k=6`);
    esRef.current = es;

    es.onmessage = (e) => {
      if (e.data === "[END]") { setLoading(false); es.close(); }
      else setAns((a) => a + e.data);
    };
    es.onerror = () => { setLoading(false); es.close(); setAns("Error fetching answer."); };
  };

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center p-10">
      <h1 className="text-3xl font-semibold mb-6 text-gray-800">ScaleRAG Research Assistant</h1>

      <div className="w-full max-w-xl flex flex-col gap-4">
        <textarea
          className="border rounded-md p-3 text-gray-800"
          rows={3}
          placeholder="Ask a question..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <div className="flex gap-2">
          <button
            onClick={generate}
            className="bg-blue-600 text-white px-5 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
            disabled={loading || !q.trim()}
          >
            {loading ? "Streaming..." : "Ask"}
          </button>
          <button
            onClick={() => { esRef.current?.close(); setLoading(false); }}
            className="border px-4 py-2 rounded-md"
            disabled={!loading}
          >
            Stop
          </button>
        </div>
      </div>

      <div className="mt-8 bg-white shadow-md rounded-lg p-6 w-full max-w-xl text-gray-800 whitespace-pre-wrap min-h-[120px]">
        <b>Answer:</b>
        <div className="mt-2">{ans || (loading ? "…" : "")}</div>
      </div>
    </main>
  );
}
