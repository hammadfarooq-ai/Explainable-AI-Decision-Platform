import React, { useState } from "react";
import { api } from "../api/client";

export const RagQueryForm: React.FC = () => {
  const [title, setTitle] = useState("Example Policy");
  const [text, setText] = useState("This is an example business document for the RAG pipeline.");
  const [question, setQuestion] = useState("What are the key recommendations?");
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ingest = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.post("/rag/documents", { title, text });
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Ingestion failed");
    } finally {
      setLoading(false);
    }
  };

  const ask = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.post("/rag/recommend", { question });
      setAnswer(res.data.answer);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Recommendation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h3>Ingest Document</h3>
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title"
      />
      <br />
      <textarea
        rows={3}
        cols={60}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <br />
      <button type="button" onClick={ingest} disabled={loading}>
        Ingest
      </button>

      <h3>Ask Question</h3>
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "100%" }}
      />
      <button type="button" onClick={ask} disabled={loading}>
        Ask
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {answer && (
        <div>
          <h4>Answer</h4>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
};

