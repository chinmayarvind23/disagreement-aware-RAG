import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

export default function Home() {
  const [q, setQ] = useState("");
  const [resp, setResp] = useState(null);
  const [curve, setCurve] = useState([]);
  const [auc, setAuc] = useState(null);
  const [loadingQA, setLoadingQA] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const { data } = await axios.get(`${API_BASE}/metrics`);
        const pts = (data.coverage_curve || []).map((p) => ({
          tau: p.tau,
          coverage: p.coverage,
          halluc: p.halluc_rate,
        }));
        setCurve(pts);
        setAuc(data.roc_auc ?? null);
      } catch (e) {
        console.error(e);
        setError("Failed to load metrics from backend.");
      }
    })();
  }, []);

  const ask = async () => {
    setError("");
    setResp(null);
    if (!q.trim()) return;
    setLoadingQA(true);
    try {
      const { data } = await axios.post(`${API_BASE}/qa`, { query: q.trim() }, {
        headers: { "Content-Type": "application/json" },
      });
      setResp(data);
    } catch (e) {
      console.error(e);
      setError("Failed to fetch answer from backend.");
    } finally {
      setLoadingQA(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-4xl font-bold mb-6 text-center">
        Disagreement-Aware RAG Demo
      </h1>

      {/* QA Box */}
      <div className="mb-10">
        <label className="block mb-2 font-medium">Ask a question</label>
        <div className="flex gap-3">
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g., What is Puerto Rico's status in the U.S.?"
            className="border px-3 py-2 flex-1 rounded"
          />
          <button
            onClick={ask}
            disabled={loadingQA}
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-60"
          >
            {loadingQA ? "Thinking…" : "Ask"}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 rounded bg-red-50 border border-red-200 text-red-700">
            {error}
          </div>
        )}

        {resp && (
          <div className="mt-6 p-4 bg-gray-50 rounded border">
            <div className="mb-4">
              <div className="font-semibold mb-1">Answer</div>
              <div>{resp.answer}</div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-3 bg-white rounded border">
                <div className="font-semibold mb-2">Decision</div>
                <div className="inline-block px-2 py-1 rounded bg-gray-100">
                  {resp.decision}
                </div>
              </div>

              <div className="p-3 bg-white rounded border">
                <div className="font-semibold mb-2">Disagreement Risk</div>
                <div className="text-sm font-mono">
                  p_disagree: {Number(resp.risk?.p_disagree ?? 0).toFixed(3)}
                </div>
                <div className="text-sm font-mono">
                  sc_var: {Number(resp.risk?.sc_var ?? 0).toFixed(3)}
                </div>
                <div className="text-sm font-mono">
                  overlap: {Number(resp.risk?.overlap ?? 0).toFixed(3)}
                </div>
                <div className="text-sm font-mono">
                  entropy_proxy: {Number(resp.risk?.entropy_proxy ?? 0).toFixed(3)}
                </div>
              </div>

              <div className="p-3 bg-white rounded border">
                <div className="font-semibold mb-2">Sources</div>
                <ul className="list-disc ml-5 text-sm">
                  {resp.sources?.map((s, i) => (
                    <li key={i}>
                      {s.title || s.id || `Source ${i + 1}`}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      <hr className="my-8" />

      {/* Metrics Dashboard */}
      <h2 className="text-2xl font-semibold mb-2">Metrics Dashboard</h2>
      <p className="mb-4">
        ROC AUC (held-out test):{" "}
        <span className="font-mono text-lg">
          {auc == null ? "—" : Number(auc).toFixed(3)}
        </span>
      </p>

      {curve.length > 0 ? (
        <ResponsiveContainer width="100%" height={420}>
          <LineChart
            data={curve}
            margin={{ top: 20, right: 30, left: 10, bottom: 12 }}
          >
            <XAxis
              dataKey="tau"
              type="number"
              domain={["dataMin", "dataMax"]}
              label={{ value: "τ (abstention threshold)", position: "insideBottom", dy: 10 }}
            />
            <YAxis
              domain={[0, 1]}
              label={{ value: "Rate", angle: -90, position: "insideLeft", dx: -10 }}
            />
            <Tooltip formatter={(v) => Number(v).toFixed(3)} />
            <Legend verticalAlign="top" height={36} />
            <Line
              name="Coverage"
              type="monotone"
              dataKey="coverage"
              stroke="#4f46e5"
              dot={{ r: 3 }}
            />
            <Line
              name="Hallucination Rate"
              type="monotone"
              dataKey="halluc"
              stroke="#059669"
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="text-sm text-gray-600">
          No curve yet. Make sure you ran: <code>python -m scripts.evals</code>
        </div>
      )}
    </div>
  );
}