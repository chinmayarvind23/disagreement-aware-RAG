import React, { useState, useEffect } from "react";
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

export default function Home() {
  const [q, setQ] = useState("");
  const [resp, setResp] = useState(null);
  const [curve, setCurve] = useState([]);
  const [auc, setAuc] = useState(0);
  useEffect(() => {
    axios.get("/metrics").then(({ data }) => {
      const pts = data.coverage_curve.map((p) => ({
        coverage: p.coverage,
        halluc: p.halluc_rate,
      }));
      setCurve(pts);
      setAuc(data.roc_auc);
    });
  }, []);

  const ask = async () => {
    if (!q) return;
    const r = await axios.post("/api/qa", { query: q });
    setResp(r.data);
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-6 text-center">
        Disagreement-Aware RAG Demo
      </h1>

      {/* --- QA Box --- */}
      <div className="mb-8">
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask a question…"
          className="border px-3 py-2 w-full mb-2"
        />
        <button
          onClick={ask}
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Ask
        </button>
        {resp && (
          <pre className="mt-4 p-4 bg-gray-100 rounded">
            {JSON.stringify(resp, null, 2)}
          </pre>
        )}
      </div>

      <hr className="my-8" />

      {/* --- Metrics Dashboard --- */}
      <h2 className="text-2xl font-semibold mb-4">Metrics Dashboard</h2>
      <p className="mb-4">
        ROC AUC on held-out test set:{" "}
        <span className="font-mono text-lg">{auc.toFixed(3)}</span>
      </p>

      {curve.length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={curve}
            margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
          >
            <XAxis
              dataKey="coverage"
              name="Coverage"
              label={{ value: "Coverage", position: "insideBottom", dy: 10 }}
            />
            <YAxis
              dataKey="halluc"
              name="Hallucination Rate"
              label={{
                value: "Hallucination Rate",
                angle: -90,
                position: "insideLeft",
                dx: -10,
              }}
            />
            <Tooltip
              formatter={(val) => val.toFixed(3)}
              cursor={{ strokeDasharray: "3 3" }}
            />
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
      )}
    </div>
);
}
