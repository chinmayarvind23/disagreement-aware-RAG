import { useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

export default function Home() {
  const [q, setQ] = useState("");
  const [resp, setResp] = useState();
  const [curve, setCurve] = useState([]);

  const ask = async () => {
    const r = await axios.post("/api/qa", { query: q });
    setResp(r.data);
  };
  const loadCurve = async () => {
    const tsv = await axios.get("/coverage_curve.tsv");
    const rows = tsv.data
      .split("\n")
      .slice(1)
      .map(l => { const [tau, c, h] = l.split("\t"); return { tau: +tau, coverage: +c, halluc: +h }; });
    setCurve(rows);
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Disagreement-Aware RAG Demo</h1>
      <input
        value={q}
        onChange={e => setQ(e.target.value)}
        placeholder="Ask a question…"
        className="border p-2 w-full mb-2"
      />
      <button onClick={ask} className="bg-blue-500 text-white px-4 py-2 rounded">
        Ask
      </button>
      {resp && (
        <div className="mt-4">
          <h2 className="font-semibold">Answer (decision: {resp.decision})</h2>
          <p>{resp.answer}</p>
          <h3 className="mt-2 font-semibold">Sources:</h3>
          <ul>
            {resp.sources.map((s,i) => (
              <li key={i}>{s.source}: {s.text.slice(0,120)}…</li>
            ))}
          </ul>
          <h3 className="mt-2 font-semibold">Risk:</h3>
          <pre>{JSON.stringify(resp.risk,null,2)}</pre>
        </div>
      )}
      <hr className="my-6" />
      <button onClick={loadCurve} className="bg-green-500 text-white px-4 py-2 rounded mb-4">
        Load Coverage Curve
      </button>
      {curve.length > 0 && (
        <LineChart width={600} height={300} data={curve}>
          <XAxis dataKey="tau" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="coverage" stroke="#8884d8" />
          <Line type="monotone" dataKey="halluc" stroke="#82ca9d" />
        </LineChart>
      )}
    </div>
  );
}