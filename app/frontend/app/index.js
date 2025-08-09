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

export default function Home() {
  const [q, setQ] = useState("");
  const [resp, setResp] = useState(null);
  const [curve, setCurve] = useState([]);
  const [auc, setAuc] = useState(null);
  const [loadingQA, setLoadingQA] = useState(false);

  useEffect(() => {
    axios.get("/metrics").then(({ data }) => {
      const pts = (data.coverage_curve || []).map((p) => ({
        tau: p.tau,
        coverage: p.coverage,
        halluc: p.halluc_rate,
      }));
      setCurve(pts);
      setAuc(data.roc_auc);
    }).catch(() => {});
  }, []);

  const ask = async () => {
    if (!q.trim()) return;
    setLoadingQA(true);
    setResp(null);
    try {
      const r = await axios.post("/api/qa", { query: q });
      setResp(r.data);
    } catch (e) {
      alert("QA request failed. Is FastAPI running on :8000?");
    } finally {
      setLoadingQA(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Disagreement-Aware RAG</h1>
        <p className="muted">Answers only when evidence agrees and risk is low.</p>
      </header>

      <section className="card">
        <h2>Ask a question</h2>
        <div className="row">
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Ask about the indexed corpus…"
          />
          <button onClick={ask} disabled={loadingQA}>
            {loadingQA ? "Thinking…" : "Ask"}
          </button>
        </div>

        {resp && (
          <div className="answer">
            <div className="answer-header">
              <span className={`badge ${resp.decision === "answer" ? "ok" : "warn"}`}>
                {resp.decision === "answer" ? "Answered" : "Abstained"}
              </span>
              <span className="risk">
                p(disagree): <b>{resp.risk.p_disagree.toFixed(2)}</b> ·{" "}
                overlap: <b>{resp.risk.overlap.toFixed(2)}</b> ·{" "}
                sc_var: <b>{resp.risk.sc_var.toFixed(2)}</b> ·{" "}
                entropy: <b>{resp.risk.entropy_proxy.toFixed(2)}</b>
              </span>
            </div>

            {resp.decision === "answer" ? (
              <p className="answer-text">{resp.answer}</p>
            ) : (
              <p className="muted">Abstained (risk too high based on the head + heuristics).</p>
            )}

            <div className="sources">
              <h3>Top sources</h3>
              {(resp.sources || []).slice(0, 3).map((s, i) => (
                <div className="src" key={i}>
                  <div className="src-title">{s.title || `Source ${i + 1}`}</div>
                  <div className="src-text">
                    {(s.text || "").slice(0, 300)}
                    {(s.text || "").length > 300 ? "…" : ""}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <h2>Coverage vs Hallucination (by τ)</h2>
        {auc !== null && (
          <p className="muted">
            ROC AUC on held-out test set: <b>{auc.toFixed(3)}</b>
          </p>
        )}
        {curve.length > 0 ? (
          <div className="chart">
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={curve} margin={{ top: 12, right: 24, left: 6, bottom: 12 }}>
                <XAxis dataKey="tau" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Legend />
                <Line name="Coverage" type="monotone" dataKey="coverage" dot={false} />
                <Line name="Hallucination rate" type="monotone" dataKey="halluc" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="muted">
            Run the evaluator to generate <code>data/coverage_curve.tsv</code> and{" "}
            <code>data/test_preds.npz</code>, then refresh.
          </p>
        )}
      </section>

      <footer>
        <span className="muted">τ controls conservativeness. Higher τ → answer more, risk more.</span>
      </footer>

      <style jsx>{`
        .container { max-width: 980px; margin: 0 auto; padding: 24px;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color: #0f172a; }
        header { text-align: center; margin-bottom: 24px; }
        h1 { font-size: 32px; margin: 0 0 6px; }
        h2 { font-size: 20px; margin: 0 0 12px; }
        h3 { font-size: 16px; margin: 12px 0 8px; }
        .muted { color: #64748b; }
        .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; margin-bottom: 20px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
        .row { display: flex; gap: 10px; }
        input { flex: 1; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 12px; font-size: 14px; }
        button { background: #2563eb; color: #fff; border: none; border-radius: 10px; padding: 10px 16px; font-weight: 600; cursor: pointer; }
        button:disabled { opacity: 0.6; cursor: default; }
        .answer { margin-top: 14px; }
        .answer-header { display: flex; align-items: center; gap: 10px; justify-content: space-between; flex-wrap: wrap; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }
        .badge.ok { background: #ecfdf5; color: #065f46; border: 1px solid #10b981; }
        .badge.warn { background: #fff7ed; color: #9a3412; border: 1px solid #f59e0b; }
        .risk { color: #334155; font-size: 13px; }
        .answer-text { margin: 10px 0 6px; line-height: 1.5; white-space: pre-wrap; }
        .sources { margin-top: 10px; }
        .src { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; margin-top: 8px; }
        .src-title { font-weight: 600; margin-bottom: 6px; }
        .src-text { color: #334155; font-size: 14px; }
        .chart { margin-top: 8px; }
        footer { text-align: center; margin-top: 12px; }
        code { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }
      `}</style>
    </div>
  );
}
