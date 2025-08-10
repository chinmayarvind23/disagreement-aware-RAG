"use client";
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
const API = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const COLORS = {
  light: {
    "--background": "#ffffff",
    "--foreground": "#0f172a",
    "--muted": "#64748b",
    "--panel": "#ffffff",
    "--border": "#e5e7eb",
    "--code": "#f1f5f9",
    "--btn": "#2563eb",
    "--btn-text": "#ffffff",
    "--src-text": "#334155",
    "--bg": "#ffffff",
    "--text": "#0f172a",
  },
  dark: {
    "--background": "#0b0f19",
    "--foreground": "#e5e7eb",
    "--muted": "#a3a3a3",
    "--panel": "#0f172a",
    "--border": "#30343b",
    "--code": "rgba(0,0,0,0.45)",
    "--btn": "#3f3f46",
    "--btn-text": "#f4f4f5",
    "--src-text": "#d1d5db",
    "--bg": "#0b0f19",
    "--text": "#e5e7eb",
  },
} as const;

function applyTheme(next: "light" | "dark") {
  const root = document.documentElement;
  Object.entries(COLORS[next]).forEach(([k, v]) =>
    root.style.setProperty(k, String(v))
  );
}

type Source = { title: string; text: string };
type QAResp = {
  answer: string;
  sources: Source[];
  risk: { p_disagree: number; sc_var: number; overlap: number; entropy_proxy: number };
  decision: "answer" | "abstain";
};

export default function Home() {
  const [q, setQ] = useState("");
  const [resp, setResp] = useState<QAResp | null>(null);
  const [curve, setCurve] = useState<any[]>([]);
  const [auc, setAuc] = useState<number | null>(null);
  const [loadingQA, setLoadingQA] = useState(false);

  // light/dark theme
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("theme") : null;
    const prefersDark = typeof window !== "undefined"
      && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    const next = (saved as "light" | "dark") || (prefersDark ? "dark" : "light");
    setTheme(next);
    applyTheme(next);
  }, []);

  useEffect(() => {
    if (!theme) return;
    localStorage.setItem("theme", theme);
    applyTheme(theme);
  }, [theme]);


  useEffect(() => {
  axios.get(`${API}/metrics`)
    .then(({ data }) => {
      const pts = data.coverage_curve.map((p: any) => ({
        tau: p.tau,
        coverage: p.coverage,
        halluc: p.halluc_rate,
      }));
      setCurve(pts);
      setAuc(data.roc_auc);
    })
    .catch(console.error);
}, []);

  const ask = async () => {
  if (!q.trim()) return;
  setLoadingQA(true);
  setResp(null);
  try {
    const r = await axios.post(`${API}/qa`, { query: q });
    setResp(r.data);
  } catch (e: any) {
    console.error("QA error:", e?.response?.data || e?.message || e);
    alert("QA request failed. Check backend logs.");
  } finally {
    setLoadingQA(false);
  }
  };

  return (
    <div className={`container`}>
      <header>
        <h1>Disagreement-Aware RAG</h1>
        <div className="right">
          <button
            className="toggle"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title="Toggle light/dark theme"
          >
            {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
          </button>
        </div>
        <p className="muted">
          Calibrate answers using disagreement risk, evidence overlap, and self-consistency.
        </p>
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
              <p className="muted">
                The system abstained (risk too high based on the head + heuristics).
              </p>
            )}

            <div className="sources">
              <h3>Top sources</h3>
              {resp.sources?.slice(0, 3).map((s, i) => (
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
                <XAxis dataKey="tau" tickFormatter={(t) => t.toFixed(2)} />
                <YAxis domain={[0, 1]} />
                <Tooltip
                  labelFormatter={(t: number) => `τ = ${t.toFixed(2)}`}
                  formatter={(v: number, name) => [v.toFixed(3), name]}
                  contentStyle={{
                    backgroundColor: "var(--panel)",
                    border: "1px solid var(--border)",
                    borderRadius: 10,
                    color: "var(--text)",
                    boxShadow: "0 6px 18px rgba(0,0,0,0.25)",
                  }}
                  labelStyle={{ color: "var(--text)" }}
                  itemStyle={{ color: "var(--text)" }}
                />
                <Legend />
                <Line name="Coverage" type="monotone" dataKey="coverage" dot={false} stroke="#2563eb" />
                <Line name="Hallucination rate" type="monotone" dataKey="halluc" dot={false} stroke="#10b981" />
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
        <span className="muted">τ controls how conservative the system is. Higher τ → answer more, risk more.</span>
      </footer>

      <style jsx>{`
        :root {
          --bg: #ffffff;
          --text: #0f172a;
          --muted: #64748b;
          --panel: #ffffff;
          --border: #e5e7eb;
          --code: #f1f5f9;
          --btn: #2563eb;
          --btn-text: #ffffff;
          --src-text: #334155;
        }
        .dark {
          --bg: #0b0f19;
          --text: #e5e7eb;
          --muted: #a3a3a3;
          --panel: #0f172a;
          --border: #30343b;
          --code: rgba(0,0,0,0.45);
          --btn: #3f3f46;
          --btn-text: #f4f4f5;
          --src-text: #d1d5db;
        }
        .container {
          max-width: 980px;
          margin: 0 auto;
          padding: 24px;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          color: var(--text);
          background: var(--bg);
          min-height: 100vh;
          transition: background 0.2s ease, color 0.2s ease;
        }
        header { text-align: center; margin-bottom: 24px; position: relative; }
        header .right { position: absolute; right: 0; top: 0; }
        .toggle {
          background: var(--btn);
          color: var(--btn-text);
          border: 1px solid var(--border);
          padding: 8px 12px;
          border-radius: 10px;
          cursor: pointer;
        }
        h1 { font-size: 32px; margin: 0 0 6px; }
        h2 { font-size: 20px; margin: 0 0 12px; }
        h3 { font-size: 16px; margin: 12px 0 8px; }
        .muted { color: var(--muted); }
        .card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 18px;
          margin-bottom: 20px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .row { display: flex; gap: 10px; }
        input {
          flex: 1;
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 10px 12px;
          font-size: 14px;
          color: var(--text);
          background: var(--bg);
        }
        button {
          background: var(--btn);
          color: var(--btn-text);
          border: none;
          border-radius: 10px;
          padding: 10px 16px;
          font-weight: 600;
          cursor: pointer;
        }
        button:disabled { opacity: 0.6; cursor: default; }
        .answer { margin-top: 14px; }
        .answer-header {
          display: flex;
          align-items: center;
          gap: 10px;
          justify-content: space-between;
          flex-wrap: wrap;
        }
        .badge {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 700;
          border: 1px solid var(--border);
        }
        .badge.ok { background: #ecfdf5; color: #065f46; border-color: #10b981; }
        .badge.warn { background: #fff7ed; color: #9a3412; border-color: #f59e0b; }
        .risk { color: var(--text); opacity: 0.9; font-size: 13px; }
        .answer-text { margin: 10px 0 6px; line-height: 1.5; white-space: pre-wrap; }
        .sources { margin-top: 10px; }
        .src { border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin-top: 8px; }
        .src-title { font-weight: 600; margin-bottom: 6px; }
        .src-text { color: var(--src-text); font-size: 14px; }
        .chart { margin-top: 8px; }
        footer { text-align: center; margin-top: 12px; }
        code { background: var(--code); padding: 2px 6px; border-radius: 6px; }
      `}</style>
    </div>
  );
}
