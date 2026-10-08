import { useState, useMemo, useEffect } from "react";

/* ------------------------------------------------------------------
   Spashta — UPI Fraud Desk (explainable fraud review dashboard)
   Stack per PPT: React frontend -> Spring Boot (JWT) -> Python ML
   microservice (RF/XGBoost + SHAP/LIME) -> MySQL.
   Data below is SAMPLE data. Set VITE_API_URL to use your backend.
------------------------------------------------------------------- */

// Backend URL. Leave empty for demo mode. In a Vite project main.jsx sets window.SPASHTA_API_URL from .env
const API = (typeof window !== "undefined" && window.SPASHTA_API_URL) || "";
const BASE = 0.08; // model's average risk before looking at the transaction

const clamp = (n, a, b) => Math.min(b, Math.max(a, n));
const inr = (n) => "₹" + Number(n).toLocaleString("en-IN");
const scoreOf = (items) => clamp(BASE + items.reduce((s, i) => s + i.shap, 0), 0.01, 0.99);
const band = (s) => (s >= 0.8 ? ["High", "risk"] : s >= 0.5 ? ["Review", "amber"] : ["Low", "safe"]);
const limeOf = (c, i) => c.lime ?? +(c.shap * (0.88 + ((i * 7) % 5) * 0.04)).toFixed(2);

const C = (key, label, value, shap, why) => ({ key, label, value, shap, why });

const SEED = [
  {
    id: "UPI-7731", time: "08 Oct, 02:14", payer: "anita.s@okhdfc", payee: "rohit.k99@okaxis", amount: 48500,
    device: "New Android device", city: "Pune → Jaipur", status: "pending",
    items: [
      C("amt", "Amount vs 30-day average", "9.4× average", 0.31, "the amount is 9.4× the payer's 30-day average"),
      C("new", "Payee relationship", "First payment to this VPA", 0.18, "the payee is new to this payer"),
      C("dev", "Device", "Not seen before", 0.14, "the payment came from an unrecognised device"),
      C("hr", "Time of day", "02:14", 0.09, "it was made at 02:14 at night"),
      C("geo", "Distance from usual place", "640 km", 0.08, "it was 640 km from the payer's usual location"),
      C("vel", "Payments in last hour", "1", -0.01, "only one payment was made in the last hour"),
      C("age", "Account age", "4.2 years", -0.06, "the account is 4.2 years old"),
    ],
  },
  {
    id: "UPI-7728", time: "08 Oct, 23:48", payer: "mohan.d@oksbi", payee: "refund.support@ibl", amount: 9999,
    device: "New Android device", city: "Nashik", status: "pending",
    items: [
      C("amt", "Amount vs 30-day average", "5.5× average", 0.22, "the amount is 5.5× the payer's 30-day average"),
      C("new", "Payee relationship", "First payment to this VPA", 0.17, "the payee is new to this payer"),
      C("vel", "Payments in last hour", "4", 0.16, "4 payments were made within the last hour"),
      C("dev", "Device", "Not seen before", 0.12, "the payment came from an unrecognised device"),
      C("age", "Account age", "3 months", 0.09, "the account is only 3 months old"),
      C("hr", "Time of day", "23:48", 0.08, "it was made late at night"),
      C("geo", "Distance from usual place", "12 km", -0.02, "it was close to the payer's usual location"),
    ],
  },
  {
    id: "UPI-7719", time: "08 Oct, 11:20", payer: "kavita.r@okicici", payee: "kyc.update@axl", amount: 75000,
    device: "Known device", city: "Pune → Delhi", status: "pending",
    items: [
      C("amt", "Amount vs 30-day average", "12× average", 0.34, "the amount is 12× the payer's 30-day average"),
      C("new", "Payee relationship", "First payment to this VPA", 0.19, "the payee is new to this payer"),
      C("dev", "Device", "Not seen before", 0.15, "the payment came from an unrecognised device"),
      C("geo", "Distance from usual place", "1,120 km", 0.11, "it was 1,120 km from the payer's usual location"),
      C("age", "Account age", "1.1 years", 0.03, "the account is fairly new"),
      C("vel", "Payments in last hour", "1", 0.02, "one payment was made in the last hour"),
      C("hr", "Time of day", "11:20", -0.01, "it was made during normal hours"),
    ],
  },
  {
    id: "UPI-7702", time: "08 Oct, 19:05", payer: "sagar.p@okaxis", payee: "urbanfit@paytm", amount: 22000,
    device: "Known device", city: "Pimpri", status: "pending",
    items: [
      C("amt", "Amount vs 30-day average", "6.3× average", 0.3, "the amount is 6.3× the payer's 30-day average"),
      C("new", "Payee relationship", "First payment to this VPA", 0.16, "the payee is new to this payer"),
      C("vel", "Payments in last hour", "3", 0.14, "3 payments were made within the last hour"),
      C("dev", "Device", "Known device", -0.05, "the device is one the payer uses"),
      C("geo", "Distance from usual place", "3 km", -0.03, "it was close to the payer's usual location"),
      C("hr", "Time of day", "19:05", -0.02, "it was made during normal hours"),
      C("age", "Account age", "7 years", -0.07, "the account is 7 years old"),
    ],
  },
];

const IMPORTANCE = [
  ["Amount vs 30-day average", 0.31], ["Payee relationship", 0.22], ["Device", 0.17],
  ["Payments in last hour", 0.12], ["Distance from usual place", 0.09], ["Time of day", 0.06], ["Account age", 0.03],
];
const MODELS = [
  ["Logistic regression", "95.1%", "82.4%", "Built in", "Direct"],
  ["Random Forest", "97.8%", "91.2%", "SHAP, fast on trees", "Post-hoc"],
  ["XGBoost", "98.4%", "93.6%", "SHAP, fast on trees", "Post-hoc"],
];

/* Local stand-in for the ML microservice, used when the API is unreachable */
function localPredict(f) {
  const ratio = f.amount / 3500;
  const items = [
    C("amt", "Amount vs 30-day average", ratio.toFixed(1) + "× average", +clamp((ratio - 1) * 0.035, -0.05, 0.34).toFixed(2), `the amount is ${ratio.toFixed(1)}× a typical payment`),
    C("new", "Payee relationship", f.newPayee ? "First payment to this VPA" : "Known payee", f.newPayee ? 0.17 : -0.06, f.newPayee ? "the payee is new to this payer" : "the payee is already known"),
    C("dev", "Device", f.newDevice ? "Not seen before" : "Known device", f.newDevice ? 0.14 : -0.04, f.newDevice ? "the device is unrecognised" : "the device is known"),
    C("hr", "Time of day", String(f.hour).padStart(2, "0") + ":00", f.hour < 5 ? 0.09 : -0.02, f.hour < 5 ? "it was made at night" : "it was made during normal hours"),
    C("geo", "Distance from usual place", f.km + " km", f.km > 300 ? 0.1 : f.km > 50 ? 0.04 : -0.03, f.km > 50 ? `it was ${f.km} km from the usual location` : "it was close to the usual location"),
    C("vel", "Payments in last hour", String(f.velocity), +clamp((f.velocity - 1) * 0.05, -0.01, 0.16).toFixed(2), `${f.velocity} payment(s) in the last hour`),
  ];
  return { items, score: scoreOf(items) };
}

function explain(items) {
  const sorted = [...items].sort((a, b) => b.shap - a.shap);
  const up = sorted.filter((i) => i.shap > 0.05).slice(0, 2);
  const down = sorted[sorted.length - 1];
  if (!up.length) return "Nothing in this transaction stands out. " + cap(down.why) + ".";
  return `Flagged mainly because ${up.map((u) => u.why).join(" and ")}. Lowering the risk: ${down.why}.`;
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
.sp{--ink:#1b1740;--ink2:#2b2670;--paper:#f2f1f8;--card:#fff;--risk:#d92c5b;--safe:#0f8f84;--amber:#d98a0b;--line:#dcd9ee;--mute:#6a6790;
 font-family:'Instrument Sans',system-ui,sans-serif;color:var(--ink);background:var(--paper);min-height:100vh;display:grid;grid-template-columns:230px 1fr;font-size:15px;line-height:1.5}
.sp *{box-sizing:border-box}
.sp h1,.sp h2,.sp h3,.sp .num{font-family:'Bricolage Grotesque','Instrument Sans',sans-serif;letter-spacing:-.01em}
.sp button{font:inherit;cursor:pointer}
.sp :focus-visible{outline:3px solid #7b6cff;outline-offset:2px}
.rail{background:var(--ink);color:#e8e6ff;padding:26px 16px;display:flex;flex-direction:column;gap:6px;position:sticky;top:0;height:100vh}
.brand{font-family:'Bricolage Grotesque';font-size:26px;font-weight:700;padding:0 10px}
.brand-sub{padding:0 10px 22px;color:#a9a5de;font-size:13px}
.nav{background:none;border:0;color:#cfcdf5;text-align:left;padding:10px;border-radius:10px;display:flex;justify-content:space-between;align-items:center}
.nav:hover{background:#2a2575}
.nav[aria-current=page]{background:#fff;color:var(--ink);font-weight:600}
.pill{background:var(--risk);color:#fff;border-radius:99px;font-size:12px;padding:0 8px;font-weight:600}
.rail-note{margin-top:auto;font-size:12px;color:#a9a5de;padding:0 10px}
.main{padding:28px 34px 60px;min-width:0}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:20px}
.top h1{font-size:30px;margin:0}
.top p{margin:2px 0 0;color:var(--mute)}
.chip{border:1px solid var(--line);background:#fff;border-radius:99px;padding:4px 12px;font-size:13px;color:var(--mute)}
.stats{display:flex;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);border-radius:14px;margin-bottom:20px}
.stat{flex:1 1 150px;padding:14px 20px;border-right:1px solid var(--line)}
.stat:last-child{border:0}
.stat .num{font-size:28px;font-weight:700;display:block;line-height:1.1}
.stat span{color:var(--mute);font-size:13px}
.split{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}
.queue{display:flex;flex-direction:column;gap:8px}
.qi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;text-align:left;color:inherit;display:grid;grid-template-columns:1fr auto;gap:2px 10px}
.qi[aria-pressed=true]{border-color:var(--ink);box-shadow:inset 4px 0 0 var(--ink)}
.qi b{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.qi small{color:var(--mute)}
.qi .sc{grid-row:1/3;grid-column:2;text-align:right}
.sc .num{font-size:24px;font-weight:700;line-height:1}
.risk{color:var(--risk)}.amber{color:var(--amber)}.safe{color:var(--safe)}
.state{font-size:12px;font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 28px}
.head{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start}
.amt{font-size:44px;font-weight:700;line-height:1}
.verdict{text-align:right}
.verdict .num{font-size:44px;font-weight:700;line-height:1}
.gauge{position:relative;height:10px;border-radius:99px;background:linear-gradient(90deg,#d6f1ee,#fbe9c4 50%,#f7c3d1);margin:10px 0 4px;min-width:200px}
.gauge i{position:absolute;top:-4px;width:4px;height:18px;border-radius:2px;background:var(--ink)}
.gauge i.b{background:var(--mute);height:12px;top:-1px}
.gl{display:flex;justify-content:space-between;font-size:12px;color:var(--mute)}
.why{font-size:18px;max-width:62ch;margin:22px 0 6px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:16px 0 10px}
.seg button{background:#fff;border:0;padding:6px 16px;color:var(--mute)}
.seg button[aria-pressed=true]{background:var(--ink);color:#fff}
.ax{display:grid;grid-template-columns:200px 1fr 54px;gap:10px;font-size:12px;color:var(--mute);margin-bottom:4px}
.ax div{display:flex;justify-content:space-between}
.row{display:grid;grid-template-columns:200px 1fr 54px;gap:10px;align-items:center;padding:7px 0;border-top:1px solid #efedf8}
.row small{display:block;color:var(--mute);font-size:12px}
.track{position:relative;height:22px}
.track:before{content:"";position:absolute;left:50%;top:-6px;bottom:-6px;width:1px;background:var(--line)}
.bar{position:absolute;top:3px;height:16px;border-radius:4px;transform-origin:var(--o);animation:grow .45s ease-out both}
@keyframes grow{from{transform:scaleX(0)}}
.w{font-variant-numeric:tabular-nums;text-align:right;font-weight:600;font-size:14px}
.note{color:var(--mute);font-size:13px;margin:10px 0 0}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px 24px;margin:26px 0;padding:18px 0;border-block:1px solid var(--line)}
.facts dt{font-size:12px;color:var(--mute)}.facts dd{margin:0;font-weight:500}
textarea,input[type=text],input[type=number]{font:inherit;width:100%;border:1px solid var(--line);border-radius:10px;padding:9px 12px;background:#fff;color:var(--ink)}
label.f{display:block;font-size:13px;color:var(--mute);margin-bottom:12px}
.btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.btn{border-radius:10px;padding:10px 18px;border:1.5px solid var(--ink);background:#fff;color:var(--ink);font-weight:600}
.btn.p{background:var(--risk);border-color:var(--risk);color:#fff}
.btn.d{background:var(--ink);color:#fff}
.log{list-style:none;margin:18px 0 0;padding:0;font-size:13px;color:var(--mute)}
.log li{padding:5px 0;border-top:1px solid #efedf8}
.banner{border-radius:10px;padding:10px 14px;font-weight:600;margin-top:10px}
.banner.risk{background:#fde6ec}.banner.safe{background:#d9f2ef}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.panel h2{font-size:20px;margin:0 0 4px}
.panel p.s{margin:0 0 16px;color:var(--mute);font-size:14px}
.imp{display:grid;grid-template-columns:190px 1fr 44px;gap:10px;align-items:center;padding:5px 0;font-size:14px}
.imp div.b{height:12px;border-radius:3px;background:var(--ink2)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:var(--mute);font-weight:500;font-size:12px}
th,td{padding:9px 8px;border-bottom:1px solid #efedf8}
.stack{display:flex;height:26px;border-radius:6px;overflow:hidden;margin:6px 0 4px;background:#e7e5f5}
.stack i{display:block;height:100%}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0 18px}
.chk{display:flex;gap:8px;align-items:center;margin:8px 0 12px}
.phone{background:var(--ink);color:#fff;border-radius:32px;padding:26px 22px;box-shadow:0 18px 40px #1b174033}
.phone input[type=text],.phone input[type=number]{background:#2b2670;border-color:#403a8f;color:#fff}
.pn{font-family:'Bricolage Grotesque';font-size:22px;font-weight:700;margin-bottom:12px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 14px}
.chips button{background:#2b2670;color:#cfcdf5;border:1px solid #403a8f;border-radius:99px;padding:3px 11px;font-size:12px}
.chips button:hover{background:#403a8f}
.vd{border-radius:16px;padding:16px;margin:12px 0}
.vd.ok{background:#d9f2ef}.vd.warn{background:#fcefd2}.vd.stop{background:#fde6ec}
.phone .btn{border-color:#fff}
.btn:disabled{cursor:not-allowed}
@media (max-width:900px){.sp{grid-template-columns:1fr}.rail{position:static;height:auto;flex-direction:row;flex-wrap:wrap;padding:14px}.brand-sub,.rail-note{display:none}.brand{padding-right:16px}.main{padding:20px 16px}.split,.grid2{grid-template-columns:1fr}.ax,.row{grid-template-columns:120px 1fr 46px}.amt,.verdict .num{font-size:34px}}
@media (prefers-reduced-motion:reduce){.bar{animation:none}}
`;

function ExplainBars({ items, method }) {
  const w = (c, i) => (method === "lime" ? limeOf(c, i) : c.shap);
  const max = Math.max(0.35, ...items.map((c, i) => Math.abs(w(c, i))));
  const sorted = items.map((c, i) => ({ ...c, v: w(c, i) })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  return (
    <div key={method + items.map((i) => i.shap).join()}>
      <div className="ax"><span />
        <div><span>◀ lowers risk</span><span>raises risk ▶</span></div><span /></div>
      {sorted.map((c) => {
        const pct = (Math.abs(c.v) / max) * 50;
        const pos = c.v >= 0;
        return (
          <div className="row" key={c.key}>
            <div>{c.label}<small>{c.value}</small></div>
            <div className="track" role="img" aria-label={`${c.label}: ${pos ? "raises" : "lowers"} risk by ${Math.abs(c.v).toFixed(2)}`}>
              <div className="bar" style={{ [pos ? "left" : "right"]: "50%", width: pct + "%", background: pos ? "var(--risk)" : "var(--safe)", "--o": pos ? "left" : "right" }} />
            </div>
            <div className={"w " + (pos ? "risk" : "safe")}>{pos ? "+" : "−"}{Math.abs(c.v).toFixed(2)}</div>
          </div>
        );
      })}
    </div>
  );
}

function Gauge({ score }) {
  return (
    <div style={{ minWidth: 220 }}>
      <div className="gauge" aria-hidden="true">
        <i className="b" style={{ left: BASE * 100 + "%" }} />
        <i style={{ left: `calc(${score * 100}% - 2px)` }} />
      </div>
      <div className="gl"><span>0%</span><span>Typical risk {Math.round(BASE * 100)}%</span><span>100%</span></div>
    </div>
  );
}

function Case({ c, onAction }) {
  const [method, setMethod] = useState("shap");
  const [note, setNote] = useState("");
  const score = scoreOf(c.items);
  const [label, tone] = band(score);
  const act = (type) => { onAction(c.id, type, note); setNote(""); };
  return (
    <section className="panel" aria-label={"Case " + c.id}>
      <div className="head">
        <div>
          <div className="num amt">{inr(c.amount)}</div>
          <div style={{ marginTop: 6 }}>{c.payer} → <b>{c.payee}</b></div>
          <small style={{ color: "var(--mute)" }}>{c.id} · {c.time}</small>
        </div>
        <div className="verdict">
          <div className={"num " + tone}>{Math.round(score * 100)}%</div>
          <div className={"state " + tone}>{label} fraud risk</div>
          <Gauge score={score} />
        </div>
      </div>

      <p className="why">{explain(c.items)}</p>

      <div className="seg" role="group" aria-label="Explanation method">
        <button aria-pressed={method === "shap"} onClick={() => setMethod("shap")}>SHAP</button>
        <button aria-pressed={method === "lime"} onClick={() => setMethod("lime")}>LIME</button>
      </div>
      <ExplainBars items={c.items} method={method} />
      <p className="note">
        {method === "shap"
          ? "Each bar shows how far one factor pushed the risk score away from the typical 8%."
          : "LIME fits a small model around this one payment. Weights approximate SHAP and can differ slightly."}
      </p>

      <dl className="facts">
        <div><dt>Payer</dt><dd>{c.payer}</dd></div>
        <div><dt>Payee</dt><dd>{c.payee}</dd></div>
        <div><dt>Device</dt><dd>{c.device}</dd></div>
        <div><dt>Location</dt><dd>{c.city}</dd></div>
      </dl>

      {c.status === "pending" ? (
        <>
          <label className="f" htmlFor={"n" + c.id}>Reviewer note (saved to the audit trail)</label>
          <textarea id={"n" + c.id} rows={2} value={note} onChange={(e) => setNote(e.target.value)} placeholder="Why do you agree or disagree with the model?" />
          <div className="btns">
            <button className="btn p" onClick={() => act("confirmed")}>Confirm as fraud</button>
            <button className="btn" onClick={() => act("overridden")}>Mark as legitimate</button>
          </div>
        </>
      ) : (
        <div>
          <div className={"banner " + (c.status === "confirmed" ? "risk" : "safe")}>
            {c.status === "confirmed" ? "Confirmed as fraud. Payment stays blocked." : "Marked legitimate. Model decision overridden."}
          </div>
          <div className="btns"><button className="btn d" onClick={() => onAction(c.id, "pending", "")}>Reopen case</button></div>
        </div>
      )}
      <ul className="log" aria-label="Audit trail">
        {c.log.map((l, i) => <li key={i}><b>{l.who}</b> · {l.t} · {l.what}</li>)}
      </ul>
    </section>
  );
}

function Alerts({ cases, onAction }) {
  const [sel, setSel] = useState(cases[0].id);
  const cur = cases.find((c) => c.id === sel) || cases[0];
  const pending = cases.filter((c) => c.status === "pending").length;
  return (
    <>
      <div className="stats">
        <div className="stat"><span className="num">{pending}</span><span>waiting for review</span></div>
        <div className="stat"><span className="num">{cases.filter((c) => c.status === "confirmed").length}</span><span>confirmed fraud</span></div>
        <div className="stat"><span className="num">{cases.filter((c) => c.status === "overridden").length}</span><span>overridden by analysts</span></div>
        <div className="stat"><span className="num">{inr(cases.filter((c) => c.status === "pending").reduce((s, c) => s + c.amount, 0))}</span><span>held in queue</span></div>
      </div>
      <div className="split">
        <div className="queue" role="list">
          {cases.map((c) => {
            const s = scoreOf(c.items), [lb, tone] = band(s);
            return (
              <button key={c.id} className="qi" aria-pressed={c.id === sel} onClick={() => setSel(c.id)}>
                <b>{c.payee}</b>
                <span className="sc"><span className={"num " + tone}>{Math.round(s * 100)}</span><br /><small>{lb}</small></span>
                <small>{inr(c.amount)} · {c.time}{c.status !== "pending" ? " · " + (c.status === "confirmed" ? "fraud" : "cleared") : ""}</small>
              </button>
            );
          })}
        </div>
        <Case key={cur.id} c={cur} onAction={onAction} />
      </div>
    </>
  );
}

function Insights() {
  return (
    <>
      <div className="grid2">
        <section className="panel">
          <h2>What the model pays attention to</h2>
          <p className="s">Global SHAP importance across all scored payments.</p>
          {IMPORTANCE.map(([n, v]) => (
            <div className="imp" key={n}><span>{n}</span><div className="b" style={{ width: (v / 0.31) * 100 + "%" }} /><span className="w">{v.toFixed(2)}</span></div>
          ))}
        </section>
        <section className="panel">
          <h2>Fraud is rare, so we rebalance it</h2>
          <p className="s">SMOTE adds synthetic fraud examples to the training set only.</p>
          <small>Before SMOTE</small>
          <div className="stack"><i style={{ width: "0.8%", background: "var(--risk)" }} /></div>
          <small>0.8% fraud · 99.2% legitimate</small>
          <div style={{ height: 16 }} />
          <small>After SMOTE</small>
          <div className="stack"><i style={{ width: "50%", background: "var(--risk)" }} /></div>
          <small>50% fraud · 50% legitimate</small>
        </section>
      </div>
      <section className="panel" style={{ marginTop: 20 }}>
        <h2>Accuracy vs. explainability</h2>
        <p className="s">Sample numbers. Replace them with your own test-set results.</p>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead><tr><th>Model</th><th>Accuracy</th><th>Fraud recall</th><th>Explanation</th><th>Type</th></tr></thead>
            <tbody>{MODELS.map((m) => <tr key={m[0]}>{m.map((x, i) => <td key={i} style={i === 0 ? { fontWeight: 600 } : null}>{x}</td>)}</tr>)}</tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function Check() {
  const [f, setF] = useState({ amount: 48500, payee: "rohit.k99@okaxis", hour: 2, km: 640, velocity: 1, newPayee: true, newDevice: true });
  const [res, setRes] = useState(null);
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.type === "number" ? Number(e.target.value) : e.target.value });

  async function run() {
    setBusy(true);
    try {
      if (!API) throw new Error("no api");
      const r = await fetch(API + "/api/transactions/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + (localStorage.getItem("jwt") || "") },
        body: JSON.stringify(f),
      });
      if (!r.ok) throw new Error(r.status);
      const d = await r.json();
      setRes({ items: d.contributions, score: d.score }); setOffline(false);
    } catch {
      setRes(localPredict(f)); setOffline(true);
    }
    setBusy(false);
  }
  const [lb, tone] = res ? band(res.score) : [];
  return (
    <div className="split" style={{ gridTemplateColumns: "380px 1fr" }}>
      <section className="panel">
        <h2>Payment details</h2>
        <p className="s">Enter a payment to see how the model would score it.</p>
        <div className="form">
          <label className="f">Amount (₹)<input type="number" value={f.amount} onChange={set("amount")} /></label>
          <label className="f">Payee UPI ID<input type="text" value={f.payee} onChange={set("payee")} /></label>
          <label className="f">Hour (0–23)<input type="number" min="0" max="23" value={f.hour} onChange={set("hour")} /></label>
          <label className="f">Distance from usual place (km)<input type="number" value={f.km} onChange={set("km")} /></label>
          <label className="f">Payments in last hour<input type="number" min="0" value={f.velocity} onChange={set("velocity")} /></label>
        </div>
        <label className="chk"><input type="checkbox" checked={f.newPayee} onChange={set("newPayee")} /> First payment to this payee</label>
        <label className="chk"><input type="checkbox" checked={f.newDevice} onChange={set("newDevice")} /> Unrecognised device</label>
        <button className="btn d" onClick={run} disabled={busy}>{busy ? "Scoring…" : "Score this payment"}</button>
      </section>
      <section className="panel" aria-live="polite">
        {!res ? (
          <><h2>No result yet</h2><p className="s">Fill in the payment and select “Score this payment”. The explanation appears here.</p></>
        ) : (
          <>
            <div className="head">
              <div><div className={"num amt " + tone}>{Math.round(res.score * 100)}%</div><div className={"state " + tone}>{lb} fraud risk</div></div>
              <Gauge score={res.score} />
            </div>
            <p className="why">{explain(res.items)}</p>
            <ExplainBars items={res.items} method="shap" />
            {offline && <p className="note">Backend not reachable, so this result uses the built-in demo scoring. Set VITE_API_URL to use your model.</p>}
          </>
        )}
      </section>
    </div>
  );
}


/* ---------- Pay Safe: pre-payment live check ---------- */
const SAVED = ["mom@okhdfc", "chaiwala@ybl", "rent.owner@oksbi", "anita.s@okhdfc"];
const REPORTED = { "refund.support@ibl": 14, "kyc.update@axl": 9, "cashback.win@paytm": 22 };
const SUS = ["refund", "kyc", "support", "helpdesk", "cashback", "reward", "lottery", "verify", "prize"];
const SAMPLES = [
  ["Saved contact", "mom@okhdfc"], ["Lookalike", "m0m@okhdfc"], ["Reported ID", "refund.support@ibl"],
  ["Scam QR link", "upi://pay?pa=kyc.update@axl&pn=KYC%20Update&am=75000"],
];
function lev(a, b) {
  const m = [...Array(b.length + 1).keys()];
  for (let i = 1; i <= a.length; i++) {
    let p = m[0]; m[0] = i;
    for (let j = 1; j <= b.length; j++) { const t = m[j]; m[j] = Math.min(m[j] + 1, m[j - 1] + 1, p + (a[i - 1] === b[j - 1] ? 0 : 1)); p = t; }
  }
  return m[b.length];
}
function parsePay(raw) {
  let vpa = raw.trim(), amt = null;
  if (/^upi:\/\//i.test(vpa)) {
    try { const q = new URL(vpa.replace(/^upi:\/\//i, "https://x/")).searchParams; vpa = q.get("pa") || ""; amt = Number(q.get("am")) || null; } catch { vpa = ""; }
  }
  return { vpa: vpa.toLowerCase(), amt };
}
function localPre(vpa, amount, collect) {
  if (!/^[\w.\-]{2,}@[a-z]{2,}$/.test(vpa)) return null;
  const known = SAVED.includes(vpa);
  const look = !known && SAVED.find((x) => lev(x, vpa) <= 2);
  const reports = REPORTED[vpa] || 0;
  const sus = SUS.find((k) => vpa.includes(k));
  const ratio = amount / 3500, hr = new Date().getHours();
  const items = [
    C("rep", "Reports by other users", reports ? reports + " reports" : "None", reports ? +clamp(0.2 + reports * 0.01, 0.2, 0.4).toFixed(2) : -0.02, reports ? `${reports} other users reported this UPI ID` : "nobody has reported this UPI ID"),
    C("rel", "Your history with payee", known ? "Saved contact" : "Never paid before", known ? -0.12 : 0.14, known ? "you have paid this contact before" : "you have never paid this UPI ID"),
    C("col", "Request type", collect ? "Collect request" : "Normal payment", collect ? 0.22 : -0.02, collect ? "it pulls money out of your account, and you never need a PIN to receive money" : "it is a normal payment you are sending"),
    C("amt", "Amount vs your average", ratio.toFixed(1) + "× average", +clamp((ratio - 1) * 0.03, -0.04, 0.3).toFixed(2), `the amount is ${ratio.toFixed(1)}× your usual payment`),
    C("hr", "Time of day", String(hr).padStart(2, "0") + ":00", hr < 5 ? 0.06 : -0.02, hr < 5 ? "it is late at night" : "it is normal hours"),
  ];
  if (look) items.push(C("look", "Lookalike ID", "Resembles " + look, 0.25, `it looks almost like your saved contact ${look}`));
  if (sus) items.push(C("sus", "Suspicious words in ID", "Contains “" + sus + "”", 0.18, `the ID contains “${sus}”, a common scam word`));
  return { items, score: scoreOf(items) };
}

function PaySafe() {
  const [raw, setRaw] = useState("refund.support@ibl");
  const [amount, setAmount] = useState(9999);
  const [collect, setCollect] = useState(false);
  const [res, setRes] = useState(null);
  const [ms, setMs] = useState(null);
  const [wait, setWait] = useState(0);
  const [ack, setAck] = useState(false);
  const [done, setDone] = useState("");
  const p = parsePay(raw);
  const amt = p.amt || amount;
  const score = res ? res.score : 0;
  const level = score >= 0.7 ? "stop" : score >= 0.35 ? "warn" : "ok";
  const text = { ok: ["Looks safe", "safe"], warn: ["Be careful", "amber"], stop: ["Don't pay this", "risk"] }[level];

  useEffect(() => {
    setDone(""); setAck(false);
    const t = setTimeout(async () => {
      const t0 = performance.now();
      let r = null;
      if (API) {
        try {
          const x = await fetch(API + "/api/transactions/precheck", { method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + (localStorage.getItem("jwt") || "") }, body: JSON.stringify({ payee: p.vpa, amount: amt, collect }) });
          if (x.ok) { const d = await x.json(); r = { items: d.contributions, score: d.score }; }
        } catch { /* fall back to local scoring */ }
      }
      setRes(r || localPre(p.vpa, amt, collect));
      setMs(Math.round(performance.now() - t0));
    }, 350);
    return () => clearTimeout(t);
  }, [raw, amount, collect]);

  useEffect(() => { setWait(res && level === "stop" ? 10 : 0); }, [level, raw, !!res]);
  useEffect(() => { if (wait <= 0) return; const t = setTimeout(() => setWait((w) => w - 1), 1000); return () => clearTimeout(t); }, [wait]);

  const canPay = level === "ok" || (level === "warn" && ack) || (level === "stop" && wait === 0 && ack);
  return (
    <div className="split" style={{ gridTemplateColumns: "400px 1fr" }}>
      <section className="phone" aria-label="Payment screen">
        <div className="pn">Pay to</div>
        <label className="f" style={{ color: "#cfcdf5" }}>UPI ID, or paste a UPI QR link
          <input type="text" value={raw} onChange={(e) => setRaw(e.target.value)} placeholder="name@bank" />
        </label>
        <label className="f" style={{ color: "#cfcdf5" }}>Amount (₹){p.amt ? " — taken from QR" : ""}
          <input type="number" value={amt} disabled={!!p.amt} onChange={(e) => setAmount(Number(e.target.value))} />
        </label>
        <label className="chk" style={{ color: "#cfcdf5" }}><input type="checkbox" checked={collect} onChange={(e) => setCollect(e.target.checked)} /> This is a “collect / request money” notification</label>
        <div className="chips">{SAMPLES.map(([n, v]) => <button key={n} onClick={() => setRaw(v)}>{n}</button>)}</div>

        {!res ? <div className="vd" style={{ background: "#2b2670", color: "#cfcdf5" }} aria-live="polite">Enter a valid UPI ID to check it before you pay.</div> : (
          <div className={"vd " + level} aria-live="polite">
            <div className="head">
              <div><div className={"num state " + text[1]} style={{ fontSize: 22 }}>{text[0]}</div><div style={{ color: "#3b3868" }}>{Math.round(score * 100)}% fraud risk</div></div>
            </div>
            <p style={{ margin: "10px 0 0", color: "#1b1740" }}>{explain(res.items)}</p>
          </div>
        )}
        {res && level !== "ok" && (
          <label className="chk" style={{ color: "#cfcdf5" }}><input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} /> I have checked this with the person directly</label>
        )}
        {res && (
          <div className="btns">
            <button className="btn p" onClick={() => { setRaw(""); setRes(null); }}>Cancel</button>
            <button className="btn" disabled={!canPay} style={{ opacity: canPay ? 1 : 0.5 }} onClick={() => setDone("Demo only: this is where the UPI PIN screen would open.")}>
              {level === "stop" && wait > 0 ? `Wait ${wait}s` : level === "ok" ? "Continue" : "Pay anyway"}
            </button>
            {level !== "ok" && <button className="btn" onClick={() => setDone("Reported. Other users will now see a warning for this UPI ID.")}>Report ID</button>}
          </div>
        )}
        {done && <div className="banner safe" style={{ color: "#1b1740" }} role="status">{done}</div>}
      </section>

      <section className="panel">
        <div className="head">
          <div><h2>Why this verdict</h2><p className="s">Updates as you type. The same SHAP-style reasons, shown to the payer before money moves.</p></div>
          {ms !== null && <span className="chip">Checked in {ms} ms{API ? "" : " · demo mode"}</span>}
        </div>
        {res ? <ExplainBars items={res.items} method="shap" /> : <p className="note">No result yet.</p>}
        <div className="facts" style={{ display: "block" }}>
          <b>How this reaches real payments</b>
          <p className="note">PhonePe and Google Pay are closed apps, so a separate app cannot add a popup inside them. Two real routes exist: (1) the payer opens Spashta, or scans the QR in it, before paying; (2) a bank or PSP calls this same <code>/precheck</code> API during authorisation, in under 200 ms, and shows the warning inside its own app.</p>
        </div>
      </section>
    </div>
  );
}

const TITLES = {
  pay: ["Pay Safe", "Check a UPI ID or QR before you send money."],
  alerts: ["Flagged payments", "Review each alert with the reasons the model gave."],
  insights: ["Model insights", "How the models behave overall, and what they trade away."],
  check: ["Check a payment", "Try a transaction and read the explanation."],
};

export default function App() {
  const [view, setView] = useState("pay");
  const [cases, setCases] = useState(() =>
    SEED.map((c) => ({ ...c, log: [{ who: "Model", t: c.time, what: `Flagged at ${Math.round(scoreOf(c.items) * 100)}% risk` }] }))
  );
  const pending = useMemo(() => cases.filter((c) => c.status === "pending").length, [cases]);

  const onAction = (id, status, note) =>
    setCases((cs) => cs.map((c) => c.id !== id ? c : {
      ...c, status,
      log: [...c.log, { who: "You", t: new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }),
        what: (status === "pending" ? "Reopened case" : status === "confirmed" ? "Confirmed fraud" : "Marked legitimate") + (note ? ` — “${note}”` : "") }],
    }));

  const [h, sub] = TITLES[view];
  const nav = (k, t, extra) => (
    <button className="nav" aria-current={view === k ? "page" : undefined} onClick={() => setView(k)}>{t}{extra}</button>
  );
  return (
    <div className="sp">
      <style>{CSS}</style>
      <nav className="rail" aria-label="Main">
        <div className="brand">Spashta</div>
        <div className="brand-sub">UPI fraud desk</div>
        {nav("pay", "Pay Safe · live")}
        {nav("alerts", "Alerts", pending ? <span className="pill">{pending}</span> : null)}
        {nav("insights", "Model insights")}
        {nav("check", "Check a payment")}
        <div className="rail-note">Sample data. Explanations from Random Forest / XGBoost with SHAP and LIME.</div>
      </nav>
      <main className="main">
        <header className="top">
          <div><h1>{h}</h1><p>{sub}</p></div>
          <span className="chip">Model: XGBoost · sample data</span>
        </header>
        {view === "pay" && <PaySafe />}
        {view === "alerts" && <Alerts cases={cases} onAction={onAction} />}
        {view === "insights" && <Insights />}
        {view === "check" && <Check />}
      </main>
    </div>
  );
}
