// Vanilla-JS dashboard — no build step. Reads snapshots/index.json and per-day summary/deltas.
const fmt = new Intl.NumberFormat("en-US");

async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

function barChart(containerId, obj, { topN = 10, total = null } = {}) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  const entries = Object.entries(obj || {})
    .filter(([k]) => k !== "unknown" && k !== "unspecified")
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN);
  const max = entries.reduce((m, [, v]) => Math.max(m, v), 0) || 1;
  for (const [k, v] of entries) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("div");
    label.textContent = k;
    label.title = k;
    label.style.overflow = "hidden";
    label.style.textOverflow = "ellipsis";
    label.style.whiteSpace = "nowrap";
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(v / max) * 100}%`;
    track.appendChild(fill);
    const num = document.createElement("div");
    num.className = "num-cell";
    num.textContent = total ? `${((v / total) * 100).toFixed(1)}%` : fmt.format(v);
    row.append(label, track, num);
    container.appendChild(row);
  }
  if (entries.length === 0) {
    container.innerHTML = `<p class="caption">No data.</p>`;
  }
}

function lineChart(containerId, series) {
  // series: [{ key, color, points: [{date, value}] }]
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  const w = container.clientWidth || 800, h = 220;
  const m = { top: 12, right: 12, bottom: 28, left: 48 };
  const allPts = series.flatMap(s => s.points);
  if (allPts.length === 0) {
    container.innerHTML = `<p class="caption">No snapshots yet.</p>`;
    return;
  }
  const dates = [...new Set(allPts.map(p => p.date))].sort();
  const maxV = Math.max(...allPts.map(p => p.value || 0)) || 1;
  const x = i => m.left + (i / Math.max(dates.length - 1, 1)) * (w - m.left - m.right);
  const y = v => m.top + (1 - v / maxV) * (h - m.top - m.bottom);

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  // axes
  const axisG = document.createElementNS(ns, "g");
  axisG.setAttribute("class", "axis");
  // x labels
  dates.forEach((d, i) => {
    if (dates.length <= 8 || i % Math.ceil(dates.length / 8) === 0 || i === dates.length - 1) {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", x(i));
      t.setAttribute("y", h - 8);
      t.setAttribute("text-anchor", "middle");
      t.textContent = d.slice(5);
      axisG.appendChild(t);
    }
  });
  // y ticks (3)
  for (let k = 0; k <= 3; k++) {
    const v = (maxV * k) / 3;
    const yp = y(v);
    const ln = document.createElementNS(ns, "line");
    ln.setAttribute("x1", m.left); ln.setAttribute("x2", w - m.right);
    ln.setAttribute("y1", yp); ln.setAttribute("y2", yp);
    axisG.appendChild(ln);
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", m.left - 6); t.setAttribute("y", yp + 4);
    t.setAttribute("text-anchor", "end");
    t.textContent = fmt.format(Math.round(v));
    axisG.appendChild(t);
  }
  svg.appendChild(axisG);

  for (const s of series) {
    const path = document.createElementNS(ns, "path");
    const idx = new Map(dates.map((d, i) => [d, i]));
    const pts = s.points.filter(p => p.value != null).map(p => `${x(idx.get(p.date))},${y(p.value)}`);
    if (pts.length === 0) continue;
    path.setAttribute("d", `M ${pts.join(" L ")}`);
    path.setAttribute("class", s.cssClass);
    svg.appendChild(path);
  }
  container.appendChild(svg);

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = series.map(s =>
    `<span><span class="dot" style="background:${s.color}"></span>${s.key}</span>`,
  ).join(" &nbsp; ");
  container.appendChild(legend);
}

function row(text) { const li = document.createElement("li"); li.textContent = text; return li; }

async function main() {
  let index;
  try {
    index = await fetchJSON("snapshots/index.json");
  } catch {
    setText("latest-meta", "No snapshots published yet.");
    return;
  }
  const snaps = index.snapshots || [];
  if (snaps.length === 0) {
    setText("latest-meta", "No snapshots published yet.");
    return;
  }
  const latest = snaps[snaps.length - 1];
  const summary = await fetchJSON(latest.summary_path);
  setText("latest-meta",
    `Latest snapshot: ${summary.date} — ${snaps.length} weekly snapshot(s) on file.`);
  setText("kpi-total", fmt.format(summary.total_repos || 0));
  setText("kpi-quant", fmt.format(summary.quantized_repos || 0));
  setText("kpi-gguf", fmt.format(summary.gguf_repos || 0));
  setText("kpi-dl", fmt.format(summary.downloads_30d_sum || 0));

  // Time series
  const points = await Promise.all(snaps.map(async s => {
    try {
      const j = await fetchJSON(s.summary_path);
      return { date: j.date, total: j.total_repos, gguf: j.gguf_repos };
    } catch { return null; }
  }));
  const valid = points.filter(Boolean);
  lineChart("chart-growth", [
    { key: "total", color: "#ff8a5b", cssClass: "line-total",
      points: valid.map(p => ({ date: p.date, value: p.total })) },
    { key: "gguf", color: "#5cc8ff", cssClass: "line-gguf",
      points: valid.map(p => ({ date: p.date, value: p.gguf })) },
  ]);

  // Distributions
  barChart("chart-lab", summary.by_originating_lab, { topN: 12, total: summary.total_repos });
  barChart("chart-country", summary.by_uploader_country, { topN: 8, total: summary.total_repos });
  barChart("chart-compute", summary.by_compute_class, { topN: 8, total: summary.total_repos });
  barChart("chart-quant", summary.by_quant_format, { topN: 10, total: summary.total_repos });

  // Top movers from latest deltas (if present), else fall back to summary.top_30d
  let deltas = null;
  try {
    deltas = await fetchJSON(`snapshots/${summary.date}/deltas.json`);
  } catch {}
  const moversBody = document.querySelector("#movers tbody");
  if (deltas && deltas.download_surges && deltas.download_surges.length) {
    setText("movers-meta", `Largest 30-day download deltas vs ${deltas.old_snapshot}.`);
    for (const m of deltas.download_surges.slice(0, 20)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><a href="https://huggingface.co/${m.repo_id}" target="_blank" rel="noopener">${m.repo_id}</a></td>
        <td>—</td><td>—</td><td>—</td>
        <td>+${fmt.format(m.delta)}</td>`;
      moversBody.appendChild(tr);
    }
  } else if (summary.top_30d) {
    setText("movers-meta", `Top 30-day downloads in latest snapshot.`);
    for (const m of summary.top_30d.slice(0, 20)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><a href="https://huggingface.co/${m.repo_id}" target="_blank" rel="noopener">${m.repo_id}</a></td>
        <td>${m.family || "—"}</td>
        <td>${m.originating_lab || "—"}</td>
        <td>${m.quant_format || "—"}</td>
        <td>${fmt.format(m.downloads_30d || 0)}</td>`;
      moversBody.appendChild(tr);
    }
  }

  // Events
  const removedUl = document.getElementById("removed");
  const flipsUl = document.getElementById("license-flips");
  if (deltas) {
    setText("events-meta", `Changes since ${deltas.old_snapshot}: +${deltas.n_added} new, −${deltas.n_removed} removed.`);
    for (const r of (deltas.removed || []).slice(0, 15)) {
      const li = document.createElement("li");
      li.innerHTML = `<code>${r.repo_id}</code> ${r.originating_lab ? `(${r.originating_lab})` : ""}`;
      removedUl.appendChild(li);
    }
    for (const f of (deltas.license_flips || []).slice(0, 15)) {
      const li = document.createElement("li");
      li.innerHTML = `<code>${f.repo_id}</code>: ${f.old || "∅"} → ${f.new || "∅"}`;
      flipsUl.appendChild(li);
    }
    if ((deltas.removed || []).length === 0) removedUl.appendChild(row("No removals."));
    if ((deltas.license_flips || []).length === 0) flipsUl.appendChild(row("No license changes."));
  } else {
    setText("events-meta", "Delta data appears once a second snapshot exists.");
  }
}

main().catch(err => {
  console.error(err);
  setText("latest-meta", `Error loading data: ${err.message}`);
});
