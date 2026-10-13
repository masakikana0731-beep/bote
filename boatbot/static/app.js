const $ = (s, el = document) => el.querySelector(s);
const yen = (n) => (n || 0).toLocaleString() + "円";
let state = null;

function token() { try { return localStorage.getItem("boatbot_token") || ""; } catch { return ""; } }

async function api(path, body) {
  const opt = { headers: { "x-token": token() } };
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["content-type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  if (r.status === 401) {
    const t = prompt("アクセス用トークンを入力してください");
    if (t) { try { localStorage.setItem("boatbot_token", t); } catch {} return api(path, body); }
  }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

const boats = (combo) => combo.split("-").map((n) => `<span class="n n${n}">${n}</span>`).join("");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- 今日
function renderBudget() {
  const s = state.settings;
  const pct = s.daily_budget ? Math.min(100, (state.spent / s.daily_budget) * 100) : 100;
  $("#budget").innerHTML = `今日の投資 ${yen(state.spent)} / ${yen(s.daily_budget)}（残り ${yen(state.remaining)}・${state.races}/${s.max_races_per_day}R）
    <div class="bar"><div style="width:${pct}%"></div></div>`;
  const mode = { paper: "🧪 仮想 自動投資中", real: "🤖 本番 自動投資中" }[s.auto_mode];
  $("#automode").classList.toggle("hidden", !mode);
  $("#automode").innerHTML = mode ? `<span>${mode}</span><button data-stop="1">緊急停止</button>` : "";
}

function renderBuyLog() {
  $("#buylog").innerHTML = state.buy_log.map((l) => `<div class="row"><span class="hint">${l.at.slice(11, 16)} ${l.mode === "real" ? "本番" : "仮想"} ${yen(l.total)}</span>
    <span>${l.ok ? "✅" : l.uncertain ? "❓" : "❌"} ${esc(l.message)}</span></div>`).join("") || `<p class="hint">まだありません</p>`;
}

function recCard(r) {
  const label = { pending: "未対応", bought: "購入済", skipped: "見送った", pass: "ツールが見送り",
    paper: "仮想購入", failed: "自動購入失敗", unknown: "購入不明・要確認" }[r.status];
  const hit = r.returned > 0;
  let body = "";
  if (r.status === "pass") {
    body = `<div class="hint">${esc(r.info.skip_reason)}</div>`;
  } else {
    body = `<table class="t"><tr><th>3連単</th><th>オッズ</th><th>推定確率</th><th>金額</th></tr>` +
      r.tickets.map((t) => `<tr><td>${boats(t.combo)}</td><td>${t.odds}</td><td>${(t.prob * 100).toFixed(1)}%</td><td>${yen(t.amount)}</td></tr>`).join("") +
      `<tr><th>合計</th><th></th><th>${(r.info.hit_prob * 100).toFixed(0)}%</th><th>${yen(r.total)}</th></tr></table>`;
  }
  let result = "";
  if (r.result_combo && r.result_combo !== "-") {
    result = `<div>結果 ${boats(r.result_combo)} 払戻 ${yen(r.payout)}/100円 ${hit ? `<span class="tag hit">的中 +${yen(r.returned)}</span>` : ""}</div>`;
  }
  let actions = "";
  if (r.status === "unknown") {  // 履歴を見て本人が確定させる
    actions = `<div class="actions">
      <a class="btn primary" href="${esc(state.teleboat_url)}" target="_blank" rel="noopener">履歴を確認</a>
      <button class="good" data-status="bought" data-id="${r.id}">買えていた</button>
      <button data-status="skipped" data-id="${r.id}">買えていなかった</button></div>`;
  } else if (["pending", "skipped", "bought", "failed"].includes(r.status)) {
    actions = `<div class="actions">
      <button data-copy="${r.id}">📋 買い目コピー</button>
      <a class="btn primary" href="${esc(state.teleboat_url)}" target="_blank" rel="noopener">テレボートを開く</a>
      ${r.status !== "bought" ? `<button class="good" data-status="bought" data-id="${r.id}">購入した</button>` : ""}
      ${r.status === "pending" ? `<button data-status="skipped" data-id="${r.id}">見送る</button>` : ""}
      ${r.status === "bought" ? `<button data-status="pending" data-id="${r.id}">取消</button>` : ""}
    </div>`;
  }
  return `<div class="card">
    <div class="head"><span class="title">${esc(r.venue)} ${r.rno}R</span>
      <span class="hint">締切 ${r.deadline}</span><span class="tag ${r.status}">${label}</span></div>
    ${body}${result}${actions}</div>`;
}

function renderToday() {
  $("#error").classList.toggle("hidden", !state.last_error);
  $("#error").textContent = state.last_error ? "エラー: " + state.last_error : "";
  $("#recs").innerHTML = state.recs.length ? state.recs.map(recCard).join("") : `<p class="hint">まだありません</p>`;
  $("#upcoming").innerHTML = state.upcoming.map((u) => `<div class="row">
      <span>${esc(u.venue)} ${u.rno}R <span class="hint">締切 ${u.deadline}</span></span>
      <button data-predict="${u.jcd}:${u.rno}">今すぐ予想</button></div>`).join("") || `<p class="hint">本日の対象レースは終了 / 未取得</p>`;
  renderBuyLog();
}

document.addEventListener("click", async (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  try {
    if (t.dataset.stop) {
      if (!confirm("自動投資を停止しますか？")) return;
      await api("/api/stop", {});
      await load();
      fillSettings();
    } else if (t.dataset.status) {
      await api(`/api/recs/${t.dataset.id}/status`, { status: t.dataset.status });
      await load();
    } else if (t.dataset.copy) {
      const r = state.recs.find((x) => x.id == t.dataset.copy);
      const text = `${r.venue}${r.rno}R 3連単\n` + r.tickets.map((x) => `${x.combo} ${x.amount}円`).join("\n");
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {  // http（自宅LAN）ではclipboard APIが使えないので旧方式
        const ta = Object.assign(document.createElement("textarea"), { value: text });
        document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
      }
      t.textContent = "✅ コピーしました";
    } else if (t.dataset.predict) {
      const [jcd, rno] = t.dataset.predict.split(":");
      t.disabled = true; t.textContent = "予想中…";
      await api("/api/predict", { jcd, rno: Number(rno) });
      await load();
    } else if (t.dataset.tab) {
      document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b === t));
      document.querySelectorAll("main section").forEach((s) => s.classList.toggle("hidden", s.id !== "tab-" + t.dataset.tab));
      if (t.dataset.tab === "stats") loadStats();
    }
  } catch (err) {
    alert(err.message);
    await load();
  }
});

// ---------- 設定
function buildSliders() {
  document.querySelectorAll(".slider").forEach((el) => {
    const name = el.dataset.name;
    el.innerHTML = `${el.textContent}<span class="val"></span>
      <input type="range" min="0" max="100" step="5" name="${name}">
      <div class="ends"><span>${el.dataset.left}</span><span>${el.dataset.right}</span></div>`;
    const input = $("input", el);
    input.addEventListener("input", () => ($(".val", el).textContent = input.value));
  });
  $("#venues").innerHTML = Object.entries(state.venues).sort(([a], [b]) => a.localeCompare(b))
    .map(([c, n]) => `<label><input type="checkbox" value="${c}">${n}</label>`).join("");
}

function fillSettings() {
  const s = state.settings, f = $("#settings-form");
  for (const [k, v] of Object.entries(s)) {
    const el = f.elements[k];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    el.dispatchEvent(new Event("input"));
  }
  $("#real-note").textContent = (state.allow_real
    ? "本番モードはPC側で許可されています。"
    : "本番モードを使うにはPC側で BOATBOT_ALLOW_REAL=1 と teleboat_flow.py の準備が必要です（README参照）。")
    + (state.hard_cap != null ? ` 最終上限：1日${yen(state.hard_cap)}` : "");
  const sel = new Set(s.venues.split(",").filter(Boolean));
  document.querySelectorAll("#venues input").forEach((c) => (c.checked = sel.has(c.value)));
}

$("#settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target, data = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    data[el.name] = el.type === "checkbox" ? el.checked : el.type === "number" || el.type === "range" ? Number(el.value) : el.value;
  }
  data.venues = [...document.querySelectorAll("#venues input:checked")].map((c) => c.value).join(",");
  if (data.auto_mode === "real" && state.settings.auto_mode !== "real" &&
      !confirm(`本番の自動購入をONにします。\n1日 最大${yen(data.daily_budget)}・1レース最大${yen(data.race_budget)}・最大${data.max_races_per_day}R\n\nテレボートの約定ではツールによる投票が禁止されており、利用停止になる可能性があります。よろしいですか？`)) return;
  await api("/api/settings", data);
  $("#saved").textContent = "保存しました ✔";
  setTimeout(() => ($("#saved").textContent = ""), 2000);
  await load();
});

// ---------- 成績
async function loadStats() {
  const rows = await api("/api/stats");
  const pl = (bet, ret) => { const d = ret - bet; return `<span class="${d >= 0 ? "plus" : "minus"}">${d >= 0 ? "+" : ""}${yen(d)}</span>`; };
  const sum = rows.reduce((a, r) => { for (const k in r) if (k !== "hd") a[k] = (a[k] || 0) + (r[k] || 0); return a; }, {});
  $("#stats").innerHTML = rows.length ? `<table class="t">
    <tr><th>日付</th><th>実際</th><th>仮想</th><th>的中</th></tr>
    <tr><th>通算</th><th>${pl(sum.real_bet, sum.real_ret)}</th><th>${pl(sum.virt_bet, sum.virt_ret)}</th><th>${sum.hits}/${sum.settled}</th></tr>
    ${rows.map((r) => `<tr><td>${r.hd.slice(4, 6)}/${r.hd.slice(6)}</td><td>${pl(r.real_bet, r.real_ret)}</td><td>${pl(r.virt_bet, r.virt_ret)}</td><td>${r.hits}/${r.settled}</td></tr>`).join("")}
  </table>` : `<p class="hint">まだデータがありません</p>`;
}

async function load() {
  state = await api("/api/state");
  renderBudget();
  renderToday();
}

(async () => {
  await load();
  buildSliders();
  fillSettings();
  setInterval(() => { if (!document.hidden) load(); }, 30000);
})();
