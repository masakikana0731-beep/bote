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

// 方針2：艇ごとにモデルと市場（オッズ）の見立てを並べる
function insightTable(r) {
  const ins = r.info && r.info.insight;
  if (!ins || !ins.boats) return "";
  const p = (x) => (x == null ? "-" : (x * 100).toFixed(x >= 0.99 && x < 1 ? 1 : 0) + "%");
  const rows = ins.boats.map((b) => {
    const d = b.diff_win == null ? "" : `${b.diff_win >= 0 ? "+" : ""}${(b.diff_win * 100).toFixed(0)}`;
    return `<tr${b.notable ? ' class="notable"' : ""}><td>${boats(String(b.waku))}${b.notable ? " ⭐" : ""}</td>
      <td>${p(b.model_win)}</td><td>${p(b.market_win)}</td><td class="${b.diff_win >= 0 ? "plus" : "minus"}">${d}</td>
      <td>${p(b.model_top2)}</td></tr>`;
  }).join("");
  const solid = (ins.solid || []).map((x) => `<span class="tag solid">ほぼ確実な本命 ${x.waku}号艇（${x.type} ${p(x.prob)}）</span>`).join(" ");
  return `<details class="insight" data-rid="${r.id}"><summary>モデルと市場の見立て${solid ? " 🎯" : ""}</summary>
    ${solid ? `<div>${solid}</div><div class="hint">過去2年、この条件で買うと回収率100〜104%（ほぼ元返し）。負けにくいだけで、儲けは小さい</div>` : ""}
    <table class="t"><tr><th>艇</th><th>モデル1着</th><th>市場1着</th><th>差</th><th>モデル2着内</th></tr>${rows}</table>
    <div class="hint">⭐ = モデルが市場より5ポイント以上高く見ている艇。市場（オッズ）のほうが平均的には正確なので、参考程度に。
    締切前のオッズは確定までに動く（単勝の大穴は確定時に半分以下になることが多い）</div></details>`;
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
    ${body}${insightTable(r)}${result}${actions}</div>`;
}

function renderToday() {
  $("#error").classList.toggle("hidden", !state.last_error);
  $("#error").textContent = state.last_error ? "エラー: " + state.last_error : "";
  // 30秒ごとの再描画で、開いている「モデルと市場の見立て」が閉じないようにする
  const open = new Set([...document.querySelectorAll("details.insight[open]")].map((d) => d.dataset.rid));
  $("#recs").innerHTML = state.recs.length ? state.recs.map(recCard).join("") : `<p class="hint">まだありません</p>`;
  document.querySelectorAll("details.insight").forEach((d) => { if (open.has(d.dataset.rid)) d.open = true; });
  $("#upcoming").innerHTML = state.upcoming.map((u) => `<div class="row">
      <span>${esc(u.venue)} ${u.rno}R <span class="hint">締切 ${u.deadline}</span></span>
      <button data-predict="${u.jcd}:${u.rno}">今すぐ予想</button></div>`).join("") || `<p class="hint">本日の対象レースは終了 / 未取得</p>`;
  renderBuyLog();
}

document.addEventListener("click", async (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  try {
    if (t.dataset.preset) {
      applyPreset(t.dataset.preset);
    } else if (t.dataset.stop) {
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
// optimize.py の結果（2026/7/15〜10/12・3,106レース、1日3,000円・最大10R）。お金の設定は変えない
const PRESETS = {
  fun: { gamble: 30, odds_width: 30, tickets: 6, selectivity: 0, skip_fav: 0, crowd: 40, whim: 0, allocation: "prob", max_wind: 4,
    note: "的中率 約25%・回収率 約86%（シミュレーション）。よく当たるがじわじわ減る" },
  balance: { gamble: 10, odds_width: 55, tickets: 5, selectivity: 60, skip_fav: 0, crowd: 0, whim: 0, allocation: "equal", max_wind: 0,
    note: "的中率 約19%・回収率 約99%（90%区間 75〜125%）。収支トントン前後" },
  jackpot: { gamble: 100, odds_width: 55, tickets: 6, selectivity: 40, skip_fav: 80, crowd: 0, whim: 0, allocation: "equal", max_wind: 4,
    note: "的中率 約6%・回収率 約137%（90%区間 82〜200%）。40連敗以上もある。過去の偶然の可能性あり" },
};

function applyPreset(key) {
  const p = PRESETS[key], f = $("#settings-form");
  for (const [k, v] of Object.entries(p)) {
    const el = f.elements[k];
    if (!el) continue;
    el.value = v;
    el.dispatchEvent(new Event("input"));
  }
  $("#preset-note").textContent = p.note + "（保存を押すと反映）";
}

function buildSliders() {
  document.querySelectorAll(".slider").forEach((el) => {
    const name = el.dataset.name;
    el.innerHTML = `${el.textContent}<span class="val"></span>
      <input type="range" min="0" max="100" step="5" name="${name}">
      <div class="ends"><span>${el.dataset.left}</span><span>${el.dataset.right}</span></div>`;
    const input = $("input", el);
    // model.target_odds と同じ式：0=8倍 〜 100=300倍（対数で等間隔）
    const v = () => Number(input.value);
    const label = () => ({
      gamble: () => `${v()}（狙う配当 約${Math.round(8 * Math.pow(300 / 8, v() / 100))}倍）`,
      odds_width: () => `${v()}（目安の約${Math.exp(0.4 + 1.1 * v() / 100).toFixed(1)}倍の範囲）`,  // model.odds_sigma
      selectivity: () => v() ? `${v()}（期待値${(1.5 * v() / 100).toFixed(2)}以上）` : "0（制限なし）",
      skip_fav: () => v() ? `${v()}（1着確率${Math.round(100 - 50 * v() / 100)}%超で見送り）` : "0（見送らない）",  // model.skip_fav_threshold
    }[name] || (() => input.value))();
    input.addEventListener("input", () => ($(".val", el).textContent = label()));
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
  loadPaper(pl);
}

async function loadPaper(pl) {
  const p = await api("/api/papertrade");
  if (!p.scored_days) { $("#paper").innerHTML = `<p class="hint">まだ採点した日がありません</p>`; return; }
  const roi = (x) => (x == null ? "-" : `<span class="${x >= 1 ? "plus" : "minus"}">${(x * 100).toFixed(0)}%</span>`);
  $("#paper").innerHTML = `<p class="hint">採点した日 ${p.scored_days}日（最新 ${p.last_day.slice(4, 6)}/${p.last_day.slice(6)}）</p>` +
    p.strategies.map((s) => `<div class="card">
      <div class="head"><span class="title">${esc(s.name)}</span><span>${roi(s.roi)}</span></div>
      <div class="hint">${esc(s.desc)}</div>
      <div>通算 ${s.races}レース・的中${s.hits} 投資${yen(s.bet)} 払戻${yen(s.ret)} ${pl(s.bet, s.ret)}</div>
      <details><summary class="hint">日別</summary><table class="t"><tr><th>日付</th><th>レース</th><th>的中</th><th>収支</th></tr>
      ${s.daily.map((d) => `<tr><td>${d.day.slice(4, 6)}/${d.day.slice(6)}</td><td>${d.races}</td><td>${d.hits}</td><td>${pl(d.bet, d.ret)}</td></tr>`).join("")}
      </table></details></div>`).join("") +
    `<p class="hint">的中が数回しかない買い方は、たまたまの大当たりで大きく上下する。数か月分たまるまでは参考程度に</p>`;
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
