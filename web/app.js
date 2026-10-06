/* Stats Nuke: the website for the StatsNuke forecasting system.
 *
 * One page app, no build step. Data comes from one JSON snapshot that the Live workflow
 * exports every three hours (`fplh web export`). The backend is Supabase: email + password
 * accounts, a `members` table holding each person's access (admin, friend, pending,
 * declined) and a `snapshots` table that only approved members can read (row-level
 * security, see web/supabase/schema.sql). Without Supabase config and with an embedded
 * snapshot, the app runs as a preview with the access flow simulated in memory.
 */
(() => {
  "use strict";

  const CFG = window.STATSNUKE_CONFIG || {};
  const EMBEDDED = window.STATSNUKE_SNAPSHOT || null;
  const REPO = CFG.repo || "VIKIII2269/StatsNuke";

  // ---------- tiny DOM helpers (strings always go in as text, never HTML) ----------
  const $ = (s, el = document) => el.querySelector(s);
  function h(tag, attrs, ...kids) {
    const svgTags = ["svg", "path", "circle", "line", "polyline", "rect", "g", "text", "polygon"];
    const el = svgTags.includes(tag)
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "class") el.setAttribute("class", v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const icon = (name, cls) => {
    const span = h("span", { class: cls || "ico", "aria-hidden": "true" });
    span.innerHTML = ICONS[name] || ""; // static, trusted markup
    return span.firstChild ? span.firstChild : span;
  };
  const md = (text) => {
    // **bold** and `code` only; everything else stays text
    const out = h("span", { class: "md" });
    String(text || "")
      .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
      .forEach((part) => {
        if (part.startsWith("**") && part.endsWith("**")) out.append(h("b", {}, part.slice(2, -2)));
        else if (part.startsWith("`") && part.endsWith("`")) out.append(h("code", {}, part.slice(1, -1)));
        else if (part) out.append(document.createTextNode(part));
      });
    return out;
  };

  // ---------- formatting ----------
  const fmt = {
    n: (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d)),
    pct: (x, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`),
    spct: (x, d = 1) => (x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)}%`),
    signed: (x, d = 0) => (x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`),
    money: (x) => (x == null ? "–" : `£${Number(x).toFixed(1)}m`),
    date: (iso) =>
      iso
        ? new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })
        : "–",
    day: (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" }) : "–"),
    ago: (iso) => {
      if (!iso) return "never";
      const s = (Date.now() - new Date(iso).getTime()) / 1000;
      if (s < 60) return "just now";
      if (s < 3600) return `${Math.round(s / 60)} min ago`;
      if (s < 86400 * 2) return `${Math.round(s / 3600)} h ago`;
      return `${Math.round(s / 86400)} days ago`;
    },
    until: (iso) => {
      if (!iso) return "–";
      let s = Math.max(0, (new Date(iso).getTime() - Date.now()) / 1000);
      const d = Math.floor(s / 86400); s -= d * 86400;
      const hh = Math.floor(s / 3600); s -= hh * 3600;
      const m = Math.floor(s / 60);
      return d ? `${d}d ${hh}h` : hh ? `${hh}h ${m}m` : `${m}m`;
    },
    odds: (p) => (p ? (1 / p).toFixed(2) : "–"),
  };
  const titleCase = (s) => String(s || "").replace(/(^|[-_ ])(\w)/g, (_, a, b) => (a ? " " : "") + b.toUpperCase());

  // ---------- icons (24px stroke) ----------
  const S = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
  const ICONS = {
    gameweek: S('<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/><path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/>'),
    players: S('<circle cx="10.5" cy="10.5" r="6"/><path d="m15 15 6 6"/><path d="M8 10.5h5M10.5 8v5"/>'),
    team: S('<path d="M8 3 3.5 5.5 5 10l2-.8V21h10V9.2l2 .8 1.5-4.5L16 3c-.6 1.6-2.1 2.6-4 2.6S8.6 4.6 8 3Z"/>'),
    benchmarks: S('<path d="M3 21h18"/><rect x="9" y="7" width="6" height="14"/><rect x="3" y="12" width="6" height="9"/><rect x="15" y="10" width="6" height="11"/><path d="m12 2 .9 1.8 2 .3-1.4 1.4.3 2-1.8-.9-1.8.9.3-2L9.1 4.1l2-.3z"/>'),
    ledger: S('<path d="M4 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2a2 2 0 0 0 0 4v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2a2 2 0 0 0 0-4Z"/><path d="M14 5v12" stroke-dasharray="2 2.2"/>'),
    scorers: S('<circle cx="12" cy="12" r="9"/><path d="m12 7 4 3-1.5 4.7h-5L8 10z"/><path d="M12 3v4M21 10.5l-5-.5M18 19l-3.5-4.3M6 19l3.5-4.3M3 10.5l5-.5"/>'),
    fixtures: S('<rect x="3" y="4.5" width="18" height="16" rx="2.5"/><path d="M3 9.5h18M8 2.5v4M16 2.5v4"/><path d="M7.5 13.5h3M13.5 13.5h3M7.5 17h3"/>'),
    lab: S('<path d="M9 3h6M10 3v6.5L4.8 18.3A1.8 1.8 0 0 0 6.4 21h11.2a1.8 1.8 0 0 0 1.6-2.7L14 9.5V3"/><path d="M7.5 14.5h9"/>'),
    health: S('<path d="M3 12h4l2.5-6 4 12 2.5-6h5"/>'),
    data: S('<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/><path d="M4.5 5.5v13c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8v-13"/><path d="M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8"/>'),
    access: S('<circle cx="8" cy="15" r="4.5"/><path d="m11.2 11.8 8.8-8.8M16.5 6.5l2.5 2.5M14.5 8.5l2 2"/>'),
    lock: S('<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7.5a4 4 0 0 1 8 0V11"/>'),
    back: S('<path d="m15 5-7 7 7 7"/>'),
    sun: S('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
    moon: S('<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z"/>'),
    user: S('<circle cx="12" cy="8" r="4"/><path d="M4 21c1.3-4 4.4-6 8-6s6.7 2 8 6"/>'),
    out: S('<path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 17l5-5-5-5M15 12H3"/>'),
    clock: S('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
    check: S('<path d="m5 12.5 4.5 4.5L19 7.5"/>'),
    x: S('<path d="M6 6l12 12M18 6 6 18"/>'),
    inbox: S('<path d="M3 13h5l1.5 3h5l1.5-3h5"/><path d="M5.5 5h13L21 13v6a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-6z"/>'),
    ext: S('<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
  };

  function trefoil(cls, spin) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("class", cls || "mark");
    svg.setAttribute("aria-hidden", "true");
    const sector = (a0, a1, r1, r2) => {
      const p = (r, a) => [50 + r * Math.cos((a * Math.PI) / 180), 50 + r * Math.sin((a * Math.PI) / 180)];
      const [x0, y0] = p(r2, a0), [x1, y1] = p(r2, a1), [x2, y2] = p(r1, a1), [x3, y3] = p(r1, a0);
      return `M${x0} ${y0}A${r2} ${r2} 0 0 1 ${x1} ${y1}L${x2} ${y2}A${r1} ${r1} 0 0 0 ${x3} ${y3}Z`;
    };
    const g = document.createElementNS(ns, "g");
    if (spin) g.setAttribute("class", "spin");
    for (const c of [-90, 30, 150]) {
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", sector(c - 30, c + 30, 15, 44));
      path.setAttribute("class", "blade");
      path.setAttribute("fill", "currentColor");
      g.append(path);
    }
    svg.append(g);
    const hub = document.createElementNS(ns, "circle");
    hub.setAttribute("cx", "50"); hub.setAttribute("cy", "50"); hub.setAttribute("r", "9");
    hub.setAttribute("class", "hub"); hub.setAttribute("fill", "currentColor");
    svg.append(hub);
    return svg;
  }

  // ---------- theme ----------
  const Theme = {
    get() { try { return localStorage.getItem("sn-theme") || "system"; } catch { return "system"; } },
    set(v) {
      try { localStorage.setItem("sn-theme", v); } catch { /* private mode */ }
      Theme.apply();
    },
    apply() {
      const v = Theme.get();
      if (v === "system") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", v);
    },
  };
  Theme.apply();

  // ---------- backends ----------
  function supabaseBackend() {
    const sb = window.supabase.createClient(CFG.supabaseUrl, CFG.supabaseAnonKey, {
      auth: { persistSession: true, autoRefreshToken: true },
    });
    const err = (e) => { if (e) throw new Error(e.message || String(e)); };
    return {
      mode: "live",
      async session() { const { data } = await sb.auth.getSession(); return data.session; },
      async signIn(email, password) { const { error } = await sb.auth.signInWithPassword({ email, password }); err(error); },
      async signUp(name, email, password, note) {
        const { data, error } = await sb.auth.signUp({ email, password, options: { data: { name, note } } });
        err(error);
        return { confirm: !data.session };
      },
      async signOut() { await sb.auth.signOut(); },
      async me() {
        const { data: s } = await sb.auth.getSession();
        if (!s.session) return null;
        const uid = s.session.user.id;
        const { data, error } = await sb.from("members").select("*").eq("user_id", uid).maybeSingle();
        err(error);
        return data
          ? { id: uid, role: data.role, name: data.name, email: data.email }
          : { id: uid, role: "pending", name: s.session.user.user_metadata?.name || "", email: s.session.user.email };
      },
      async snapshot() {
        const { data, error } = await sb.from("snapshots").select("data, updated_at").eq("id", "latest").maybeSingle();
        err(error);
        return data ? data.data : null;
      },
      async requests() {
        const { data, error } = await sb.from("members").select("*").order("requested_at", { ascending: false });
        err(error);
        return data || [];
      },
      async decide(uid, role) {
        const { error } = await sb.from("members").update({ role, decided_at: new Date().toISOString() }).eq("user_id", uid);
        err(error);
      },
    };
  }

  function previewBackend() {
    // Simulates the access flow in memory; nothing leaves the page.
    let me = null;
    const ago = (h) => new Date(Date.now() - h * 3600e3).toISOString();
    const members = [
      { user_id: "ex-1", name: "Example: Sam", email: "sam@example.com", note: "Plays in the office league", role: "pending", requested_at: ago(3), example: true },
      { user_id: "ex-2", name: "Example: Priya", email: "priya@example.com", note: "Friend from uni", role: "pending", requested_at: ago(20), example: true },
      { user_id: "ex-3", name: "Example: Alex", email: "alex@example.com", note: "", role: "friend", requested_at: ago(70), example: true },
    ];
    return {
      mode: "preview",
      async session() { return me; },
      async signIn() { me = { id: "owner", role: "admin", name: "Owner", email: "you (preview)" }; },
      async signUp(name, email) {
        me = { id: "friend", role: "pending", name, email };
        members.unshift({ user_id: "friend", name, email, note: "", role: "pending", requested_at: new Date().toISOString() });
        return { confirm: false };
      },
      async signOut() { me = null; },
      async me() { return me; },
      async snapshot() { return EMBEDDED; },
      async requests() { return members; },
      async decide(uid, role) { const m = members.find((x) => x.user_id === uid); if (m) m.role = role; if (me && me.id === uid) me.role = role; },
      simulateApproval() { if (me) me.role = "friend"; },
    };
  }

  // ---------- app state ----------
  const state = { api: null, me: null, snap: null, requests: [], runs: null };
  const root = () => $("#app");

  // ---------- loader ----------
  const Loader = {
    el: null,
    mount() {
      const word = h("div", { class: "wordmark", "aria-label": "Stats Nuke" });
      [..."STATS NUKE"].forEach((c, i) =>
        word.append(c === " " ? h("span", { class: "gap" }) : h("span", { style: { animationDelay: `${0.35 + i * 0.045}s` } }, c)),
      );
      const reactor = h("div", { class: "reactor" }, h("div", { class: "halo" }));
      const svg = trefoil("", true);
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("cx", "50"); ring.setAttribute("cy", "50"); ring.setAttribute("r", "50"); ring.setAttribute("class", "ring");
      ring.setAttribute("stroke-dasharray", "315"); ring.setAttribute("stroke-dashoffset", "315");
      svg.append(ring);
      reactor.append(svg);
      this.el = h(
        "div",
        { class: "loader", role: "status", "aria-live": "polite" },
        h("div", { class: "loader-core" }, reactor, word,
          h("div", { class: "loader-status" }, h("div", { class: "bar" }, h("i")), h("div", { class: "loader-step" }, "Starting up"))),
      );
      document.body.append(this.el);
      this.started = Date.now();
    },
    step(text, frac) {
      if (!this.el) return;
      $(".loader-step", this.el).textContent = text;
      $(".bar i", this.el).style.width = `${Math.round(frac * 100)}%`;
    },
    async done() {
      const min = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 1500;
      const wait = Math.max(0, min - (Date.now() - this.started));
      this.step("Ready", 1);
      await new Promise((r) => setTimeout(r, wait));
      this.el.classList.add("done");
      setTimeout(() => this.el && this.el.remove(), 700);
    },
  };

  const toast = (msg) => {
    let t = $(".toast");
    if (!t) { t = h("div", { class: "toast", role: "status" }); document.body.append(t); }
    t.textContent = msg;
    t.classList.add("on");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.remove("on"), 2400);
  };

  // ---------- apps ----------
  const APPS = [
    { id: "gameweek", name: "Gameweek", tone: "t-yellow", sub: (s) => (s.next.gw ? `GW${s.next.gw}` : ""), view: viewGameweek },
    { id: "players", name: "Players", tone: "t-blue", sub: (s) => `${s.players.length}`, view: viewPlayers },
    { id: "team", name: "Model team", tone: "t-ink", sub: (s) => teamSub(s), view: viewTeam },
    { id: "benchmarks", name: "Benchmarks", tone: "t-paper", sub: () => "p50 · p90 · OpenFPL", view: viewBenchmarks },
    { id: "fixtures", name: "Fixtures", tone: "t-paper", sub: () => "ticker", view: viewFixtures },
    { id: "ledger", name: "Ledger", tone: "t-ink", sub: (s) => `${s.bets.bets.length} paper bets`, view: viewLedger },
    { id: "scorers", name: "Scorers", tone: "t-blue", sub: () => "props test", view: viewScorers },
    { id: "lab", name: "Lab", tone: "t-yellow", sub: () => "experiments", view: viewLab },
    { id: "health", name: "Health", tone: "t-ink", admin: true, sub: () => "pipeline", view: viewHealth },
    { id: "data", name: "Data", tone: "t-paper", admin: true, sub: () => "lake checks", view: viewData },
    { id: "access", name: "Access", tone: "t-yellow", admin: true, sub: () => "requests", view: viewAccess },
  ];
  const isAdmin = () => state.me && state.me.role === "admin";
  const visibleApps = () => APPS.filter((a) => !a.admin || isAdmin());

  function teamSub(s) {
    const scored = s.team.filter((w) => w.points != null);
    if (!scored.length) return s.next.gw ? `starts GW${s.next.gw}` : "";
    return `${scored.reduce((a, w) => a + w.points, 0)} pts`;
  }

  // ---------- shell pieces ----------
  function topbar() {
    const theme = Theme.get();
    const pop = h("div", { class: "menu-pop", hidden: true });
    const btn = h("button", { class: "btn icon-btn", "aria-label": "Account", "aria-haspopup": "true", onclick: (e) => { e.stopPropagation(); pop.hidden = !pop.hidden; } }, icon("user"));
    pop.append(
      h("div", { class: "who" }, h("div", { class: "pname" }, state.me.name || "Signed in"), h("div", { class: "small muted" }, state.me.email || ""),
        h("div", { style: { marginTop: "6px" } }, h("span", { class: `chip ${state.me.role === "admin" ? "accent" : "info"}` }, state.me.role === "admin" ? "Owner" : "Friend"))),
      h("div", { class: "eyebrow", style: { padding: "6px 12px 4px" } }, "Theme"),
      h("div", { style: { padding: "0 6px 6px" } }, seg(["system", "light", "dark"], theme, (v) => { Theme.set(v); render(); }, (v) => titleCase(v))),
      h("button", { class: "btn ghost", onclick: async () => { await state.api.signOut(); state.me = null; render(); } }, icon("out"), "Sign out"),
    );
    return h("header", { class: "topbar" },
      h("a", { class: "brand", href: "#" }, trefoil("mark"), h("span", { class: "wordmark" }, "Stats Nuke")),
      h("div", { class: "top-actions" }, statusChip(), h("div", { class: "menu" }, btn, pop)));
  }

  function worst(checks) {
    const order = { fail: 3, warn: 2, info: 1, ok: 0 };
    return checks.reduce((w, c) => (order[c.status] > order[w] ? c.status : w), "ok");
  }
  function statusChip() {
    const s = state.snap;
    if (!s) return null;
    if (isAdmin()) {
      const w = worst(s.data.checks);
      const n = s.data.checks.filter((c) => c.status === w).length;
      const text = w === "ok" || w === "info" ? "All systems nominal" : `${n} ${w === "fail" ? "failing" : "warning"}${n > 1 ? "s" : ""}`;
      return h("a", { class: `chip live ${w === "info" ? "ok" : w}`, href: "#data" }, h("span", { class: "dot" }), text);
    }
    return h("span", { class: "chip" }, `Updated ${fmt.ago(s.generated_at)}`);
  }

  function seg(options, value, onChange, label) {
    return h("div", { class: "seg", role: "tablist" },
      options.map((o) => h("button", { class: o === value ? "on" : "", role: "tab", "aria-selected": String(o === value), onclick: () => onChange(o) }, label ? label(o) : o)));
  }

  function pageHead(app, title, meta) {
    return h("div", {},
      h("a", { class: "btn ghost back", href: "#" }, icon("back"), "Home"),
      h("div", { class: "page-head" },
        h("div", { class: `tile ${app.tone}` }, icon(app.id)),
        h("div", {}, h("div", { class: "eyebrow" }, app.name), h("h1", {}, title)),
        h("div", { class: "meta small muted" }, meta || `Snapshot ${fmt.ago(state.snap.generated_at)}`)));
  }

  const section = (title, aside, ...body) =>
    h("section", { class: "section" }, h("div", { class: "section-head" }, h("h2", {}, title), aside ? h("div", { class: "small muted" }, aside) : null), ...body);
  const stat = (label, value, detail, hl) =>
    h("div", { class: `stat${hl ? " hl" : ""}` }, h("div", { class: "eyebrow" }, label), h("div", { class: "v" }, value), detail ? h("div", { class: "d" }, detail) : null);
  const empty = (iconName, title, detail) =>
    h("div", { class: "card empty" }, icon(iconName), h("div", { class: "pname" }, title), detail ? h("div", { class: "small" }, detail) : null);
  const chip = (status, text) => h("span", { class: `chip ${status}` }, h("span", { class: "dot" }), text || status);

  function table(cols, rows, opts = {}) {
    const thead = h("tr", {}, cols.map((c) => {
      const sortable = opts.onSort && c.key;
      return h("th", { class: [c.n ? "n" : "", sortable ? "sort" : "", c.key && opts.sortKey === c.key ? "on" : ""].join(" "), onclick: sortable ? () => opts.onSort(c.key) : null }, c.label);
    }));
    const body = rows.map((r) => h("tr", {}, cols.map((c) => h("td", { class: c.n ? "n" : "" }, c.render ? c.render(r) : r[c.key]))));
    return h("div", { class: "card flush" }, h("div", { class: "scroll" }, h("table", {}, h("thead", {}, thead), h("tbody", {}, body))));
  }

  // ---------- players helpers ----------
  const byId = () => {
    if (!state.pmap) state.pmap = new Map(state.snap.players.map((p) => [p.id, p]));
    return state.pmap;
  };
  const pinfo = (id) => byId().get(id) || { id, name: id.replace("fpl:", "#"), team: "", pos: "" };
  const playerCell = (p) => h("span", {}, h("span", { class: "pname" }, p.name), h("span", { class: "pteam" }, p.team));
  const posTag = (pos) => h("span", { class: `pos ${pos}` }, pos);
  const statusOf = (p) => {
    if (!p.status || p.status === "a") return null;
    const label = { d: "Doubt", i: "Injured", s: "Suspended", u: "Unavailable", n: "Not in squad" }[p.status] || p.status;
    return h("span", { class: `chip ${p.status === "d" ? "warn" : "fail"}`, title: p.news || label }, p.chance != null && p.status === "d" ? `${p.chance}%` : label);
  };
  const heatStyle = (x, max) => {
    const a = Math.max(0, Math.min(1, x / max));
    return { background: `color-mix(in srgb, var(--cherenkov) ${Math.round(a * 38)}%, transparent)` };
  };

  // ---------- charts ----------
  function lineChart(series, opts = {}) {
    const W = 640, H = opts.height || 220, L = 34, R = 12, T = 12, B = 24;
    const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
    if (!xs.length) return null;
    const ys = series.flatMap((s) => s.points.map((p) => p[1]));
    let lo = Math.min(0, ...ys), hi = Math.max(...ys);
    if (hi === lo) hi = lo + 1;
    const step = niceStep((hi - lo) / 4);
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    const x = (v) => L + (xs.length === 1 ? (W - L - R) / 2 : ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - L - R));
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" });
    for (let v = lo; v <= hi + 1e-9; v += step) {
      svg.append(h("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(h("text", { class: "axis", x: L - 6, y: y(v) + 3, "text-anchor": "end" }, Math.round(v)));
    }
    const every = Math.ceil(xs.length / 10);
    xs.forEach((v, i) => { if (i % every === 0 || i === xs.length - 1) svg.append(h("text", { class: "axis", x: x(v), y: H - 6, "text-anchor": "middle" }, `${opts.xPrefix || ""}${v}`)); });
    series.forEach((s, i) => {
      const k = s.k ?? i;
      const pts = s.points.map((p) => `${x(p[0])},${y(p[1])}`).join(" ");
      svg.append(h("polyline", { class: `s${k}`, points: pts, fill: "none", "stroke-width": s.bold ? 2.6 : 1.8, "stroke-linejoin": "round", "stroke-dasharray": s.dash ? "4 4" : null }));
      const last = s.points[s.points.length - 1];
      if (last) svg.append(h("circle", { class: `s${k}`, cx: x(last[0]), cy: y(last[1]), r: s.bold ? 4 : 3, style: { fill: "var(--panel)" }, "stroke-width": 2 }));
    });
    return h("div", {}, svg, h("div", { class: "legend" }, series.map((s, i) => h("span", {}, h("i", { class: `k${s.k ?? i}` }), s.name))));
  }
  function niceStep(raw) {
    const p = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-9)));
    const m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
  }
  function barList(rows, fmtv) {
    const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
    return h("div", { class: "bars" }, rows.map((r) =>
      h("div", { class: `bar-row ${r.cls || ""}` }, h("div", {}, r.label), h("div", { class: "track" }, h("i", { style: { width: `${(Math.abs(r.value) / max) * 100}%` } })), h("div", { class: "num small" }, fmtv ? fmtv(r.value) : r.value))));
  }

  // ---------- views ----------
  function viewHome() {
    const s = state.snap;
    const plan = s.plan;
    const scored = s.team.filter((w) => w.points != null);
    const ours = scored.reduce((a, w) => a + w.points, 0);
    const avg = scored.reduce((a, w) => a + (w.average || 0), 0);
    const bs = s.bets.summary;
    const facts = [
      ["Next deadline", s.next.gw ? fmt.until(s.next.deadline) : "Season over", s.next.gw ? `GW${s.next.gw} · ${fmt.date(s.next.deadline)}` : ""],
      ["Model team", plan ? `${fmt.n(plan.expected_points)} xP` : "–", plan ? `Captain ${pinfo(plan.captain).name}${plan.provisional ? " · provisional" : ""}` : "No plan yet"],
      ["Season", scored.length ? `${ours} pts` : "Starts GW6", scored.length ? `${fmt.signed(ours - avg)} vs average manager` : "First decision 30 h before the deadline"],
      ["Paper ledger", `${s.bets.bets.length} bets`, bs.bets ? `CLV ${fmt.spct(bs.mean_clv)} over ${bs.bets} settled` : "Waiting for closing prices"],
    ];
    const tile = (a, i) => {
      const badge = a.id === "access" ? state.requests.filter((r) => r.role === "pending").length : 0;
      return h("a", { class: "app pop", href: `#${a.id}`, style: { animationDelay: `${0.05 + i * 0.045}s` } },
        h("div", { class: `tile ${a.tone}` }, icon(a.id), badge ? h("span", { class: "badge" }, badge) : null, a.admin ? h("span", { class: "lock" }, icon("lock")) : null),
        h("div", { class: "label" }, a.name), h("div", { class: "sub" }, a.sub(s)));
    };
    const apps = visibleApps();
    return h("main", { class: "wrap view" },
      topbar(),
      h("div", { class: "hero" },
        h("div", { class: "eyebrow" }, `Season ${s.season.replace("-", "/")}${s.next.gw ? ` · Gameweek ${s.next.gw}` : ""}`),
        h("h1", {}, s.next.gw ? ["Gameweek ", s.next.gw, " locks in ", h("em", {}, fmt.until(s.next.deadline)), "."] : "The season is over."),
        h("p", {}, "Model v2 forecasts every player, plays its own FPL team against real managers and paper-trades the odds. Everything here refreshes every three hours.")),
      h("div", { class: "strip" }, facts.map(([k, v, d]) => h("div", { class: "fact" }, h("div", { class: "eyebrow" }, k), h("div", { class: "v" }, v), h("div", { class: "d" }, d)))),
      h("div", { class: "group" }, h("div", { class: "group-head" }, h("div", { class: "eyebrow" }, "Apps")),
        h("div", { class: "apps" }, apps.filter((a) => !a.admin).map(tile))),
      isAdmin()
        ? h("div", { class: "group" }, h("div", { class: "group-head" }, h("div", { class: "eyebrow" }, "Control room"), h("div", { class: "small muted" }, "Only you see these")),
          h("div", { class: "apps" }, apps.filter((a) => a.admin).map((a, i) => tile(a, i + 8))))
        : null,
    );
  }

  function pitch(plan, valueOf) {
    const rows = ["GK", "DEF", "MID", "FWD"].map((pos) => plan.xi.filter((id) => pinfo(id).pos === pos));
    const card = (id) => {
      const p = pinfo(id);
      return h("div", { class: "pcard" },
        id === plan.captain ? h("span", { class: "cap" }, "C") : id === plan.vice ? h("span", { class: "cap v" }, "V") : null,
        posTag(p.pos), h("div", { class: "nm" }, p.name), h("div", { class: "pteam", style: { margin: 0 } }, p.team), h("div", { class: "xp" }, valueOf(id)));
    };
    return h("div", { class: "pitch" }, rows.map((r) => h("div", { class: "line-up" }, r.map(card))), h("div", { class: "bench" }, plan.bench.map(card)));
  }

  function viewGameweek(app) {
    const s = state.snap;
    const plan = s.plan;
    const gw = s.next.gw;
    const out = [pageHead(app, gw ? `Gameweek ${gw}` : "No upcoming gameweek",
      gw ? h("div", {}, h("div", {}, `Deadline ${fmt.date(s.next.deadline)}`), h("div", {}, s.forecast.made_at ? `Forecast ${fmt.ago(s.forecast.made_at)}` : "No forecast yet")) : null)];
    if (!plan) {
      out.push(empty("clock", "No forecast for this deadline yet", "The provisional plan appears within 12 hours; the real decision is made 30 hours before the deadline."));
    } else {
      const capt = pinfo(plan.captain);
      const xp = (id) => (plan.expected || {})[id];
      out.push(section("The model team's plan", plan.provisional ? "Provisional: the decision is made 30 h before the deadline" : `Decided ${fmt.ago(plan.advised)}`,
        h("div", { class: "stats" },
          stat("Expected points", fmt.n(plan.expected_points), "XI with captain, after hits", true),
          stat("Captain", capt.name, `${fmt.n(xp(plan.captain))} xP · vice ${pinfo(plan.vice).name}`),
          (plan.sells || []).length
            ? stat("Transfers", `${plan.buys.length}`, plan.hits ? `${plan.hits} hit${plan.hits > 1 ? "s" : ""} (−${plan.hits * 4})` : `${plan.free_transfers_before ?? "–"} free before`)
            : stat("Squad", "Fresh", "£100m, picked from scratch"),
          stat("Bank", fmt.money(plan.bank / 10), plan.chip ? `Chip: ${titleCase(plan.chip)}` : "No chip")),
        pitch(plan, (id) => `${fmt.n(xp(id))} xP`),
        (plan.sells || []).length
          ? h("div", { class: "card" }, h("div", { class: "eyebrow" }, "Transfers"),
            h("div", { class: "cols", style: { marginTop: "12px" } },
              h("div", {}, h("div", { class: "small muted" }, "Out"), plan.sells.map((id) => h("div", {}, playerCell(pinfo(id))))),
              h("div", {}, h("div", { class: "small muted" }, "In"), plan.buys.map((id) => h("div", {}, playerCell(pinfo(id)))))))
          : null));
    }
    if (s.forecast.gws.length) {
      const top = (pos) => s.players.filter((p) => p.pos === pos && p.xp.length).sort((a, b) => b.xp[0] - a.xp[0]).slice(0, 6);
      out.push(section("Top picks by position", `Expected points, GW${s.forecast.gws[0]}`,
        h("div", { class: "cols two" }, ["GK", "DEF", "MID", "FWD"].map((pos) =>
          table([
            { label: pos, render: (p) => playerCell(p) },
            { label: "Price", n: true, render: (p) => fmt.n(p.price) },
            { label: "Start", n: true, render: (p) => fmt.pct(p.p_start) },
            { label: "xP", n: true, render: (p) => h("b", {}, fmt.n(p.xp[0], 2)) },
          ], top(pos))))));
      const caps = s.players.filter((p) => p.xp.length).sort((a, b) => b.xp[0] - a.xp[0]).slice(0, 10);
      out.push(section("Captain shortlist", "Highest expected points this gameweek",
        h("div", { class: "card" }, barList(caps.map((p) => ({ label: `${p.name} (${p.team})`, value: p.xp[0], cls: plan && p.id === plan.captain ? "me" : "alt" })), (v) => fmt.n(v, 2)))));
      const flagged = s.players.filter((p) => p.status !== "a" && (p.own || 0) >= 5).sort((a, b) => (b.own || 0) - (a.own || 0)).slice(0, 12);
      if (flagged.length)
        out.push(section("Team news", "Flagged players owned by 5 %+ of managers",
          table([
            { label: "Player", render: (p) => playerCell(p) },
            { label: "Owned", n: true, render: (p) => `${fmt.n(p.own)}%` },
            { label: "Status", render: (p) => statusOf(p) },
            { label: "News", render: (p) => h("span", { class: "small muted" }, p.news || "–") },
          ], flagged)));
    }
    return out;
  }

  const playersUi = { q: "", pos: "ALL", sort: "xp0", limit: 60, max: 16 };
  function viewPlayers(app) {
    const s = state.snap;
    const u = playersUi;
    const gws = s.forecast.gws;
    const sorters = {
      xp0: (p) => p.xp[0] || 0, xp5: (p) => p.xp5, price: (p) => p.price, own: (p) => p.own || 0,
      value: (p) => (p.price ? p.xp5 / p.price : 0), start: (p) => p.p_start || 0, goal: (p) => p.p_goal || 0,
    };
    if (!gws.length && ["xp0", "xp5", "value", "start", "goal"].includes(u.sort)) u.sort = "own";
    const q = u.q.trim().toLowerCase();
    let rows = s.players.filter((p) => (u.pos === "ALL" || p.pos === u.pos) && p.price <= u.max && (!q || `${p.name} ${p.full} ${p.team}`.toLowerCase().includes(q)));
    rows = rows.sort((a, b) => sorters[u.sort](b) - sorters[u.sort](a));
    const maxXp = Math.max(...s.players.map((p) => p.xp[0] || 0), 1);
    const setSort = (k) => { u.sort = k; render(); };
    const cols = [
      { label: "Player", render: (p) => h("span", {}, posTag(p.pos), " ", playerCell(p)) },
      { label: "Price", key: "price", n: true, render: (p) => fmt.n(p.price) },
      { label: "Own", key: "own", n: true, render: (p) => `${fmt.n(p.own)}%` },
      ...gws.map((g, i) => ({ label: `GW${g}`, key: i === 0 ? "xp0" : null, n: true, render: (p) => h("span", { class: "heat", style: heatStyle(p.xp[i] || 0, maxXp) }, fmt.n(p.xp[i], 1)) })),
      ...(gws.length ? [
        { label: `${gws.length} GW`, key: "xp5", n: true, render: (p) => h("b", {}, fmt.n(p.xp5)) },
        { label: "Pts/£m", key: "value", n: true, render: (p) => fmt.n(p.price ? p.xp5 / p.price : null, 2) },
        { label: "P(start)", key: "start", render: (p) => h("div", { class: "meter", title: fmt.pct(p.p_start) }, h("i", { style: { width: `${(p.p_start || 0) * 100}%` } })) },
        { label: "P(goal)", key: "goal", n: true, render: (p) => fmt.pct(p.p_goal) },
      ] : []),
      { label: "", render: (p) => statusOf(p) },
    ];
    const search = h("input", { class: "input", id: "player-search", type: "search", placeholder: "Search players or teams", value: u.q, style: { maxWidth: "280px" },
      oninput: (e) => { u.q = e.target.value; u.limit = 60; clearTimeout(viewPlayers.t); viewPlayers.t = setTimeout(() => { render(); const el = $("#player-search"); el.focus(); el.setSelectionRange(u.q.length, u.q.length); }, 160); } });
    const maxSel = h("select", { class: "input", id: "player-max", style: { width: "auto" }, onchange: (e) => { u.max = Number(e.target.value); render(); } },
      [16, 12, 10, 8, 7, 6, 5.5, 5, 4.5].map((v) => h("option", { value: v, selected: v === u.max }, v === 16 ? "Any price" : `≤ £${v}m`)));
    return [
      pageHead(app, "Every player, five gameweeks out", gws.length ? `Model v2 · GW${gws[0]}–${gws[gws.length - 1]}` : "No forecast yet"),
      h("div", { class: "section" },
        h("div", { class: "controls" }, search, seg(["ALL", "GK", "DEF", "MID", "FWD"], u.pos, (v) => { u.pos = v; render(); }, (v) => (v === "ALL" ? "All" : v)), maxSel),
        h("div", { class: "small muted" }, `${rows.length} players · sorted by ${({ xp0: "next gameweek", xp5: "five gameweeks", price: "price", own: "ownership", value: "points per £m", start: "P(start)", goal: "P(goal)" })[u.sort]} · tap a column to sort`),
        table(cols, rows.slice(0, u.limit), { onSort: setSort, sortKey: u.sort }),
        rows.length > u.limit ? h("button", { class: "btn", onclick: () => { u.limit += 100; render(); } }, `Show ${Math.min(100, rows.length - u.limit)} more`) : null),
    ];
  }

  function benchSeries() {
    const s = state.snap;
    const b = s.benchmarks;
    const from = s.team.length ? Math.min(...s.team.map((w) => w.gw)) : null;
    const scored = new Set(b.model.map((m) => m.gw));
    const cum = (pts) => { let t = 0; return pts.map(([g, v]) => [g, (t += v)]); };
    const series = [];
    if (b.model.length) series.push({ name: "Model team", k: 0, bold: true, points: cum(b.model.map((m) => [m.gw, m.points])) });
    const names = { p99: "Top 1% (p99)", p90: "Top 10% (p90)", p50: "Median manager (p50)" };
    const keys = { p99: 4, p90: 1, p50: 2 };
    for (const k of ["p99", "p90", "p50"]) {
      const pts = (b.panels[k] || []).filter((r) => (from == null || r.gw >= from) && (!scored.size || scored.has(r.gw)));
      if (pts.length) series.push({ name: names[k], k: keys[k], points: cum(pts.map((r) => [r.gw, r.points])) });
    }
    const avg = s.gameweeks.filter((g) => g.average != null && scored.has(g.gw));
    if (avg.length) series.push({ name: "FPL average", k: 3, dash: true, points: cum(avg.map((g) => [g.gw, g.average])) });
    return series;
  }

  function viewTeam(app) {
    const s = state.snap;
    const weeks = s.team;
    const scored = weeks.filter((w) => w.points != null);
    const out = [pageHead(app, "The model team", weeks.length ? `Since GW${weeks[0].gw}` : null)];
    if (!scored.length) {
      out.push(empty("team", "No gameweek scored yet", weeks.length ? `GW${weeks[0].gw} is decided; points arrive once FPL finalises the gameweek.` : `The model team starts at the GW${s.next.gw} deadline with £100m and a free squad.`));
      if (s.plan) out.push(section("Current squad plan", s.plan.provisional ? "Provisional" : "Decided", pitch(s.plan, (id) => `${fmt.n((s.plan.expected || {})[id])} xP`)));
      return out;
    }
    const total = scored.reduce((a, w) => a + w.points, 0);
    const avg = scored.reduce((a, w) => a + (w.average || 0), 0);
    const best = scored.reduce((a, w) => (w.points > a.points ? w : a));
    const hits = scored.reduce((a, w) => a + (w.hits || 0), 0);
    out.push(h("div", { class: "stats section" },
      stat("Season points", `${total}`, `${scored.length} gameweeks`, true),
      stat("Against the average", fmt.signed(total - avg), `average manager ${avg}`),
      stat("Best gameweek", `${best.points}`, `GW${best.gw}`),
      stat("Hits taken", `${hits}`, `−${hits * 4} points`)));
    const series = benchSeries();
    if (series.length) out.push(section("Running total", "Against real managers (cumulative)", h("div", { class: "card" }, lineChart(series, { xPrefix: "GW", label: "cumulative points" }))));
    const last = scored[scored.length - 1];
    if (last.picks) {
      out.push(section(`GW${last.gw} in detail`, `${last.points} points · expected ${fmt.n(last.expected_points)}`,
        pitch(last, (id) => `${last.picks[id] ? last.picks[id].points * (id === last.captain ? (last.chip === "triple_captain" ? 3 : 2) : 1) : 0} pts`)));
    }
    out.push(section("Week by week", null, table([
      { label: "GW", render: (w) => h("b", {}, w.gw) },
      { label: "Points", n: true, render: (w) => (w.points != null ? h("b", {}, w.points) : "–") },
      { label: "Expected", n: true, render: (w) => fmt.n(w.expected_points) },
      { label: "Average", n: true, render: (w) => w.average ?? "–" },
      { label: "Highest", n: true, render: (w) => w.highest ?? "–" },
      { label: "Captain", render: (w) => pinfo(w.captain).name },
      { label: "Transfers", render: (w) => ((w.buys || []).length ? `${w.buys.map((id) => pinfo(id).name).join(", ")}${w.hits ? ` (−${w.hits * 4})` : ""}` : "–") },
      { label: "Chip", render: (w) => (w.chip ? titleCase(w.chip) : "") },
    ], [...weeks].reverse())));
    return out;
  }

  function viewBenchmarks(app) {
    const s = state.snap;
    const b = s.benchmarks;
    const out = [pageHead(app, "How it stacks up", "Real managers, OpenFPL and the backtests")];
    const series = benchSeries();
    const scoredGws = new Set(b.model.map((m) => m.gw));
    if (b.model.length && series.length) {
      const totals = series.map((x) => ({ label: x.name, value: x.points[x.points.length - 1][1], cls: x.k === 0 ? "me" : x.k === 1 ? "alt" : "" }));
      out.push(section("This season against real managers", `Gameweeks ${Math.min(...scoredGws)}–${Math.max(...scoredGws)}, points net of hits`,
        h("div", { class: "cols" }, h("div", { class: "card" }, lineChart(series, { xPrefix: "GW" })), h("div", { class: "card" }, barList(totals.sort((a, c) => c.value - a.value), (v) => Math.round(v))))));
    } else {
      out.push(section("This season against real managers", null,
        empty("benchmarks", "Comparison starts after the model team's first scored gameweek",
          Object.keys(b.panel_meta || {}).length ? "Manager panels are chosen and tracked." : "Panels of managers at the median, top 10 % and top 1 % are chosen on the first live run.")));
    }
    if (b.panel_meta && Object.keys(b.panel_meta).length)
      out.push(h("p", { class: "note" }, `Panels: 20 managers each at overall rank ${Object.entries(b.panel_meta).map(([k, v]) => `${v.rank.toLocaleString()} (${k})`).join(", ")}, chosen ${fmt.day(Object.values(b.panel_meta)[0].chosen_at)}. Each benchmark is the panel's median gameweek score.`));
    if (b.note) out.push(h("p", { class: "note" }, b.note));
    const live = b.openfpl_live || [];
    out.push(section("Against OpenFPL, live", "Mean squared error of expected vs actual points, same player-fixtures (lower is better)",
      live.length
        ? table([
            { label: "GW", render: (r) => h("b", {}, r.gw) },
            { label: "Players", n: true, render: (r) => r.n },
            { label: "Model v2", n: true, render: (r) => h("b", {}, fmt.n(r.mse_v2, 3)) },
            { label: "OpenFPL replica", n: true, render: (r) => fmt.n(r.mse_replica, 3) },
            { label: "Difference", n: true, render: (r) => h("span", { style: { color: r.mse_v2 <= r.mse_replica ? "var(--ok)" : "var(--fail)" } }, fmt.signed(r.mse_v2 - r.mse_replica, 3)) },
          ], live)
        : empty("lab", "Scored after the first live gameweek", "Both forecasts are stored at every deadline and scored once FPL finalises the points.")));
    const bt = b.backtest;
    out.push(section("Backtest", `${bt.seasons}, chosen on 2021/22; 2025/26 held out`,
      h("div", { class: "stats" },
        stat("Season replay vs OpenFPL", `${fmt.signed(bt.replay_vs_openfpl.per_gw, 2)}`, `points per gameweek [${fmt.signed(bt.replay_vs_openfpl.low, 2)}, ${fmt.signed(bt.replay_vs_openfpl.high, 2)}], p ${bt.replay_vs_openfpl.p}`, true),
        stat("Forecast error vs OpenFPL", fmt.signed(bt.mse_vs_openfpl.delta, 3), `MSE [${fmt.signed(bt.mse_vs_openfpl.low, 3)}, ${fmt.signed(bt.mse_vs_openfpl.high, 3)}]`),
        stat("Paper betting CLV", fmt.spct(bt.betting.clv), `${bt.betting.bets} bets, ${bt.betting.span} [${fmt.spct(bt.betting.low)}, ${fmt.spct(bt.betting.high)}]`)),
      h("div", { class: "cols" },
        h("div", { class: "card" }, h("div", { class: "eyebrow", style: { marginBottom: "12px" } }, "Season replay, total points over three seasons"),
          barList(bt.replay.map((r) => ({ label: r.strategy, value: r.total, cls: r.strategy === "Model v2" ? "me" : r.strategy.startsWith("OpenFPL") ? "alt" : "" })), (v) => v.toLocaleString())),
        table([
          { label: "Forecast", render: (r) => (r.model === "Model v2" ? h("b", {}, r.model) : r.model) },
          { label: "MSE", n: true, render: (r) => fmt.n(r.mse, 3) },
          { label: "MAE", n: true, render: (r) => fmt.n(r.mae, 3) },
          { label: "Spearman", n: true, render: (r) => fmt.n(r.spearman, 3) },
          { label: "Top-10", n: true, render: (r) => fmt.n(r.top10, 3) },
        ], bt.forecast))));
    return out;
  }

  const fixturesUi = { gw: null };
  function viewFixtures(app) {
    const s = state.snap;
    const gws = [...new Set(s.fixtures.map((f) => f.gw).filter((g) => g != null))].sort((a, b) => a - b);
    const u = fixturesUi;
    if (u.gw == null) u.gw = s.next.gw || gws[gws.length - 1];
    const out = [pageHead(app, "Fixtures and the ticker", `${s.fixtures.length} fixtures`)];
    // ticker: per team, the next five gameweeks, shaded by the team's expected FPL points
    const fg = s.forecast.gws;
    if (fg.length) {
      const teamXp = {};
      for (const p of s.players) {
        const t = (teamXp[p.team] ||= fg.map(() => []));
        p.xp.forEach((v, i) => t[i].push(v));
      }
      const score = Object.fromEntries(Object.entries(teamXp).map(([t, cols]) => [t, cols.map((c) => c.sort((a, b) => b - a).slice(0, 11).reduce((a, v) => a + v, 0))]));
      const opp = (team, gw) => s.fixtures.filter((f) => f.gw === gw && (f.home === team || f.away === team)).map((f) => (f.home === team ? f.away : f.home.toLowerCase()));
      const max = Math.max(...Object.values(score).flat(), 1);
      const teams = Object.keys(score).sort((a, b) => score[b].reduce((x, y) => x + y, 0) - score[a].reduce((x, y) => x + y, 0));
      out.push(section("Fixture ticker", "Top-11 expected FPL points per team; uppercase home, lowercase away",
        table([
          { label: "Team", render: (t) => h("b", {}, t) },
          ...fg.map((g, i) => ({ label: `GW${g}`, n: true, render: (t) => h("span", { class: "heat", style: heatStyle(score[t][i], max) }, `${opp(t, g).join(" ") || "–"} ${fmt.n(score[t][i], 0)}`) })),
          { label: "Total", n: true, render: (t) => h("b", {}, fmt.n(score[t].reduce((a, v) => a + v, 0), 0)) },
        ], teams)));
    }
    const idx = gws.indexOf(u.gw);
    const list = s.fixtures.filter((f) => f.gw === u.gw);
    out.push(section(`Gameweek ${u.gw}`, null,
      h("div", { class: "controls" },
        h("button", { class: "btn", disabled: idx <= 0, onclick: () => { u.gw = gws[idx - 1]; render(); } }, icon("back"), "Previous"),
        h("button", { class: "btn", disabled: idx >= gws.length - 1, onclick: () => { u.gw = gws[idx + 1]; render(); } }, "Next")),
      h("div", { class: "card flush" }, h("div", { class: "list" }, list.map((f) =>
        h("div", { class: "row" }, h("div", { class: "small muted num", style: { minWidth: "110px" } }, fmt.date(f.kickoff)),
          h("div", { class: "t" }, `${f.home} v ${f.away}`),
          h("div", { class: "num" }, f.hg != null ? h("b", {}, `${f.hg}–${f.ag}`) : h("span", { class: "muted" }, "–"))))))));
    return out;
  }

  function viewLedger(app) {
    const s = state.snap;
    const bs = s.bets.summary;
    const out = [pageHead(app, "The paper ledger", "Paper only: nothing is ever placed")];
    out.push(h("div", { class: "stats section" },
      stat("Paper bets", `${s.bets.bets.length}`, `${bs.bets || 0} settled`, true),
      stat("Mean CLV", bs.bets ? fmt.spct(bs.mean_clv) : "–", bs.bets ? `[${fmt.spct(bs.clv_low)}, ${fmt.spct(bs.clv_high)}]` : "after the first closes"),
      stat("Hit rate", bs.bets ? fmt.pct(bs.hit_rate) : "–", "settled bets"),
      stat("Paper ROI", bs.bets ? fmt.spct(bs.roi) : "–", "1 unit per bet")));
    out.push(h("p", { class: "note section" }, "A paper bet is a soft bookmaker's price above the Betfair exchange's de-vigged fair price by more than 3 %. Closing-line value (CLV) compares the price taken with the fair price at the last snapshot before kickoff. Backtest 2016/17–2024/25: +2.9 % CLV over 466 bets."));
    if (!s.bets.bets.length) out.push(empty("ledger", "No paper bets yet", "They appear when an edge clears 3 %."));
    else
      out.push(section("All paper bets", null, table([
        { label: "Match", render: (b) => h("span", {}, h("b", {}, `${b.home} v ${b.away}`), h("div", { class: "small muted" }, fmt.date(b.kickoff))) },
        { label: "Pick", render: (b) => `${b.market === "1x2" ? "Result" : "Goals 2.5"}: ${titleCase(b.outcome)}` },
        { label: "Book", render: (b) => titleCase(b.book.replace(/_(uk|eu)$/, "")) },
        { label: "Price", n: true, render: (b) => h("b", {}, fmt.n(b.price, 2)) },
        { label: "Fair", n: true, render: (b) => fmt.odds(b.fair) },
        { label: "Edge", n: true, render: (b) => fmt.spct(b.ev) },
        { label: "CLV", n: true, render: (b) => (b.clv != null ? h("span", { style: { color: b.clv >= 0 ? "var(--ok)" : "var(--fail)" } }, fmt.spct(b.clv)) : "–") },
        { label: "Result", render: (b) => (b.settled ? chip(b.won ? "ok" : "fail", b.won ? `Won ${fmt.signed(b.profit, 2)}u` : "Lost") : chip("info", "Open")) },
      ], s.bets.bets)));
    return out;
  }

  function viewScorers(app) {
    const p = state.snap.props;
    const out = [pageHead(app, "Anytime scorers", "Our P(score) against the bookmakers")];
    out.push(h("p", { class: "note section" }, "For each player with anytime-scorer prices, our probability that they score (given they play) is compared with the bookmakers' implied probability and an equal blend. Lower log loss is better. A blended attack input is adopted only once it beats both over enough gameweeks."));
    if (!p.weeks.length) {
      out.push(empty("scorers", "No scored gameweek yet", p.note || "Prices are captured in the closing windows before kickoff."));
      return out;
    }
    const se = p.season;
    out.push(h("div", { class: "stats section" },
      stat("Ours", fmt.n(se.ours, 4), "log loss", se.ours <= Math.min(se.market, se.blend)),
      stat("Market", fmt.n(se.market, 4), "log loss", se.market < Math.min(se.ours, se.blend)),
      stat("Blend", fmt.n(se.blend, 4), "log loss", se.blend < Math.min(se.ours, se.market)),
      stat("Player-fixtures", `${se.n}`, "played, with prices")));
    out.push(section("By gameweek", null, table([
      { label: "GW", render: (r) => h("b", {}, r.gw) },
      { label: "Players", n: true, render: (r) => r.n },
      { label: "Ours", n: true, render: (r) => fmt.n(r.ours, 4) },
      { label: "Market", n: true, render: (r) => fmt.n(r.market, 4) },
      { label: "Blend", n: true, render: (r) => fmt.n(r.blend, 4) },
    ], p.weeks)));
    return out;
  }

  function viewLab(app) {
    const lab = state.snap.lab.filter((x) => !/queue|tonight/i.test(x.title));
    const out = [pageHead(app, "The lab", "Every experiment, kept only if its CI excludes 0")];
    const statusCell = (txt) => {
      const t = String(txt || "");
      const cls = t.startsWith("✅") ? "ok" : t.startsWith("❌") ? "fail" : t.startsWith("⏳") ? "info" : t.startsWith("➖") || t.startsWith("💤") ? "warn" : "";
      const label = t.replace(/^[✅❌⏳➖💤]\s*/u, "");
      return cls ? h("span", {}, h("span", { class: `chip ${cls}` }, h("span", { class: "dot" }), { ok: "Kept", fail: "Rejected", info: "Running", warn: "Parked" }[cls]), label ? h("div", { class: "small muted", style: { marginTop: "4px" } }, md(label)) : null) : md(t);
    };
    for (const sec of lab) {
      if (sec.level === 1) continue;
      const body = sec.tables.map((t) => {
        const si = t.header.findIndex((c) => /status/i.test(c));
        return table(t.header.map((c, i) => ({ label: c.replace(/\*\*/g, ""), render: (r) => (i === si ? statusCell(r[i]) : md(r[i])) })), t.rows);
      });
      const notes = sec.notes.slice(0, 2).map((n) => h("p", { class: "note" }, md(n.replace(/^[-*]\s*/, ""))));
      out.push(section(sec.title.replace(/^[✅❌]\s*/u, ""), null, ...notes, ...body));
    }
    return out;
  }

  async function loadRuns() {
    // live from GitHub's public API; the snapshot's copy is the fallback
    try {
      const r = await fetch(`https://api.github.com/repos/${REPO}/actions/runs?per_page=60`, { headers: { Accept: "application/vnd.github+json" } });
      if (!r.ok) throw new Error(String(r.status));
      const j = await r.json();
      state.runs = {
        live: true,
        at: new Date().toISOString(),
        runs: j.workflow_runs.map((run) => ({ workflow: run.name, event: run.event, status: run.status, conclusion: run.conclusion, started: run.run_started_at, updated: run.updated_at, branch: run.head_branch, sha: (run.head_sha || "").slice(0, 7), url: run.html_url })),
      };
    } catch {
      state.runs = { live: false, at: state.snap.generated_at, runs: state.snap.health.runs || [] };
    }
    if (location.hash === "#health") render();
  }

  function viewHealth(app) {
    if (!state.runs) loadRuns();
    const src = state.runs || { live: false, at: state.snap.generated_at, runs: state.snap.health.runs || [] };
    const runs = src.runs;
    const out = [pageHead(app, "System health", h("div", {}, src.live ? chip("ok", "Live from GitHub") : chip("info", "From the last snapshot"), h("div", { style: { marginTop: "6px" } }, `Checked ${fmt.ago(src.at)}`)))];
    const by = {};
    for (const r of runs) (by[r.workflow] ||= []).push(r);
    const state_ = (r) => (r.status !== "completed" ? "running" : r.conclusion === "success" ? "success" : r.conclusion === "failure" ? "failure" : r.conclusion || "skipped");
    const tone = (st) => ({ success: "ok", failure: "fail", running: "info" })[st] || "warn";
    const cards = Object.entries(by).map(([wf, list]) => {
      const last = list[0];
      const lastOk = list.find((r) => r.conclusion === "success");
      const done = list.filter((r) => r.status === "completed");
      const rate = done.length ? done.filter((r) => r.conclusion === "success").length / done.length : null;
      return h("div", { class: "card", style: { display: "grid", gap: "10px" } },
        h("div", { class: "section-head" }, h("h2", {}, wf), chip(tone(state_(last)), titleCase(state_(last)))),
        h("div", { class: "small muted" }, `Last run ${fmt.ago(last.started)} · ${last.event} · ${last.branch}`),
        h("div", { class: "small muted" }, lastOk ? `Last success ${fmt.ago(lastOk.updated)}` : "No success in recent runs"),
        h("div", { class: "spark", "aria-label": "recent runs, oldest left" }, list.slice(0, 24).reverse().map((r) => h("i", { class: state_(r), style: { height: `${state_(r) === "success" ? 100 : 60}%` }, title: `${state_(r)} · ${fmt.date(r.started)}` }))),
        h("div", { class: "small" }, rate != null ? `${fmt.pct(rate)} of the last ${done.length} succeeded` : ""));
    });
    out.push(section("Workflows", "Live every 3 h · Collect every 3 h, hourly near deadlines · CI on every push", runs.length ? h("div", { class: "cols" }, cards) : empty("health", "No runs visible", state.snap.health.note)));
    if (runs.length)
      out.push(section("Recent runs", null, table([
        { label: "Workflow", render: (r) => h("b", {}, r.workflow) },
        { label: "Result", render: (r) => chip(tone(state_(r)), titleCase(state_(r))) },
        { label: "Started", render: (r) => fmt.date(r.started) },
        { label: "Trigger", render: (r) => r.event },
        { label: "Commit", render: (r) => h("span", { class: "mono small" }, r.sha) },
        { label: "", render: (r) => h("a", { href: r.url, target: "_blank", rel: "noopener" }, "Open ", icon("ext", "ico")) },
      ], runs.slice(0, 25))));
    return out;
  }

  function viewData(app) {
    const d = state.snap.data;
    const counts = { ok: 0, warn: 0, fail: 0, info: 0 };
    d.checks.forEach((c) => (counts[c.status] = (counts[c.status] || 0) + 1));
    const out = [pageHead(app, "Data and checks", `Lake as of ${fmt.date(state.snap.generated_at)}`)];
    out.push(h("div", { class: "stats section" },
      stat("Checks passing", `${counts.ok}/${d.checks.length}`, `${counts.warn} warning · ${counts.fail} failing`, counts.fail === 0 && counts.warn === 0),
      stat("Feeds", `${d.feeds.length}`, `${d.feeds.reduce((a, f) => a + f.captures, 0).toLocaleString()} captures`),
      stat("Silver rows", d.silver.reduce((a, t) => a + t.rows, 0).toLocaleString(), `${d.silver.length} tables`)));
    out.push(section("Pipeline checks", null, h("div", { class: "card flush" }, h("div", { class: "list" }, d.checks.map((c) =>
      h("div", { class: "row" }, h("div", { class: `sev ${c.status}` }), h("div", {}, h("div", { class: "t" }, c.name), h("div", { class: "d" }, c.detail)), chip(c.status === "info" ? "info" : c.status, { ok: "OK", warn: "Warning", fail: "Failing", info: "Info" }[c.status])))))));
    out.push(section("Feeds", "Bronze captures by source", table([
      { label: "Feed", render: (f) => h("span", { class: "mono small" }, f.feed) },
      { label: "Captures", n: true, render: (f) => f.captures.toLocaleString() },
      { label: "First", render: (f) => fmt.day(f.first) },
      { label: "Last", render: (f) => fmt.ago(f.last) },
      { label: "Status", render: (f) => chip(f.status, { ok: "Fresh", warn: "Late", fail: "Stale", info: "On demand" }[f.status]) },
    ], d.feeds)));
    out.push(section("Silver tables", "Validated tables the models read", table([
      { label: "Table", render: (t) => h("span", { class: "mono small" }, t.table) },
      { label: "Rows", n: true, render: (t) => t.rows.toLocaleString() },
    ], d.silver)));
    return out;
  }

  const accessUi = { tab: "pending" };
  function viewAccess(app) {
    const u = accessUi;
    const reqs = state.requests;
    const tabs = { pending: reqs.filter((r) => r.role === "pending"), friend: reqs.filter((r) => r.role === "friend" || r.role === "admin"), rejected: reqs.filter((r) => r.role === "rejected") };
    const out = [pageHead(app, "Access requests", state.api.mode === "preview" ? chip("warn", "Preview: example requests") : `${reqs.length} accounts`)];
    const act = async (r, role, msg) => {
      try { await state.api.decide(r.user_id, role); r.role = role; toast(msg); render(); }
      catch (e) { toast(`Could not save: ${e.message}`); }
    };
    out.push(h("div", { class: "section" },
      seg(["pending", "friend", "rejected"], u.tab, (v) => { u.tab = v; render(); }, (v) => `${({ pending: "Pending", friend: "Members", rejected: "Declined" })[v]} (${tabs[v].length})`),
      tabs[u.tab].length
        ? h("div", { class: "card flush" }, h("div", { class: "list" }, tabs[u.tab].map((r) =>
          h("div", { class: "row" },
            h("div", { class: "status-orb", style: { width: "40px", height: "40px", margin: 0 } }, h("b", {}, (r.name || r.email || "?").trim()[0].toUpperCase())),
            h("div", {}, h("div", { class: "t" }, r.name || "No name", r.role === "admin" ? h("span", { class: "chip accent", style: { marginLeft: "8px" } }, "Owner") : null),
              h("div", { class: "d" }, r.email), r.note ? h("div", { class: "d" }, `“${r.note}”`) : null, h("div", { class: "d" }, `Requested ${fmt.ago(r.requested_at)}`)),
            r.role === "admin" ? null
              : h("div", { class: "controls" },
                r.role !== "friend" ? h("button", { class: "btn primary sm", onclick: () => act(r, "friend", `${r.name || r.email} can now sign in`) }, icon("check"), "Approve") : null,
                r.role !== "rejected" ? h("button", { class: "btn sm danger", onclick: () => act(r, "rejected", r.role === "friend" ? "Access revoked" : "Request declined") }, icon("x"), r.role === "friend" ? "Revoke" : "Decline") : null)))))
        : empty("inbox", { pending: "No pending requests", friend: "No members yet", rejected: "Nothing declined" }[u.tab], u.tab === "pending" ? "Friends request access from the sign-in screen with “I'm a friend”." : null)));
    out.push(h("p", { class: "note" }, "Approved friends see the apps (Gameweek, Players, Model team, Benchmarks, Fixtures, Ledger, Scorers, Lab). Health, Data and Access stay yours. Revoking takes effect on their next load."));
    return out;
  }

  // ---------- gate (signed out, pending, declined) ----------
  const gateUi = { tab: "signin", msg: "", busy: false };
  function viewGate() {
    const u = gateUi;
    const configured = state.api.mode !== "none";
    const field = (id, label, type, extra) => h("div", { class: "field" }, h("label", { for: id }, label), h(type === "textarea" ? "textarea" : "input", { class: "input", id, name: id, type: type === "textarea" ? null : type, ...extra }));
    const err = h("div", { class: "err", role: "alert" }, u.msg);
    const submit = async (e) => {
      e.preventDefault();
      if (u.busy) return;
      const f = e.target;
      u.busy = true; err.textContent = "";
      try {
        if (u.tab === "signin") await state.api.signIn(f.email.value.trim(), f.password.value);
        else {
          if (f.password.value.length < 8) throw new Error("Use a password of at least 8 characters.");
          const r = await state.api.signUp(f.name.value.trim(), f.email.value.trim(), f.password.value, f.note.value.trim());
          if (r.confirm) { u.msg = ""; u.tab = "confirm"; u.busy = false; render(); return; }
        }
        u.busy = false;
        await boot(true);
      } catch (ex) {
        u.busy = false;
        err.textContent = ex.message || "Something went wrong.";
      }
    };
    let body;
    if (!configured) {
      body = h("div", { class: "card", style: { display: "grid", gap: "12px" } }, h("div", { class: "pname" }, "The site isn't connected to its database yet"),
        h("p", { class: "small muted" }, "The owner adds the Supabase URL and public key as repository variables and redeploys. Setup steps are in web/README.md."));
    } else if (u.tab === "confirm") {
      body = h("div", { class: "card", style: { display: "grid", gap: "16px", textAlign: "center" } }, h("div", { class: "status-orb" }, icon("inbox")),
        h("div", { class: "pname" }, "Confirm your email"), h("p", { class: "small muted" }, "We sent you a link. Open it, then sign in here. Your request reaches the owner as soon as your email is confirmed."),
        h("button", { class: "btn", onclick: () => { u.tab = "signin"; render(); } }, "Back to sign in"));
    } else {
      body = h("div", { class: "card", style: { display: "grid", gap: "20px" } },
        seg(["signin", "friend"], u.tab, (v) => { u.tab = v; u.msg = ""; render(); }, (v) => (v === "signin" ? "Sign in" : "I'm a friend")),
        h("form", { class: "form", onsubmit: submit, novalidate: true },
          u.tab === "friend" ? [field("name", "Your name", "text", { required: true, autocomplete: "name", maxlength: 80 })] : null,
          field("email", "Email", "email", { required: true, autocomplete: "email" }),
          field("password", "Password", "password", { required: true, autocomplete: u.tab === "signin" ? "current-password" : "new-password", minlength: 8 }),
          u.tab === "friend" ? field("note", "A note for the owner (optional)", "textarea", { maxlength: 280, placeholder: "How you know them, which league you play in" }) : null,
          err,
          h("button", { class: "btn primary", type: "submit" }, u.tab === "signin" ? "Sign in" : "Request access")),
        u.tab === "friend" ? h("p", { class: "small muted" }, "The owner approves each request. You can sign in as soon as you're approved.") : null);
    }
    return h("main", { class: "gate view" }, h("div", { class: "gate-card" },
      h("div", { class: "gate-brand" }, trefoil("mark"), h("div", { class: "wordmark" }, "Stats Nuke"), h("p", {}, "FPL forecasts, a model team that plays the real game, and a paper betting ledger.")),
      body,
      state.api.mode === "preview" ? h("p", { class: "small muted", style: { textAlign: "center" } }, "Preview: any email and password signs you in as the owner.") : null));
  }

  function viewWaiting() {
    const declined = state.me.role === "rejected";
    const api = state.api;
    return h("main", { class: "gate view" }, h("div", { class: "gate-card" },
      h("div", { class: "gate-brand" }, trefoil("mark"), h("div", { class: "wordmark" }, "Stats Nuke")),
      h("div", { class: "card", style: { display: "grid", gap: "16px", textAlign: "center" } },
        h("div", { class: "status-orb", style: declined ? { background: "var(--fail-soft)", color: "var(--fail)" } : null }, icon(declined ? "x" : "clock")),
        h("div", { class: "pname" }, declined ? "Your request was declined" : "Request sent"),
        h("p", { class: "small muted" }, declined ? "Ask the owner if you think this is a mistake." : `Thanks${state.me.name ? `, ${state.me.name}` : ""}. You'll get in as soon as the owner approves. This page checks every 20 seconds.`),
        api.mode === "preview" && !declined ? h("button", { class: "btn primary", onclick: async () => { api.simulateApproval(); await boot(true); } }, "Preview: approve me") : null,
        h("button", { class: "btn ghost", onclick: async () => { await api.signOut(); state.me = null; render(); } }, icon("out"), "Sign out"))));
  }

  // ---------- render / boot ----------
  function render() {
    const el = root();
    el.replaceChildren();
    clearInterval(render.poll);
    if (!state.me) { el.append(viewGate()); return; }
    if (state.me.role === "pending" || state.me.role === "rejected") {
      el.append(viewWaiting());
      if (state.me.role === "pending") render.poll = setInterval(async () => { const me = await state.api.me(); if (me && me.role !== "pending") boot(true); }, 20000);
      return;
    }
    if (!state.snap) {
      el.append(h("main", { class: "gate view" }, h("div", { class: "gate-card" }, empty("data", "No data published yet", "The Live workflow uploads the first snapshot on its next run."))));
      return;
    }
    const id = location.hash.replace(/^#/, "");
    const app = visibleApps().find((a) => a.id === id);
    if (!app) { el.append(viewHome()); return; }
    el.append(h("main", { class: "wrap view" }, topbar(), ...[app.view(app)].flat()));
  }

  async function boot(quiet) {
    if (!quiet) Loader.step("Connecting", 0.2);
    try {
      state.me = await state.api.me();
      if (!quiet) Loader.step("Verifying access", 0.5);
      if (state.me && (state.me.role === "admin" || state.me.role === "friend")) {
        if (!quiet) Loader.step("Loading the latest snapshot", 0.75);
        state.snap = await state.api.snapshot();
        state.pmap = null;
        if (state.me.role === "admin") state.requests = await state.api.requests();
      }
    } catch (e) {
      console.error(e);
      toast(`Could not reach the server: ${e.message}`);
    }
    render();
  }

  async function main() {
    Loader.mount();
    document.addEventListener("click", () => document.querySelectorAll(".menu-pop").forEach((p) => (p.hidden = true)));
    if (CFG.supabaseUrl && CFG.supabaseAnonKey && window.supabase) state.api = supabaseBackend();
    else if (EMBEDDED) state.api = previewBackend();
    else state.api = { mode: "none", async me() { return null; } };
    await boot(false);
    await Loader.done();
    window.addEventListener("hashchange", () => { render(); window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" }); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
  else main();
})();
