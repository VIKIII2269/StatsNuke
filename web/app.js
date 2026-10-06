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

  // ---------- pixel icons (16×16 maps, drawn once to data URLs) ----------
  const PAL = { k: "#000000", w: "#ffffff", g: "#c0c0c0", d: "#808080", y: "#ffd800", o: "#d98200", b: "#1c6fe0", c: "#40d8e0", m: "#ff2fa4", r: "#e02020", G: "#22b14c", n: "#8a5a2a" };
  const PIX = {
    home: ["................", "................", ".kkkkk..........", "kyyyyyk.........", "kyyyyyykkkkkkkk.", "kyyyyyyyyyyyyyyk", "kkkkkkkkkkkkkkkk", "kyyyyyyyyyyyyyyk", "kyyyyyyyyyyyyyyk", "kyyyyyyyyyyyyyyk", "kyyyyyyyyyyyyyyk", "kyyyyyyyyyyyyyyk", "kyyyyyyyyyyyyyyk", "kooooooooooooook", "kkkkkkkkkkkkkkkk", "................"],
    gameweek: ["................", ".....kkkkkk.....", "...kkrrrrrrkk...", "..krrwwwwwwrrk..", "..krwwrrrrwwrk..", ".krwwrrrrrrwwrk.", ".krwrrwwwwrrwrk.", ".krwrrwkkwrrwrk.", ".krwrrwkkwrrwrk.", ".krwrrwwwwrrwrk.", ".krwwrrrrrrwwrk.", "..krwwrrrrwwrk..", "..krrwwwwwwrrk..", "...kkrrrrrrkk...", ".....kkkkkk.....", "................"],
    players: ["................", "..kkkkk.........", ".kcwwwck........", "kcwwnwwck.......", "kwwnnnwwk.......", "kwwwnwwwk.......", "kwwnnnwwk.......", "kcwnnnwck.......", ".kcwwwck........", "..kkkkkdd.......", ".......ddd......", "........ddd.....", ".........ddd....", "..........ddk...", "...........kk...", "................"],
    team: ["................", "....kkk..kkk....", "..kkyyykkyyykk..", ".kyyyyykkyyyyyk.", "kyyyyyyyyyyyyyyk", "kyyyyyyyyyyyyyyk", "kkkyyyyyyyyyykkk", "..kyyyyyyyyyyk..", "..kyyyykkyyyyk..", "..kyyyykyyyyyk..", "..kyyyykkkyyyk..", "..kyyyykykyyyk..", "..kyyyykkkyyyk..", "..kyyyyyyyyyyk..", "..kkkkkkkkkkkk..", "................"],
    benchmarks: ["................", "................", ".....kkkkkk.....", ".....kyyyyk.....", ".....kyyyyk.....", ".....kyyyyk.....", ".kkkkkyyyyk.....", ".kgggkyyyyk.....", ".kgggkyyyykkkkk.", ".kgggkyyyykoook.", ".kgggkyyyykoook.", ".kgggkyyyykoook.", ".kgggkyyyykoook.", ".kgggkyyyykoook.", "kkkkkkkkkkkkkkkk", "................"],
    fixtures: ["................", "...kk....kk.....", ".kkkkkkkkkkkkkk.", ".krrrrrrrrrrrrk.", ".krrrrrrrrrrrrk.", ".kkkkkkkkkkkkkk.", ".kwwwwwwwwwwwwk.", ".kwddwddwddwddk.", ".kwwwwwwwwwwwwk.", ".kwddwddwbbwddk.", ".kwwwwwwwwwwwwk.", ".kwddwddwddwddk.", ".kwwwwwwwwwwwwk.", ".kwddwddwwwwwwk.", ".kkkkkkkkkkkkkk.", "................"],
    ledger: ["................", "................", "....kkkkkkkk....", "...kyyyyyyyyk...", "...kooooooook...", "...kyyyyyyyyk...", "...kooooooook...", "...kyyyyyyyyk...", "...kooooooook...", "...kyyyyyyyyk...", "...kooooooook...", "...kyyyyyyyyk...", "...kooooooook...", "....kkkkkkkk....", "................", "................"],
    scorers: ["................", ".....kkkkkk.....", "...kkwwkkwwkk...", "..kwwwkkkkwwwk..", "..kwwwwkkwwwwk..", ".kkwwwwwwwwwwkk.", ".kkkwwwkkwwwkkk.", ".kkwwwkkkkwwwkk.", ".kwwwwwkkwwwwwk.", ".kwwwwwwwwwwwwk.", ".kkwwwwwwwwwwkk.", "..kkkwwkkwwkkk..", "..kwwwkkkkwwwk..", "...kkwwkkwwkk...", ".....kkkkkk.....", "................"],
    lab: ["................", "......kkkk......", "......kwwk......", "......kwwk......", "......kwwk......", ".....kwwwwk.....", "....kwwwwwwk....", "...kwwwwwwwwk...", "...kGGGGGGGGk...", "..kGGwGGGGGGGk..", "..kGGGGGGwGGGk..", ".kGGGGGGGGGGGGk.", ".kGGwGGGGGGGGGk.", ".kkkkkkkkkkkkkk.", "................", "................"],
    health: ["................", ".kkkkkkkkkkkkkk.", ".kggggggggggggk.", ".kgkkkkkkkkkkgk.", ".kgkkkkkkkkkkgk.", ".kgkkkkkGkkkkgk.", ".kgkkkkGkGkkkgk.", ".kgGGGGkkkGGGgk.", ".kgkkkkkkkkkkgk.", ".kgkkkkkkkkkkgk.", ".kggggggggggggk.", ".kkkkkkkkkkkkkk.", "......kggk......", "....kkkkkkkk....", "....kggggggk....", "....kkkkkkkk...."],
    data: ["................", "....kkkkkkkk....", "..kkwwwwwwwwkk..", "..kbkkwwwwkkbk..", "..kbbbkkkkbbbk..", "..kbbbbbbbbbbk..", "..kkbbbbbbbbkk..", "..kbkkbbbbkkbk..", "..kbbbkkkkbbbk..", "..kbbbbbbbbbbk..", "..kkbbbbbbbbkk..", "..kbkkbbbbkkbk..", "..kbbbkkkkbbbk..", "..kbbbbbbbbbbk..", "...kkkkkkkkkk...", "................"],
    access: ["................", "................", "................", "................", "..kkkk..........", ".kyyyyk.........", "kyykkyyk........", "kyk..kyk........", "kyk..kykkkkkkkk.", "kyykkyyyyyyyyyyk", ".kyyyyykkkkykkyk", "..kkkkk....k..k.", "................", "................", "................", "................"],
    lock: ["................", "................", ".....kkkkkk.....", "....kk....kk....", "....k......k....", "....k......k....", "..kkkkkkkkkkkk..", "..kyyyyyyyyyyk..", "..kyyyyyyyyyyk..", "..kyyyykkyyyyk..", "..kyyyykkyyyyk..", "..kyyyyyyyyyyk..", "..kyyyyyyyyyyk..", "..kkkkkkkkkkkk..", "................", "................"],
    wait: ["................", "...kkkkkkkkkk...", "...kggggggggk...", "....kyyyyyyk....", ".....kyyyyk.....", "......kyyk......", ".......kk.......", ".......kk.......", "......kwwk......", ".....kwwwwk.....", "....kwwyywwk....", "...kgyyyyyygk...", "...kkkkkkkkkk...", "................", "................", "................"],
  };
  const pixCache = {};
  function pixSrc(name) {
    if (pixCache[name]) return pixCache[name];
    const map = PIX[name] || PIX.home;
    const c = document.createElement("canvas");
    c.width = 16; c.height = 16;
    const g = c.getContext("2d");
    if (!g) return "";
    map.forEach((row, y) => [...row].forEach((ch, x) => { if (PAL[ch]) { g.fillStyle = PAL[ch]; g.fillRect(x, y, 1, 1); } }));
    return (pixCache[name] = c.toDataURL());
  }
  const pimg = (name, cls) => h("img", { src: pixSrc(name), alt: "", class: cls || null, draggable: "false" });

  // ---------- boot screen ----------
  const Loader = {
    el: null,
    mount() {
      this.started = Date.now();
      const boot = h("div", { class: "boot" });
      const bar = h("i");
      const splash = h("div", { class: "splash" }, h("div", { class: "splash-core" }, trefoil("", true),
        h("div", { class: "word" }, "Stats", h("b", {}, "Nuke")), h("div", { class: "ver" }, "FPL forecasting system · season 2026/27"),
        h("div", { class: "progress", role: "progressbar", "aria-label": "Loading" }, bar), h("div", { class: "ver step" }, "Starting up")));
      this.el = h("div", { class: "loader", role: "status", "aria-live": "polite" }, boot, splash);
      this.bar = bar;
      this.boot = boot;
      document.body.append(this.el);
      const lines = [
        ["STATSNUKE BIOS v6.0  ·  Phase 6 build", "hi"],
        ["Memory test ............ 640K ", "OK"],
        ["Mounting the lake ....... ", "OK"],
        ["Loading model v2 ........ ", "OK"],
        ["Odds tracker (paper only) ", "OK"],
      ];
      const fast = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      lines.forEach(([t, k], i) => setTimeout(() => {
        boot.append(k === "hi" ? h("span", { class: "hi" }, t) : t, k === "OK" ? h("span", { class: "ok" }, k) : "", "\n");
      }, fast ? 0 : 110 * i));
      setTimeout(() => $(".splash", this.el).classList.add("on"), fast ? 0 : 750);
    },
    step(text, frac) {
      if (!this.el) return;
      $(".step", this.el).textContent = text;
      this.bar.style.width = `${Math.round(frac * 100)}%`;
    },
    async done() {
      const min = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 2100;
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

  // ---------- status ----------
  function worst(checks) {
    const order = { fail: 3, warn: 2, info: 1, ok: 0 };
    return checks.reduce((w, c) => (order[c.status] > order[w] ? c.status : w), "ok");
  }
  function systemStatus() {
    const s = state.snap;
    const w = worst(s.data.checks);
    const n = s.data.checks.filter((c) => c.status === w).length;
    const tone = w === "info" ? "ok" : w;
    const text = tone === "ok" ? "all systems nominal" : `${n} ${w === "fail" ? "check failing" : "warning"}${n > 1 ? "s" : ""}`;
    return { tone, text };
  }

  function seg(options, value, onChange, label) {
    return h("div", { class: "seg", role: "tablist" },
      options.map((o) => h("button", { class: o === value ? "on" : "", role: "tab", "aria-selected": String(o === value), onclick: () => onChange(o) }, label ? label(o) : o)));
  }

  function pageHead(app, title, meta) {
    return h("div", { class: "page-head" },
      pimg(app.id),
      h("div", {}, h("span", { class: "hl-label" }, app.name), h("h1", {}, title)),
      h("div", { class: "meta" }, meta || `Snapshot ${fmt.ago(state.snap.generated_at)}`));
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
    return { background: `color-mix(in srgb, var(--cyan) ${Math.round(a * 38)}%, transparent)` };
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
      if (last) svg.append(h("circle", { class: `s${k}`, cx: x(last[0]), cy: y(last[1]), r: s.bold ? 4 : 3, style: { fill: "#000" }, "stroke-width": 2 }));
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
  const HOME = { id: "home", name: "Stats Nuke", pix: "home", view: viewHomeWin };
  const appById = (id) => (id === "home" ? HOME : APPS.find((a) => a.id === id));
  const link = (id, text) => h("a", { href: `#${id}`, onclick: (e) => { e.preventDefault(); openWin(id); } }, text);

  function viewHomeWin() {
    const s = state.snap;
    const plan = s.plan;
    const scored = s.team.filter((w) => w.points != null);
    const ours = scored.reduce((a, w) => a + w.points, 0);
    const avg = scored.reduce((a, w) => a + (w.average || 0), 0);
    const bt = s.benchmarks.backtest;
    const pending = state.requests.filter((r) => r.role === "pending").length;
    const st = systemStatus();
    return h("div", { class: "prose" },
      h("div", { class: "who-row" }, pimg("home"), h("div", {},
        h("span", { class: "hl-label" }, "What?"),
        h("div", {}, "I'm ", h("b", {}, "Stats Nuke"), ": a ", link("players", "forecasting system"), " for Fantasy Premier League and a ", link("ledger", "paper betting ledger"), "."))),
      h("p", {}, "Model v2 forecasts every player five gameweeks out, plays its own FPL team against real managers, and paper-trades the odds. Nothing is ever staked."),
      s.next.gw
        ? h("p", {}, "Gameweek ", s.next.gw, " locks in ", h("b", {}, fmt.until(s.next.deadline)), ". ",
          plan ? ["The model team expects ", h("b", {}, fmt.n(plan.expected_points)), " points with ", h("b", {}, pinfo(plan.captain).name), " as captain. "] : "The plan appears within 12 hours. ",
          link("gameweek", "See the plan"), ".")
        : h("p", {}, "The season is over."),
      h("p", {}, scored.length
        ? ["Season so far: ", h("b", {}, `${ours} points`), `, ${fmt.signed(ours - avg)} against the average manager. `, link("team", "The model team"), " · ", link("benchmarks", "how it stacks up"), "."]
        : ["The model team starts at the GW", s.next.gw, " deadline with £100m. ", link("benchmarks", "Benchmarks"), " follow it against managers at p50, p90 and p99, and against OpenFPL."]),
      h("p", {}, "In the season-replay backtest (", bt.seasons, ") it beat the OpenFPL replica by ", h("b", {}, `${fmt.signed(bt.replay_vs_openfpl.per_gw, 2)} points a gameweek`), ". ", link("lab", "Every experiment"), " is in the lab, kept or not."),
      isAdmin()
        ? h("p", {}, "System: ", h("span", { class: `chip ${st.tone}` }, st.text), " ", link("health", "Health"), " · ", link("data", "Data"), " · ", link("access", `Access${pending ? ` (${pending} waiting)` : ""}`))
        : null,
      h("p", { class: "muted" }, `Snapshot ${fmt.ago(s.generated_at)}. Refreshes every three hours.`, h("span", { class: "blink" }, " _")));
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
    if (WM.wins.some((w) => w.id === "health")) render();
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
            h("div", { class: "avatar" }, (r.name || r.email || "?").replace(/^Example:\s*/, "").trim()[0].toUpperCase()),
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

  // ---------- desktop (built once: stars, monument, clouds) ----------
  const Desk = { ui: null };
  function buildDesktop() {
    if (Desk.ui) return;
    const stars = h("canvas", { class: "stars", "aria-hidden": "true" });
    const monument = h("div", { class: "monument", "aria-hidden": "true" },
      h("div", { class: "portal" }), h("div", { class: "pillar" }), h("div", { class: "reactor" }, h("div", { class: "halo" }), trefoil("", true)));
    const clouds = h("div", { class: "clouds", "aria-hidden": "true" },
      [[8, -30, 26, 90], [22, -40, 30, 110], [38, -26, 26, 90], [52, -38, 30, 110], [66, -30, 26, 90], [30, -10, 16, 50], [58, -12, 14, 46]].map(([l, b, w, ht], i) =>
        h("i", { style: { left: `${l}%`, bottom: `${b}%`, width: `${w}%`, height: `${ht}%`, animationDelay: `${-i * 2.7}s` } })));
    Desk.ui = h("div", { class: "ui" });
    root().append(h("div", { class: "desktop" }, stars, monument, clouds, Desk.ui), h("div", { class: "scanlines", "aria-hidden": "true" }));
    starfield(stars);
  }

  function starfield(canvas) {
    const g = canvas.getContext("2d");
    if (!g) return;
    let pts = [];
    const draw = () => {
      g.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of pts) { g.fillStyle = `rgba(255,255,255,${p.a})`; g.fillRect(p.x, p.y, p.s, p.s); }
    };
    const size = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      let seed = 7;
      const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const n = Math.round((canvas.width * canvas.height) / (5200 * dpr * dpr));
      pts = Array.from({ length: n }, () => ({ x: rnd() * canvas.width, y: rnd() * canvas.height, s: (rnd() < 0.85 ? 1 : 2) * dpr, a: 0.25 + rnd() * 0.75 }));
      draw();
    };
    size();
    window.addEventListener("resize", size);
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches)
      setInterval(() => { for (let i = 0; i < 6 && pts.length; i++) { const p = pts[Math.floor(Math.random() * pts.length)]; p.a = 0.2 + Math.random() * 0.8; } draw(); }, 160);
  }

  // ---------- window manager ----------
  const WM = { wins: [], z: 20, active: null, start: false, scroll: {}, sel: null };
  const deskSize = () => { const d = $(".desktop"); return { w: d.clientWidth, h: d.clientHeight }; };
  const small = () => window.innerWidth <= 720;

  function place(id) {
    if (WM.wins.some((w) => w.id === id)) return;
    const { w: dw, h: dh } = deskSize();
    const home = id === "home";
    const ww = Math.min(home ? 600 : 960, dw - 24);
    const wh = home ? Math.min(470, dh - Math.round(dh * 0.3) - 16) : Math.min(680, dh - 16);
    const k = WM.wins.length % 6;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
    WM.wins.push({
      id, w: ww, h: wh, min: false, max: false, z: ++WM.z, fresh: true,
      x: clamp((dw - ww) / 2 + (home ? 0 : 40 + k * 26), 0, Math.max(0, dw - ww)),
      y: clamp(home ? Math.round(dh * 0.3) : (dh - wh) / 2 - 10 + k * 22, 0, Math.max(0, dh - wh)),
    });
  }
  function setHash(id) {
    try { history.replaceState(null, "", id === "home" ? location.pathname + location.search : `#${id}`); } catch { /* sandboxed */ }
  }
  function openWin(id) {
    if (!appById(id) || (appById(id).admin && !isAdmin())) return;
    place(id);
    const w = WM.wins.find((x) => x.id === id);
    w.min = false;
    w.z = ++WM.z;
    WM.active = id;
    WM.start = false;
    setHash(id);
    render();
  }
  function topWin() {
    const open = WM.wins.filter((w) => !w.min).sort((a, b) => b.z - a.z);
    return open.length ? open[0].id : null;
  }
  function closeWin(id) {
    WM.wins = WM.wins.filter((w) => w.id !== id);
    delete WM.scroll[id];
    WM.active = topWin();
    setHash(WM.active || "home");
    render();
  }
  function minWin(id) {
    const w = WM.wins.find((x) => x.id === id);
    if (w) w.min = true;
    WM.active = topWin();
    render();
  }
  function maxWin(id) {
    const w = WM.wins.find((x) => x.id === id);
    if (w) w.max = !w.max;
    render();
  }
  function focusWin(id) {
    if (WM.active === id) return;
    const w = WM.wins.find((x) => x.id === id);
    if (!w) return;
    w.z = ++WM.z;
    WM.active = id;
    setHash(id);
    document.querySelectorAll(".win[data-win]").forEach((el) => {
      const on = el.dataset.win === id;
      el.classList.toggle("active", on);
      if (on) el.style.zIndex = w.z;
    });
    document.querySelectorAll(".tb-wins .tb-btn").forEach((b) => b.classList.toggle("on", b.dataset.win === id));
  }
  function startDrag(e, w) {
    if (e.button !== 0 || e.target.closest(".wbtn") || w.max || small()) return;
    const el = e.currentTarget.parentElement;
    const { w: dw, h: dh } = deskSize();
    const ox = e.clientX - w.x, oy = e.clientY - w.y;
    const move = (ev) => {
      w.x = Math.min(Math.max(ev.clientX - ox, 80 - w.w), dw - 80);
      w.y = Math.min(Math.max(ev.clientY - oy, 0), dh - 28);
      el.style.left = `${w.x}px`;
      el.style.top = `${w.y}px`;
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    e.preventDefault();
  }

  const GLYPH = {
    min: '<svg viewBox="0 0 10 10"><rect x="1" y="7" width="6" height="2" fill="#000"/></svg>',
    max: '<svg viewBox="0 0 10 10"><rect x="0.5" y="0.5" width="9" height="8" fill="none" stroke="#000"/><rect x="0.5" y="0.5" width="9" height="2" fill="#000"/></svg>',
    restore: '<svg viewBox="0 0 10 10"><rect x="2.5" y="0.5" width="7" height="6" fill="none" stroke="#000"/><rect x="0.5" y="3.5" width="7" height="6" fill="#c0c0c0" stroke="#000"/><rect x="0.5" y="3.5" width="7" height="1.5" fill="#000"/></svg>',
    close: '<svg viewBox="0 0 10 10"><path d="M1 1l8 8M9 1l-8 8" stroke="#000" stroke-width="1.8"/></svg>',
  };
  const glyph = (k) => { const s = h("span", { "aria-hidden": "true" }); s.innerHTML = GLYPH[k]; return s.firstChild; };

  function winEl(w) {
    const app = appById(w.id);
    let content;
    try { content = [app.view(app)].flat(); }
    catch (e) { console.error(e); content = [empty("x", "This window could not be drawn", e.message)]; }
    const s = state.snap;
    const el = h("section", {
      class: `win${w.max ? " max" : ""}${WM.active === w.id ? " active" : ""}${w.fresh ? " opening" : ""}`,
      "data-win": w.id, role: "dialog", "aria-label": app.name,
      style: { left: `${w.x}px`, top: `${w.y}px`, width: `${w.w}px`, height: `${w.h}px`, zIndex: w.z },
      onpointerdown: () => focusWin(w.id),
    },
      h("div", { class: "titlebar", onpointerdown: (e) => startDrag(e, w), ondblclick: () => maxWin(w.id) },
        pimg(app.pix || app.id), h("span", { class: "ttl" }, `C:/statsnuke/${w.id}`),
        h("button", { class: "wbtn", "aria-label": "Minimise", onclick: () => minWin(w.id) }, glyph("min")),
        h("button", { class: "wbtn maxb", "aria-label": w.max ? "Restore" : "Maximise", onclick: () => maxWin(w.id) }, glyph(w.max ? "restore" : "max")),
        h("button", { class: "wbtn close", "aria-label": "Close", onclick: () => closeWin(w.id) }, glyph("close"))),
      h("div", { class: "win-body" }, h("div", { class: "screen", "data-screen": w.id }, content)),
      h("div", { class: "statusbar" },
        h("span", {}, `Snapshot ${fmt.ago(s.generated_at)}`),
        h("span", {}, s.next.gw ? `GW${s.next.gw} in ${fmt.until(s.next.deadline)}` : "Season over"),
        h("span", {}, state.me.role === "admin" ? "Owner" : "Friend")));
    w.fresh = false;
    if (window.ResizeObserver) {
      new ResizeObserver(() => { if (!w.max && !small() && el.isConnected && el.offsetWidth) { w.w = el.offsetWidth; w.h = el.offsetHeight; } }).observe(el);
    }
    return el;
  }

  // ---------- desktop pieces ----------
  function dicon(a, i) {
    const badge = a.id === "access" ? state.requests.filter((r) => r.role === "pending").length : 0;
    return h("button", {
      class: `dicon pop${WM.sel === a.id ? " sel" : ""}`, style: { animationDelay: `${0.04 * i}s` }, title: a.name,
      onclick: () => { WM.sel = a.id; openWin(a.id); },
    }, pimg(a.pix || a.id), badge ? h("span", { class: "badge" }, badge) : null, a.admin ? pimg("lock", "lock") : null, h("span", { class: "lbl" }, a.name));
  }

  function clockText() {
    return new Date().toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  }

  function taskbar(locked) {
    const start = h("button", {
      class: `tb-btn start${WM.start ? " on" : ""}`, "aria-haspopup": "menu", "aria-expanded": String(WM.start), disabled: locked || null,
      onclick: (e) => { e.stopPropagation(); WM.start = !WM.start; render(); },
    }, trefoil(""), "Start");
    const wins = h("div", { class: "tb-wins" }, locked ? [] : WM.wins.map((w) => {
      const app = appById(w.id);
      return h("button", {
        class: `tb-btn${WM.active === w.id && !w.min ? " on" : ""}`, "data-win": w.id,
        onclick: () => (WM.active === w.id && !w.min ? minWin(w.id) : openWin(w.id)),
      }, pimg(app.pix || app.id), h("span", {}, app.name));
    }));
    let status = null;
    if (!locked && state.snap) {
      if (isAdmin()) {
        const st = systemStatus();
        const color = { ok: "#22b14c", warn: "#ffb02e", fail: "#e02020" }[st.tone];
        status = h("a", { href: "#data", title: st.text, onclick: (e) => { e.preventDefault(); openWin("data"); } }, h("span", { class: "led", style: { background: color } }), h("span", { class: "stat-text" }, st.text));
      } else {
        status = h("span", { class: "stat-text" }, `Updated ${fmt.ago(state.snap.generated_at)}`);
      }
    }
    return h("div", { class: "taskbar" }, start, wins, h("div", { class: "tray" }, status, h("span", { class: "clock" }, clockText())));
  }

  function startMenu() {
    const item = (a) => h("button", { role: "menuitem", onclick: () => openWin(a.id) }, pimg(a.pix || a.id), a.name);
    return h("div", { class: "start-menu", role: "menu", onclick: (e) => e.stopPropagation() },
      h("div", { class: "start-banner" }, h("b", {}, "Stats"), "Nuke"),
      h("div", { class: "start-items" },
        item(HOME), visibleApps().filter((a) => !a.admin).map(item),
        isAdmin() ? [h("hr"), visibleApps().filter((a) => a.admin).map(item)] : null,
        h("hr"),
        h("div", { class: "who" }, `Signed in as ${state.me.name || state.me.email || "you"}`, h("br"), state.me.role === "admin" ? "Owner" : "Friend"),
        h("button", { role: "menuitem", onclick: signOut }, pimg("lock"), "Shut down (sign out)…")));
  }

  async function signOut() {
    await state.api.signOut();
    state.me = null;
    state.snap = null;
    WM.wins = [];
    WM.active = null;
    WM.start = false;
    render();
  }

  // ---------- dialogs: sign in, request access, waiting ----------
  function dialog(title, pix, body, onClose) {
    return h("section", { class: "win dialog active", role: "dialog", "aria-label": title },
      h("div", { class: "titlebar" }, pimg(pix), h("span", { class: "ttl" }, title),
        onClose ? h("button", { class: "wbtn close", "aria-label": "Close", onclick: onClose }, glyph("close")) : null),
      h("div", { class: "win-body" }, body));
  }

  const gateUi = { tab: "signin", busy: false };
  function viewGate() {
    const u = gateUi;
    const api = state.api;
    if (api.mode === "none")
      return dialog("Stats Nuke", "data", h("div", { class: "dlg-grid" }, pimg("data"), h("div", { class: "form" },
        h("p", {}, "This site isn't connected to its database yet."),
        h("p", { class: "dlg-note" }, "The owner adds the Supabase URL and public key as repository variables and redeploys (web/README.md)."))));
    if (u.tab === "confirm")
      return dialog("Confirm your email", "wait", h("div", { class: "dlg-grid" }, pimg("wait"), h("div", { class: "form" },
        h("p", {}, "We sent you a link. Open it, then sign in here. Your request reaches the owner once your email is confirmed."),
        h("div", { class: "dlg-actions" }, h("button", { class: "btn primary", onclick: () => { u.tab = "signin"; render(); } }, "OK")))));
    const friend = u.tab === "friend";
    const field = (id, label, type, extra) => h("div", { class: `field${type === "textarea" ? " top" : ""}` }, h("label", { for: id }, label),
      h(type === "textarea" ? "textarea" : "input", { class: "input", id, name: id, type: type === "textarea" ? null : type, ...extra }));
    const err = h("div", { class: "err", role: "alert" });
    const submit = async (e) => {
      e.preventDefault();
      if (u.busy) return;
      const f = e.target;
      u.busy = true;
      err.textContent = "";
      try {
        if (!friend) await api.signIn(f.email.value.trim(), f.password.value);
        else {
          if (!f.name.value.trim()) throw new Error("Please enter your name.");
          if (f.password.value.length < 8) throw new Error("Use a password of at least 8 characters.");
          const r = await api.signUp(f.name.value.trim(), f.email.value.trim(), f.password.value, f.note.value.trim());
          if (r.confirm) { u.busy = false; u.tab = "confirm"; render(); return; }
        }
        u.busy = false;
        await boot(true);
      } catch (ex) {
        u.busy = false;
        err.textContent = ex.message || "Something went wrong.";
      }
    };
    const form = h("form", { class: "form", onsubmit: submit, novalidate: true },
      h("p", {}, friend ? "Ask for access. The owner approves each request; you can sign in as soon as you're approved." : "Type your email and password to log on to Stats Nuke."),
      friend ? field("name", "Your name:", "text", { required: true, autocomplete: "name", maxlength: 80 }) : null,
      field("email", "Email:", "email", { required: true, autocomplete: "email" }),
      field("password", "Password:", "password", { required: true, autocomplete: friend ? "new-password" : "current-password", minlength: 8 }),
      friend ? field("note", "Note:", "textarea", { maxlength: 280, placeholder: "How you know the owner, your league" }) : null,
      err,
      h("div", { class: "dlg-actions" },
        h("button", { class: "btn primary", type: "submit" }, friend ? "Send request" : "OK"),
        h("button", { class: "btn", type: "button", onclick: () => { u.tab = friend ? "signin" : "friend"; render(); } }, friend ? "Cancel" : "I'm a friend…")),
      api.mode === "preview" ? h("p", { class: "dlg-note" }, "Preview: any email and password signs you in as the owner.") : null);
    return dialog(friend ? "Request access" : "Welcome to Stats Nuke", friend ? "players" : "access", h("div", { class: "dlg-grid" }, pimg(friend ? "players" : "access"), form));
  }

  function viewWaiting() {
    const declined = state.me.role === "rejected";
    const api = state.api;
    return dialog(declined ? "Access declined" : "Request sent", declined ? "lock" : "wait", h("div", { class: "dlg-grid" }, pimg(declined ? "lock" : "wait"), h("div", { class: "form" },
      h("p", {}, declined ? "The owner declined your request. Ask them if you think this is a mistake." : `Thanks${state.me.name ? `, ${state.me.name}` : ""}. You'll get in as soon as the owner approves. This window checks every 20 seconds.`),
      declined ? null : h("div", { class: "marquee", "aria-hidden": "true" }, h("i")),
      h("div", { class: "dlg-actions" },
        api.mode === "preview" && !declined ? h("button", { class: "btn primary", onclick: async () => { api.simulateApproval(); await boot(true); } }, "Preview: approve me") : null,
        h("button", { class: "btn", onclick: signOut }, "Sign out")))));
  }

  // ---------- render / boot ----------
  function render() {
    buildDesktop();
    const ui = Desk.ui;
    ui.querySelectorAll("[data-screen]").forEach((el) => (WM.scroll[el.dataset.screen] = el.scrollTop));
    ui.replaceChildren();
    clearInterval(render.poll);
    if (!state.me) { ui.append(viewGate(), taskbar(true)); return; }
    if (state.me.role === "pending" || state.me.role === "rejected") {
      ui.append(viewWaiting(), taskbar(true));
      if (state.me.role === "pending")
        render.poll = setInterval(async () => { const me = await state.api.me(); if (me && me.role !== "pending") boot(true); }, 20000);
      return;
    }
    if (!state.snap) {
      ui.append(dialog("Stats Nuke", "data", h("div", { class: "dlg-grid" }, pimg("data"), h("div", { class: "form" },
        h("p", {}, "No data published yet. The Live workflow uploads the first snapshot on its next run."),
        h("div", { class: "dlg-actions" }, h("button", { class: "btn", onclick: signOut }, "Sign out"))))), taskbar(true));
      return;
    }
    const apps = visibleApps();
    const left = [HOME, ...apps.filter((a) => !a.admin).slice(0, 5)];
    const right = apps.filter((a) => !left.includes(a));
    ui.append(
      h("div", { class: "icon-wrap" }, h("div", { class: "icons left" }, left.map(dicon)), h("div", { class: "icons right" }, right.map((a, i) => dicon(a, i + left.length)))),
      ...WM.wins.filter((w) => !w.min).map(winEl),
      taskbar(false),
      ...(WM.start ? [startMenu()] : []),
    );
    ui.querySelectorAll("[data-screen]").forEach((el) => { if (WM.scroll[el.dataset.screen]) el.scrollTop = WM.scroll[el.dataset.screen]; });
  }

  async function boot(quiet) {
    if (!quiet) Loader.step("Connecting to the server", 0.25);
    try {
      state.me = await state.api.me();
      if (!quiet) Loader.step("Verifying access", 0.5);
      if (state.me && (state.me.role === "admin" || state.me.role === "friend")) {
        if (!quiet) Loader.step("Loading the latest snapshot", 0.75);
        state.snap = await state.api.snapshot();
        state.pmap = null;
        if (state.me.role === "admin") state.requests = await state.api.requests();
        if (state.snap && !WM.wins.length) {
          buildDesktop();
          if (!small()) { place("home"); WM.active = "home"; }
          const want = location.hash.replace(/^#/, "");
          if (want && want !== "home" && appById(want) && (!appById(want).admin || isAdmin())) { place(want); WM.active = want; }
        }
      }
    } catch (e) {
      console.error(e);
      toast(`Could not reach the server: ${e.message}`);
    }
    render();
  }

  async function main() {
    Loader.mount();
    if (CFG.supabaseUrl && CFG.supabaseAnonKey && window.supabase) state.api = supabaseBackend();
    else if (EMBEDDED) state.api = previewBackend();
    else state.api = { mode: "none", async me() { return null; } };
    buildDesktop();
    await boot(false);
    await Loader.done();
    document.addEventListener("click", () => { if (WM.start) { WM.start = false; render(); } });
    window.addEventListener("hashchange", () => { const id = location.hash.replace(/^#/, ""); if (id && state.snap) openWin(id); });
    setInterval(() => document.querySelectorAll(".clock").forEach((c) => (c.textContent = clockText())), 15000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
  else main();
})();
