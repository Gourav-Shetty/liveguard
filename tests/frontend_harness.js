"use strict";
/*
 * Headless frontend harness for LiveGuard test_viewer.html (part of the test
 * suite; runs OUTSIDE the browser - no DOM needed).
 *
 * Runs the page's REAL inline <script> in a Node `vm` context with:
 *   - a mock DOM (elements, classList, listeners, localStorage)
 *   - a recording WebSocket subclassing Node's built-in (undici) WebSocket
 * so the actual client code talks to a real TelemetryServer over the wire.
 *
 * Usage:
 *   1. start a TelemetryServer (any free port), e.g.:
 *        LIVEGUARD_DATA_DIR=%TEMP%\\lg_manual python -c ^
 *          "from backend.telemetry_server import TelemetryServer; \\
 *           TelemetryServer(host='127.0.0.1', port=8766).start(); import time; time.sleep(3600)"
 *   2. node tests/frontend_harness.js ws://127.0.0.1:8766
 *
 * Covered: register -> auto sign-in -> live broadcasts -> logout -> wrong
 * password -> re-login, plus stored-token auto-auth on a fresh page load,
 * post-auth bad_message resilience (real server round-trip), password
 * hygiene, the badge write guard, and the dropped-connection auto-reconnect
 * (attempt badge + jittered first backoff + resumed broadcasts).
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const WS_URL = process.argv[2];
if (!WS_URL) {
  console.error("usage: node tests/frontend_harness.js ws://127.0.0.1:<port>");
  process.exit(2);
}

const HTML_PATH = path.join(__dirname, "..", "frontend", "test_viewer.html");
const HTML = fs.readFileSync(HTML_PATH, "utf8");
const m = HTML.match(/<script>([\s\S]+)<\/script>/);
if (!m) throw new Error("inline <script> not found in test_viewer.html");
const SRC = m[1];

// Fail fast with a readable diagnostic: the harness subclasses Node's
// built-in WebSocket, which exists by default only from Node 21.
if (typeof globalThis.WebSocket !== "function") {
  console.error(
    "HARNESS ERROR: globalThis.WebSocket is not available (Node " +
      process.version +
      "). Run this harness on Node >= 21 (CI uses Node 22)."
  );
  process.exit(2);
}

/* ------------------------------------------------------------------ */
/* result bookkeeping                                                  */
/* ------------------------------------------------------------------ */
const results = [];
function check(name, cond, detail) {
  results.push({ name, ok: !!cond, detail: cond ? "" : String(detail == null ? "" : detail) });
  console.log((cond ? "ok   " : "FAIL ") + name + (cond ? "" : "  -> got: " + detail));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, ms, label) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (fn()) return true;
    await sleep(40);
  }
  throw new Error("timed out after " + ms + "ms waiting for: " + label);
}

/* ------------------------------------------------------------------ */
/* page boot: mock DOM + recording WebSocket + real network            */
/* ------------------------------------------------------------------ */
const INITIAL_HIDDEN = { userBar: true, authOverlay: true, authError: true, authSpinner: true };

function boot(seedStorage) {
  const elements = {};
  const ctxStub = new Proxy({}, { get: () => () => {}, set: () => true });
  const KNOWN_IDS = [
    "ecgCanvas", "authOverlay", "authUser", "authPass", "authError", "authStatus",
    "authSubmit", "authSpinner", "tabSignin", "tabRegister", "authTag", "btnRetry",
    "btnConnect", "btnLogout", "statusDot", "authDot", "userBar", "userName",
    "diagBadge", "diagTitle", "diagSub", "valHr", "valAlerts", "wsUrl", "authForm",
  ];

  function makeElement(id) {
    const listeners = {};
    const classes = new Set();
    // className writes are counted so checks can prove the viewer's
    // applied-value guards: an unchanged badge must NOT be rewritten.
    let classNameValue = "";
    let classNameWrites = 0;
    const elm = {
      id,
      hidden: !!INITIAL_HIDDEN[id],
      value: "",
      innerText: id === "authStatus" ? "Not connected" : "",
      innerHTML: id === "valHr" ? '-- <span class="unit">BPM</span>' : "",
      disabled: false,
      width: 0,
      height: 0,
      parentElement: { clientWidth: 900 },
      classList: {
        add: (...c) => c.forEach((x) => classes.add(x)),
        remove: (...c) => c.forEach((x) => classes.delete(x)),
        toggle: (c, force) => {
          const on = force === undefined ? !classes.has(c) : !!force;
          if (on) classes.add(c); else classes.delete(c);
          return on;
        },
        contains: (c) => classes.has(c),
      },
      setAttribute() {},
      getAttribute() { return null; },
      addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
      removeEventListener() {},
      focus() {},
      getContext() { return ctxStub; },
      _fire(type, ev) { (listeners[type] || []).forEach((fn) => fn(ev || { preventDefault() {} })); },
    };
    Object.defineProperty(elm, "className", {
      get() { return classNameValue; },
      set(v) { classNameValue = String(v); classNameWrites += 1; },
      enumerable: true,
      configurable: true,
    });
    elm.classNameWrites = () => classNameWrites;
    return elm;
  }
  KNOWN_IDS.forEach((id) => { elements[id] = makeElement(id); });

  const SENT = [];
  class RecordingWebSocket extends globalThis.WebSocket {
    constructor(url) {
      super(url);
      this.__url = String(url);
    }
    send(data) {
      SENT.push(String(data));
      return super.send(data);
    }
  }

  const storage = new Map(Object.entries(seedStorage || {}));
  const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
    removeItem: (k) => storage.delete(k),
    clear: () => storage.clear(),
  };

  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    localStorage,
    WebSocket: RecordingWebSocket,
    document: {
      getElementById: (id) => (elements[id] = elements[id] || makeElement(id)),
    },
    window: { addEventListener() {} },
    requestAnimationFrame() {},
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(SRC, context, { filename: "test_viewer.html#inline-script" });
  return { context, elements, SENT, storage };
}

function frames(sent) {
  return sent.map((s, i) => {
    try { return { i, obj: JSON.parse(s) }; }
    catch (e) { return { i, obj: { __unparsable: s } }; }
  });
}
function lastFrame(sent) {
  return JSON.parse(sent[sent.length - 1]);
}
function keySet(obj) {
  return Object.keys(obj).sort().join(",");
}

/* ================================================================== */
/* Scenario 1: register -> auto sign-in -> live stream -> logout ->    */
/*             wrong password -> re-login                              */
/* ================================================================== */
(async function scenario1() {
  const username = "frontuser" + (Date.now() % 100000);
  const password = "front-end-pass1";
  const s = boot({});
  s.elements.wsUrl.value = WS_URL;

  // --- register tab + submit the login/register form -----------------
  s.elements.tabRegister._fire("click");
  check("register tab switches submit label",
    s.elements.authSubmit.innerText === "Create account", s.elements.authSubmit.innerText);

  s.elements.authUser.value = username;
  s.elements.authPass.value = password;
  s.elements.authForm._fire("submit", { preventDefault() {} });

  await waitFor(() => s.SENT.length >= 1, 6000, "register frame");
  const f0 = JSON.parse(s.SENT[0]);
  check("client frame 1 == {type:register,username,password}",
    f0.type === "register" && f0.username === username && f0.password === password && keySet(f0) === "password,type,username",
    s.SENT[0]);

  await waitFor(() => s.SENT.length >= 2, 8000, "auto sign-in frame");
  const f1 = JSON.parse(s.SENT[1]);
  check("client frame 2 == {type:auth,username,password} (auto after register_ok)",
    f1.type === "auth" && f1.username === username && f1.password === password && !("token" in f1) && keySet(f1) === "password,type,username",
    s.SENT[1]);

  await waitFor(() => s.elements.authOverlay.hidden === true, 8000, "overlay hidden after auth_ok");
  const token1 = s.storage.get("liveguard_token");
  check("auth_ok stored token in localStorage(liveguard_token)",
    typeof token1 === "string" && token1.split(".").length === 2, token1);
  check("user bar visible after sign-in", s.elements.userBar.hidden === false, s.elements.userBar.hidden);
  check("user name shown", s.elements.userName.innerText === username, s.elements.userName.innerText);
  check("no auth error shown", s.elements.authError.hidden === true, s.elements.authError.innerText);
  check("password input is empty after auth_ok",
    s.elements.authPass.value === "", JSON.stringify(s.elements.authPass.value));

  // --- live broadcasts update the UI --------------------------------
  await waitFor(() => !/^--/.test(s.elements.valHr.innerHTML), 6000, "first beat badge update");
  const hr1 = s.elements.valHr.innerHTML;
  await sleep(1900);
  const hr2 = s.elements.valHr.innerHTML;
  check("HR badge keeps updating from broadcasts", hr1 !== hr2 && !/^--/.test(hr2), hr1 + " | " + hr2);
  check("diagnosis badge shows NORMAL BEAT",
    /NORMAL BEAT/.test(s.elements.diagTitle.innerText), s.elements.diagTitle.innerText);

  // --- logout (client closes the socket; nothing must arrive after) --
  const framesBeforeLogout = s.SENT.length;
  s.context.logout();
  await waitFor(() => s.SENT.length > framesBeforeLogout, 4000, "logout frame");
  const lo = lastFrame(s.SENT);
  check("logout frame == {type:logout}",
    lo.type === "logout" && keySet(lo) === "type", JSON.stringify(lo));
  await sleep(700);
  check("overlay visible after logout", s.elements.authOverlay.hidden === false, s.elements.authOverlay.hidden);
  check("token cleared after logout", !s.storage.has("liveguard_token"), s.storage.get("liveguard_token"));
  check("user bar hidden after logout", s.elements.userBar.hidden === true, s.elements.userBar.hidden);
  check("status says 'Signed out'", s.elements.authStatus.innerText === "Signed out", s.elements.authStatus.innerText);
  const hrAfter = s.elements.valHr.innerHTML;
  await sleep(1900);
  check("no stream updates after logout", s.elements.valHr.innerHTML === hrAfter,
    hrAfter + " | " + s.elements.valHr.innerHTML);
  const afterLo = frames(s.SENT.slice(framesBeforeLogout + 1));
  check("nothing sent after logout frame",
    s.SENT.length === framesBeforeLogout + 1, JSON.stringify(afterLo));

  // --- a pre-auth server error must STILL surface in the overlay -------
  // (the socket is gone, so the server cannot deliver this itself: invoke
  // the script's own message handler exactly as handleMessage would be
  // called by ws.onmessage with an event carrying the JSON payload.)
  s.context.handleMessage({
    data: JSON.stringify({ type: "error", code: "register_failed", reason: "pre-auth server error probe" }),
  });
  check("pre-auth server error still shows in the auth overlay",
    s.elements.authOverlay.hidden === false &&
      /pre-auth server error probe/.test(s.elements.authError.innerText),
    "overlay.hidden=" + s.elements.authOverlay.hidden + " authError=" + s.elements.authError.innerText);

  // --- wrong password (single attempt) surfaces the server reason ---
  s.elements.authUser.value = username;   // logout() cleared the username field
  s.elements.authPass.value = "totally-wrong-pw";
  s.elements.authForm._fire("submit", { preventDefault() {} });
  await waitFor(() => /invalid credentials/i.test(s.elements.authError.innerText), 8000,
    "auth_error reason shown");
  check("wrong password shows server reason 'invalid credentials'",
    /invalid credentials/i.test(s.elements.authError.innerText), s.elements.authError.innerText);
  check("wrong password leaves the stored token cleared", !s.storage.has("liveguard_token"),
    s.storage.get("liveguard_token"));

  // --- correct password re-login ------------------------------------
  s.elements.authUser.value = username;
  s.elements.authPass.value = password;
  s.elements.authForm._fire("submit", { preventDefault() {} });
  await waitFor(() => s.elements.authOverlay.hidden === true, 8000, "re-login overlay hidden");
  const rel = lastFrame(s.SENT);
  check("re-login frame == {type:auth,username,password}",
    rel.type === "auth" && rel.username === username && rel.password === password && keySet(rel) === "password,type,username",
    JSON.stringify(rel));
  check("token restored after re-login", !!s.storage.get("liveguard_token"), s.storage.get("liveguard_token"));
  const tokenFinal = s.storage.get("liveguard_token");
  await sleep(1900);
  check("re-login resumed live updates", !/^--/.test(s.elements.valHr.innerHTML) &&
    s.elements.valHr.innerHTML !== hrAfter,
    hrAfter + " | " + s.elements.valHr.innerHTML);

  // --- all outbound frames conform to the wire protocol -------------
  const all = frames(s.SENT);
  const shapesOk = all.every(({ obj }) =>
    obj && typeof obj.type === "string" &&
    ["register", "auth", "logout"].includes(obj.type));
  check("every outbound frame is register/auth/logout", shapesOk,
    JSON.stringify(all.map((f) => f.obj)));

  // --- a REAL post-auth bad_message reply must not hijack the session --
  // The backend answers any unexpected client frame with
  // {type:"error", code:"bad_message"} (see tests/test_auth_flow.py
  // test_09a) and keeps the socket open, so we probe the genuine server
  // path: wrap ws.onmessage to capture frames exactly as the socket
  // delivers them, send {"type":"ping"} through the recording socket, and
  // assert the viewer leaves the live dashboard alone when the reply hits.
  vm.runInContext(
    "window.__raw = []; window.__prevOnMessage = ws.onmessage;" +
      "ws.onmessage = function (ev) { window.__raw.push(String(ev.data)); window.__prevOnMessage(ev); };",
    s.context);
  vm.runInContext('ws.send(JSON.stringify({ type: "ping" }));', s.context);
  const sawBadMessage = () =>
    (s.context.window.__raw || []).some((raw) => {
      try {
        const o = JSON.parse(raw);
        return o && o.type === "error" && o.code === "bad_message";
      } catch (e) { return false; }
    });
  await waitFor(sawBadMessage, 6000, "bad_message reply for the junk frame");
  check("server replied to the junk frame with bad_message", sawBadMessage(),
    JSON.stringify(s.context.window.__raw.slice(-3)));
  check("post-auth server error keeps the auth overlay hidden",
    s.elements.authOverlay.hidden === true, s.elements.authOverlay.hidden);
  check("post-auth server error leaves the dashboard untouched",
    s.elements.userBar.hidden === false && s.elements.authError.hidden === true,
    "userBar.hidden=" + s.elements.userBar.hidden + " authError.hidden=" + s.elements.authError.hidden);
  const hrBeforeErr = s.elements.valHr.innerHTML;
  await sleep(1300);
  check("live stream keeps updating after a post-auth server error",
    s.elements.valHr.innerHTML !== hrBeforeErr,
    hrBeforeErr + " | " + s.elements.valHr.innerHTML);

  global.__TOKEN__ = tokenFinal;
  global.__USER__ = username;
})()
  .then(() => scenario2(global.__TOKEN__, global.__USER__))
  .then((ctx) => scenario3(ctx.token, ctx.username))
  .catch((e) => {
    console.error("HARNESS ERROR: " + (e && e.stack ? e.stack : e));
    summary(1);
  });

/* ================================================================== */
/* Scenario 2: fresh page load with a stored token -> auto auth        */
/* ================================================================== */
async function scenario2(token, username) {
  const s = boot({ liveguard_token: token, liveguard_user: username });
  s.elements.wsUrl.value = WS_URL;

  s.context.toggleConnect(); // same code path as the Connect button
  await waitFor(() => s.SENT.length >= 1, 6000, "token auth frame");
  const f0 = JSON.parse(s.SENT[0]);
  check("stored-token connect sends {type:auth,token} only",
    f0.type === "auth" && f0.token === token && keySet(f0) === "token,type",
    s.SENT[0]);

  await waitFor(() => s.elements.authOverlay.hidden === true, 8000, "token session restored");
  check("token session restores user bar", s.elements.userName.innerText === username,
    s.elements.userName.innerText);
  check("token preserved in storage", s.storage.get("liveguard_token") === token,
    s.storage.get("liveguard_token"));
  await waitFor(() => !/^--/.test(s.elements.valHr.innerHTML), 6000, "stream after token auth");
  check("token session receives broadcasts", !/^--/.test(s.elements.valHr.innerHTML),
    s.elements.valHr.innerHTML);

  // --- badge-write guard: identical badge state must not be rewritten --
  // The feeder emits the same normal beat (is_anomaly=false, confidence
  // 98.5) over and over; after the first flush no className write may
  // happen. The HR check underneath proves beats kept flowing meanwhile.
  await waitFor(() => /NORMAL BEAT/.test(s.elements.diagTitle.innerText), 6000,
    "first flushed NORMAL beat badge");
  const badgeWritesBefore = s.elements.diagBadge.classNameWrites();
  const guardHr = s.elements.valHr.innerHTML;
  await sleep(2100);
  const badgeWritesAfter = s.elements.diagBadge.classNameWrites();
  check("badge className not rewritten while badge state is unchanged",
    badgeWritesBefore > 0 && badgeWritesAfter === badgeWritesBefore,
    badgeWritesBefore + " -> " + badgeWritesAfter);
  check("beats kept flowing during the badge-guard window",
    s.elements.valHr.innerHTML !== guardHr &&
      /NORMAL BEAT/.test(s.elements.diagTitle.innerText),
    guardHr + " | " + s.elements.valHr.innerHTML +
      " | badge=" + s.elements.diagTitle.innerText);

  return { token, username };
}

/* ================================================================== */
/* Scenario 3: unexpected drop of a live session -> CONNECTION LOST    */
/*             badge -> jittered backoff -> broadcasts resume          */
/* ================================================================== */
async function scenario3(token, username) {
  const s = boot({ liveguard_token: token, liveguard_user: username });
  s.elements.wsUrl.value = WS_URL;

  s.context.toggleConnect();
  await waitFor(() => s.elements.authOverlay.hidden === true, 8000, "scenario3: auth_ok");
  await waitFor(() => !/^--/.test(s.elements.valHr.innerHTML), 6000, "scenario3: live stream");

  // Unexpected close of the healthy authenticated socket (no closingReason:
  // exactly what handleClose sees when the server or network drops us).
  vm.runInContext("ws.close()", s.context);
  await waitFor(() => /CONNECTION LOST/.test(s.elements.diagTitle.innerText), 4000,
    "scenario3: connection-lost badge");
  check("unexpected drop shows the connection-lost attempt badge",
    /attempt 1\/5/.test(s.elements.diagSub.innerText), s.elements.diagSub.innerText);

  // The socket is fully closed now, so the HR badge is frozen: whatever it
  // shows can only change again once broadcasts resume after the backoff.
  const hrAtDrop = s.elements.valHr.innerHTML;
  const tDrop = Date.now();

  // First backoff: 2s +/-20% jitter (1.6s..2.4s) + token re-auth.
  await waitFor(() => !/CONNECTION LOST/.test(s.elements.diagTitle.innerText), 7000,
    "scenario3: badge clears after the first backoff");
  const elapsed = Date.now() - tDrop;
  check("first backoff waited ~2s (jittered) before reconnecting",
    elapsed >= 1400 && elapsed <= 5000, elapsed + "ms");

  await waitFor(() => s.elements.valHr.innerHTML !== hrAtDrop, 6000,
    "scenario3: broadcasts resume after backoff");
  check("broadcasts resume after the first backoff",
    s.elements.valHr.innerHTML !== hrAtDrop,
    hrAtDrop + " | " + s.elements.valHr.innerHTML);
  check("connection-lost badge cleared after reconnect",
    !/CONNECTION LOST/.test(s.elements.diagTitle.innerText),
    s.elements.diagTitle.innerText);
  check("auth overlay stayed hidden through the auto-reconnect",
    s.elements.authOverlay.hidden === true, s.elements.authOverlay.hidden);
  summary(0);
}

function summary(code) {
  const passed = results.filter((r) => r.ok).length;
  const failed = results.length - passed;
  console.log("----");
  console.log("TOTAL=" + results.length + " PASSED=" + passed + " FAILED=" + failed);
  process.exit(code || (failed ? 1 : 0));
}
