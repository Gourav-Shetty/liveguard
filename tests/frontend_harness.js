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
 * password -> re-login, plus stored-token auto-auth on a fresh page load.
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
    return {
      id,
      hidden: !!INITIAL_HIDDEN[id],
      value: "",
      innerText: id === "authStatus" ? "Not connected" : "",
      innerHTML: id === "valHr" ? '-- <span class="unit">BPM</span>' : "",
      disabled: false,
      className: "",
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

  global.__TOKEN__ = tokenFinal;
  global.__USER__ = username;
})()
  .then(() => scenario2(global.__TOKEN__, global.__USER__))
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
  summary(0);
}

function summary(code) {
  const passed = results.filter((r) => r.ok).length;
  const failed = results.length - passed;
  console.log("----");
  console.log("TOTAL=" + results.length + " PASSED=" + passed + " FAILED=" + failed);
  process.exit(code || (failed ? 1 : 0));
}
