const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function browser(location = { protocol: 'http:', hostname: '127.0.0.1', port: '9000' }) {
  const sockets = [];
  const timers = new Map();
  const messages = [];
  let timerId = 0;
  class WebSocket {
    static CONNECTING = 0;
    constructor(url) {
      this.url = new URL(url).href;
      this.readyState = 0;
      sockets.push(this);
    }
    open() { this.readyState = 1; this.onopen?.(); }
    close() {
      if (this.readyState === 3) return;
      this.readyState = 3;
      this.onclose?.();
    }
  }
  const context = vm.createContext({
    location, WebSocket, URL, ArrayBuffer, Blob,
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: (id) => timers.delete(id),
    decodeCameraBinaryFrame: (frame) => messages.push(frame),
    handleCamWSMessage: (message) => messages.push(message),
  });
  for (const file of ['constants.js', 'modules/websocket.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../web_client/static', file), 'utf8'), context);
  }
  return {
    run: (source) => vm.runInContext(source, context), sockets, timers, messages,
    fireTimer(id) { const callback = timers.get(id); timers.delete(id); callback(); },
  };
}

for (const port of [undefined, null, '', 0, -1, 65536, 1.5, 'invalid']) {
  test(`invalid or absent upstream port ${String(port)} still uses the same-origin relay`, () => {
    const page = browser();
    page.run(`App.config={visualize:{port:${JSON.stringify(port)}}}`);
    assert.equal(page.run('getCamWsUrl()'), 'ws://127.0.0.1:9000/visual-ws');
  });
}

for (const port of [1, 8877, '8877', 65535]) {
  test(`upstream port ${port} is resolved by the relay, not the browser`, () => {
    const page = browser();
    page.run(`App.config={visualize:{port:${JSON.stringify(port)}}}`);
    assert.equal(page.run('getCamWsUrl()'), 'ws://127.0.0.1:9000/visual-ws');
  });
}

test('HTTPS and IPv6 page origins produce a valid camera endpoint', () => {
  const page = browser({ protocol: 'https:', hostname: '[::1]', port: '9000' });
  assert.equal(page.run('getCamWsUrl()'), 'wss://[::1]:9000/visual-ws');
});

test('default WebSocket ports are normalized and do not reconnect repeatedly', () => {
  const page = browser({ protocol: 'http:', hostname: '127.0.0.1', port: '' });
  page.run('App.config={visualize:{port:80}}; connectCamWS(); connectCamWS()');
  assert.equal(page.sockets.length, 1);
  assert.equal(page.sockets[0].url, 'ws://127.0.0.1/visual-ws');
});

test('both pending and open connections are reused for the same configuration', () => {
  const page = browser();
  page.run('connectCamWS(); connectCamWS()');
  page.sockets[0].open();
  page.run('connectCamWS()');
  assert.equal(page.sockets.length, 1);
  assert.equal(page.timers.size, 0);
  assert.equal(page.run('App.camWsAlive'), true);
});

test('an upstream configuration change does not replace the browser relay connection', () => {
  const page = browser();
  page.run('connectCamWS(); App.config={visualize:{port:8877}}; connectCamWS()');
  assert.equal(page.sockets.length, 1);
  const [current] = page.sockets;
  assert.equal(current.readyState, 0);
  assert.equal(current.url, 'ws://127.0.0.1:9000/visual-ws');
  assert.equal(page.timers.size, 1);
  assert.equal(current.binaryType, 'arraybuffer');
});

test('automatic reconnect returns to the same-origin relay', () => {
  const page = browser();
  page.run('App.isRunning=true; connectCamWS()');
  page.sockets[0].open();
  page.sockets[0].close();
  assert.equal(page.run('App.camWsAlive'), false);
  page.run('App.config={visualize:{port:8878}}');
  page.fireTimer([...page.timers.keys()][0]);
  assert.equal(page.sockets[1].url, 'ws://127.0.0.1:9000/visual-ws');
});

test('manual disconnect cancels connection and reconnect timers', () => {
  const page = browser();
  page.run('App.isRunning=true; connectCamWS(); disconnectCamWS()');
  assert.equal(page.timers.size, 0);
  assert.equal(page.run('App.camWs'), null);
  assert.equal(page.run('App.camWsAlive'), false);
});

test('a stopped client does not automatically reconnect', () => {
  const page = browser();
  page.run('App.isRunning=false; connectCamWS()');
  page.sockets[0].close();
  assert.equal(page.timers.size, 0);
});

test('connection timeout closes a stalled connection and schedules retry', () => {
  const page = browser();
  page.run('App.isRunning=true; connectCamWS()');
  page.fireTimer([...page.timers.keys()][0]);
  assert.equal(page.sockets[0].readyState, 3);
  assert.equal(page.timers.size, 1);
});

test('an error raised again during close cannot recurse', () => {
  const page = browser();
  page.run('connectCamWS()');
  const socket = page.sockets[0];
  let closes = 0;
  socket.close = () => {
    if (++closes > 2) throw Error('recursive WebSocket error');
    socket.onerror?.();
  };
  socket.onerror();
  assert.equal(closes, 1);
  assert.equal(socket.onerror, null);
});

test('binary frames, text frames and malformed text use their expected handlers', () => {
  const page = browser();
  page.run('connectCamWS()');
  const socket = page.sockets[0];
  const binary = new ArrayBuffer(8);
  const blob = new Blob(['jpeg']);
  socket.onmessage({ data: binary });
  socket.onmessage({ data: blob });
  socket.onmessage({ data: '{"type":"status"}' });
  socket.onmessage({ data: '{malformed' });
  assert.equal(page.messages.length, 3);
  assert.equal(page.messages[0], binary);
  assert.equal(page.messages[1], blob);
  assert.equal(page.messages[2].type, 'status');
});
