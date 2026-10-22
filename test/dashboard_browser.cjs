/* Real-browser smoke test with isolated HTTP/WS fixtures. No robot services needed.
 * Node >=22, Chromium: CHROMIUM_BIN=/path/to/chromium node test/dashboard_browser.cjs
 * Optional screenshots: UI_SCREENSHOT_DIR=/tmp/openrail-ui
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const { spawn } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const stats = { running: false, paused: false, zmq_connected: false, obv_fps: 0,
  infer_count: 0, cpu_usage: 12, gpu_usage: 0, mem_usage: 24, bandwidth_m: 0 };
const config = {
  robots: { type: 'mock', mock: { action_layout: { arm: { start: 0, end: 14 } } } },
  rdm: { mode: 'async', max_len: 100 },
  controller: { wait_time: 10, period: 3.75, speed: 1, gripper_offset: 5 },
  intra_chunk: { intra_chunk_mode: 'fitting', fitting_deg: 3, fitting_num_samples: 64 },
  inter_chunk: { inter_chunk_mode: 'min_jerk' },
  vision: { preprocess: { method: 'resize', width: 224 } },
  language: { task_id: 'Pick and place', sub_task_id: 0, auto_mode: false,
    task_progress_threshold: 0.9, task_progress_win_size: 10 },
  record: { switch: false, auto: false, lerobot: {}, evaluation: { scores: [0, 0.5, 1] } },
  visualize: { camera: { open_head: false, open_wrist_left: false, open_wrist_right: false },
    trajectory: { play: false, source: ['State'], selected_joints: [0, 1, 2, 7], window_span_sec: 10 } },
  vla_zmq: { server_ip: '127.0.0.1', server_port: 5555 },
};
const requests = [];
const missingAssets = [];
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/')) {
    requests.push({ method: req.method, path: pathname });
    let value = {};
    if (pathname.endsWith('/config/path')) value = { path: 'default_conf.yaml', exists: true };
    else if (pathname.endsWith('/config/yaml_files')) value = { files: ['default_conf.yaml'] };
    else if (pathname.endsWith('/config') || pathname.endsWith('/config/patch')) value = { config };
    else if (pathname.includes('/language/load')) value = { data: {
      'Pick and place': ['Pick up the red apple.', 'Place the apple into the bowl.', 'Return to the initial position.'],
      'Sort objects': ['Move the blue block to the tray.'],
    } };
    else if (pathname.endsWith('/status')) value = { data: stats };
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(value));
    return;
  }
  const relative = pathname === '/' ? 'web_client/static/index.html'
    : pathname.startsWith('/static/') ? 'web_client' + pathname : pathname.slice(1);
  const filename = path.resolve(root, relative);
  if (!filename.startsWith(root + path.sep) || !fs.existsSync(filename) || !fs.statSync(filename).isFile()) {
    if (pathname !== '/favicon.ico') missingAssets.push(pathname);
    res.writeHead(pathname === '/favicon.ico' ? 204 : 404); res.end(); return;
  }
  const mime = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css' };
  res.writeHead(200, { 'Content-Type': mime[path.extname(filename)] || 'application/octet-stream' });
  fs.createReadStream(filename).pipe(res);
});

async function main() {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'openrail-browser-'));
  const chrome = spawn(process.env.CHROMIUM_BIN || '/usr/bin/chromium-browser', [
    '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
    '--no-proxy-server', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  let ws;
  try {
    const endpoint = await new Promise((resolve, reject) => {
      let output = '';
      const timeout = setTimeout(() => reject(Error('Chromium startup timed out')), 15000);
      chrome.on('error', error => { clearTimeout(timeout); reject(error); });
      chrome.stderr.on('data', chunk => {
        output += chunk;
        const match = output.match(/DevTools listening on (ws:\/\/[^\s]+)/);
        if (match) { clearTimeout(timeout); resolve(new URL(match[1]).origin.replace('ws:', 'http:')); }
      });
    });
    const targets = await (await fetch(endpoint + '/json/list')).json();
    ws = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(Error('DevTools connection timed out')), 10000);
      ws.addEventListener('open', () => { clearTimeout(timeout); resolve(); }, { once: true });
      ws.addEventListener('error', () => { clearTimeout(timeout); reject(Error('DevTools connection failed')); }, { once: true });
    });
    let nextId = 0;
    const pending = new Map();
    const errors = [];
    ws.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const pair = pending.get(message.id);
        if (pair) { pending.delete(message.id); message.error ? pair.reject(message.error) : pair.resolve(message.result); }
      } else if (message.method === 'Runtime.exceptionThrown') {
        errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
      }
    });
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++nextId;
      const timeout = setTimeout(() => { pending.delete(id); reject(Error(`DevTools timed out: ${method}`)); }, 15000);
      pending.set(id, {
        resolve: result => { clearTimeout(timeout); resolve(result); },
        reject: error => { clearTimeout(timeout); reject(error); },
      });
      ws.send(JSON.stringify({ id, method, params }));
    });
    const evaluate = async expression => {
      const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
      return result.result.value;
    };
    const waitFor = async expression => {
      const deadline = Date.now() + 8000;
      while (Date.now() < deadline) {
        if (await evaluate(expression)) return;
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      throw Error(`Timed out: ${expression}`);
    };
    const click = selector => evaluate(`(() => {
      const button = document.querySelector(${JSON.stringify(selector)});
      button.focus({preventScroll: true}); button.click();
    })()`);
    const resize = async (width, height) => {
      await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
      await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
    };
    await send('Runtime.enable');
    await send('Page.enable');
    await send('Page.addScriptToEvaluateOnNewDocument', { source: `
      window.WebSocket = class {
        static CONNECTING = 0;
        constructor(url) {
          this.url = url; this.readyState = 0;
          setTimeout(() => {
            this.readyState = 1; this.onopen?.();
            if (url.endsWith('/ws')) this.onmessage?.({data: JSON.stringify({type:'stats', data:${JSON.stringify(stats)}})});
          }, 10);
        }
        send() {}
        close() { this.readyState = 3; this.onclose?.(); }
      };
    ` });
    await resize(1600, 1000);
    await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
    await waitFor(`document.querySelectorAll('#lang-task-select option').length === 2 && document.querySelectorAll('.cam-expand').length === 3`);
    await evaluate('updateDashboardHealth()');
    assert.equal(await evaluate(`$('workspace-context').textContent`), 'mock / async');
    assert.equal(await evaluate(`$('status-badge').textContent`), 'STOPPED');
    const initialMutations = requests.filter(request => request.method !== 'GET').length;
    await click('[data-workspace-view="analysis"]');
    assert.equal(await evaluate(`document.querySelector('.layout').classList.contains('left-collapsed') && document.querySelector('.layout').classList.contains('right-collapsed')`), true);
    assert.equal(await evaluate(`$('lang-body').classList.contains('collapsed') && $('left-col').inert && $('panel-visual').inert`), true);
    await click('[data-workspace-view="vision"]');
    assert.equal(await evaluate(`!$('panel-visual').classList.contains('collapsed') && $('left-col').classList.contains('collapsed')`), true);
    await click('#btn-chart-legend');
    assert.equal(await evaluate('App.traj.chart.options.plugins.legend.display'), true);
    await click('#btn-chart-reset');
    await click('.cam-expand');
    assert.equal(await evaluate(`$('camera-expanded').open && $('camera-expanded').contains($('cam-preview-0'))`), true);
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
    await waitFor(`!$('camera-expanded').open && $('camera-body').contains($('cam-preview-0'))`);
    assert.equal(await evaluate(`document.activeElement === $('cam-row-0').querySelector('.cam-expand')`), true, 'Return focus to camera opener');
    assert.equal(await evaluate(`document.activeElement.getAttribute('aria-expanded')`), 'false');
    assert.equal(requests.filter(request => request.method !== 'GET').length, initialMutations, 'Display actions must not mutate robot configuration');
    await send('Page.reload');
    await waitFor(`document.body?.dataset.workspaceView === 'vision' && typeof App !== 'undefined' && App.traj.chart?.options.plugins.legend.display === true && document.querySelectorAll('#lang-task-select option').length === 2`);
    await click('[data-workspace-view="overview"]');
    await evaluate('Dashboard.lastStatsAt = Date.now() - 20000; updateDashboardHealth()');
    assert.equal(await evaluate(`$('status-badge').textContent`), 'UNKNOWN');
    assert.match(await evaluate(`$('telemetry-health-text').textContent`), /stale/);
    await evaluate('setRunningUI(true, false)');
    assert.equal(await evaluate(`$('status-badge').textContent`), 'UNKNOWN', 'Status events must not disguise stale telemetry');
    const staleTimestamp = await evaluate('Dashboard.lastStatsAt');
    await evaluate('renderStats(null); renderStats({}); renderStats({running:"true"})');
    assert.equal(await evaluate('Dashboard.lastStatsAt'), staleTimestamp, 'Malformed data must not refresh telemetry');
    const previousPolls = requests.filter(request => request.path.endsWith('/status')).length;
    await waitFor(`Dashboard.statsTransport === 'http' && hasFreshDashboardStats()`);
    assert(requests.filter(request => request.path.endsWith('/status')).length > previousPolls, 'Silent open WebSocket must trigger HTTP fallback');
    await evaluate('updateDashboardHealth()');
    assert.match(await evaluate(`$('telemetry-health-text').textContent`), /HTTP fallback/);
    await evaluate(`renderStats({...${JSON.stringify(stats)}, running: true, observe_running: true, inference_running: true,
      control_running: true, zmq_connected: true, obv_fps: 30.1, infer_count: 4984,
      img_proc_time: 14, avg_infer_time: .2876, avg_intra_traj_time: .0027,
      avg_inter_traj_time: .0047, avg_comm_time: .0068, current_prob_progress: .42}); updateDashboardHealth();`);
    assert.equal(await evaluate(`$('status-badge').textContent`), 'RUNNING');
    // Real chart datasets exercise rendering and empty-state recovery.
    await click('#btn-traj-pause');
    await evaluate(`App.traj.numJoints = 14;
      for (let index = 0; index < 120; index++) {
        ingestTrajData(Array.from({length:14}, (_,j) => Math.sin(index/18+j)*.3+j*.12-.8), [], [], 100+index/12);
      }
      refreshUnifiedChart(); updateChartReadout();`);
    assert.equal(await evaluate(`$('chart-empty').classList.contains('hidden')`), true);
    await evaluate('App.traj.lastReceivedAt.state = Date.now() - 20000; updateChartReadout()');
    assert.match(await evaluate(`$('chart-summary').textContent`), /Waiting for stream/);
    await evaluate('App.camWsAlive = false; updateChartReadout()');
    assert.match(await evaluate(`$('chart-summary').textContent`), /Stream disconnected/);
    await evaluate('App.camWsAlive = true; App.traj.lastReceivedAt.state = Date.now(); updateChartReadout()');
    assert.match(await evaluate(`$('chart-summary').textContent`), /Live display/);
    const patchesBeforeDrag = requests.filter(request => request.path.endsWith('/config/patch')).length;
    await evaluate(`for (const value of [2, 5, 9]) {
      $('traj-window-span-slider').value = value;
      $('traj-window-span-slider').dispatchEvent(new Event('input'));
    }`);
    assert.equal(requests.filter(request => request.path.endsWith('/config/patch')).length, patchesBeforeDrag);
    await evaluate(`$('traj-window-span-slider').dispatchEvent(new Event('change'))`);
    await waitFor(`$('traj-window-span-value').textContent === '9s'`);
    await new Promise(resolve => setTimeout(resolve, 100));
    assert.equal(requests.filter(request => request.path.endsWith('/config/patch')).length, patchesBeforeDrag + 1);
    // The no-selection state must explain recovery and disappear when restored.
    await evaluate('window.savedJoints = App.traj.selectedJoints; App.traj.selectedJoints = new Set(); refreshUnifiedChart(); updateChartReadout()');
    assert.equal(await evaluate(`$('chart-empty-title').textContent`), 'Choose what to plot');
    await evaluate('App.traj.selectedJoints = window.savedJoints; refreshUnifiedChart(); updateChartReadout()');
    assert.equal(await evaluate(`$('chart-empty').classList.contains('hidden')`), true);
    await click('#btn-chart-legend');
    for (const [width, height] of [[1920, 1080], [1440, 900], [1280, 720], [1151, 768], [1024, 768], [768, 1024], [390, 844], [320, 740]]) {
      await resize(width, height);
      for (const view of ['analysis', 'vision', 'overview']) {
        await click(`button[data-workspace-view="${view}"]`);
        await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
        const overflow = await evaluate(`Array.from(document.querySelectorAll('.panel:not(.collapsed), .workspace-bar, .topbar')).filter(el => {
          if (el.closest('[inert]')) return false;
          const r = el.getBoundingClientRect(); return r.width && (r.right > innerWidth + 1 || r.left < -1);
        }).map(el => el.className)`);
        assert.deepEqual(overflow, [], `${width}px ${view}: panels overflow viewport`);
        const clippedControls = await evaluate(`Array.from(document.querySelectorAll('.panel-header button, .lang-cmd-btn-row button, .chart-toolbar button')).filter(el => {
          if (!el.getClientRects().length || el.closest('[inert]')) return false;
          const p = el.closest('.panel').getBoundingClientRect(), r = el.getBoundingClientRect();
          return p.width > 0 && (r.right > p.right + 1 || r.left < p.left - 1 || r.bottom > p.bottom + 1);
        }).map(el => el.id)`);
        assert.deepEqual(clippedControls, [], `${width}px ${view}: controls are clipped`);
        const overflowInputs = await evaluate(`Array.from(document.querySelectorAll('.lang-cmd-layout select, #lang-cmd-text')).filter(el => {
          if (!el.getClientRects().length) return false;
          const p = $('lang-body').getBoundingClientRect(), r = el.getBoundingClientRect();
          return r.bottom > p.bottom + 1 || r.right > p.right + 1;
        }).map(el => el.id)`);
        assert.deepEqual(overflowInputs, [], `${width}px ${view}: task editor is clipped`);
        if (process.env.UI_SCREENSHOT_DIR && (view === 'overview' || width === 1920)) {
          fs.mkdirSync(process.env.UI_SCREENSHOT_DIR, { recursive: true });
          // Allow Chromium's compositor to finish repainting after the preset switch.
          await new Promise(resolve => setTimeout(resolve, 250));
          const shot = await send('Page.captureScreenshot', { format: 'png' });
          fs.writeFileSync(path.join(process.env.UI_SCREENSHOT_DIR, `dashboard-${width}${view === 'overview' ? '' : '-' + view}.png`), Buffer.from(shot.data, 'base64'));
        }
      }
      // Manually revealing sidebars must work from focused presets, too.
      await click('button[data-workspace-view="analysis"]');
      await click('#btn-config-reveal');
      await click('#btn-visual-reveal');
      assert.equal(await evaluate(`!$('left-col').inert && !$('panel-visual').inert`), true);
      await click('button[data-workspace-view="overview"]');
      for (const panel of ['recording', 'manual', 'config']) {
        await click(`#btn-${panel}-collapse`);
        assert.equal(await evaluate(`!$('${panel === 'config' ? 'config' : panel}-body').classList.contains('collapsed')`), true);
      }
    }
    assert.deepEqual(errors, [], 'No uncaught browser errors');
    assert.deepEqual(missingAssets, [], 'All local assets load');
    console.log('PASS: 3 views × 8 sizes; sidebar panels, persistence, silent-WS fallback, stale streams, camera focus/Esc, chart controls; display actions send no robot mutations.');
  } finally {
    ws?.close();
    chrome.kill('SIGTERM');
    server.close();
    server.closeAllConnections();
    if (chrome.pid && chrome.exitCode === null) {
      await new Promise(resolve => {
        const timeout = setTimeout(() => { chrome.kill('SIGKILL'); resolve(); }, 3000);
        chrome.once('exit', () => { clearTimeout(timeout); resolve(); });
      });
    }
    fs.rmSync(profile, { recursive: true, force: true });
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
