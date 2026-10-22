/* Display-only workspace controls. Preferences never change robot configuration. */
'use strict';

const TELEMETRY_STALE_MS = 8000;
const Dashboard = { lastStatsAt: null, statsTransport: null, view: 'overview', legend: false };

function hasFreshDashboardStats() {
  return Dashboard.lastStatsAt !== null && Date.now() - Dashboard.lastStatsAt < TELEMETRY_STALE_MS;
}

function readDashboardPreferences() {
  try {
    const value = JSON.parse(localStorage.getItem('openrail.workspace.v1'));
    return value && typeof value === 'object' ? value : {};
  } catch (_) { return {}; }
}

function saveDashboardPreferences() {
  try {
    localStorage.setItem('openrail.workspace.v1', JSON.stringify({
      view: Dashboard.view, legend: Dashboard.legend,
    }));
  } catch (_) { /* Storage may be disabled; display controls still work. */ }
}

function setWorkspaceView(view, persist = true) {
  if (!['overview', 'analysis', 'vision'].includes(view)) view = 'overview';
  Dashboard.view = view;
  document.body.dataset.workspaceView = view;
  const hideLeft = view !== 'overview';
  const hideRight = view === 'analysis';
  $('left-col').classList.toggle('collapsed', hideLeft);
  $('panel-visual').classList.toggle('collapsed', hideRight);
  $('btn-config-hide').classList.toggle('hidden', hideLeft);
  $('btn-config-reveal').classList.toggle('hidden', !hideLeft);
  $('btn-visual-collapse').classList.toggle('hidden', hideRight);
  $('btn-visual-reveal').classList.toggle('hidden', !hideRight);
  $('lang-body').classList.toggle('collapsed', view === 'analysis');
  $('traj-body').classList.remove('collapsed');
  updateCenterPanelLayoutState();
  syncCenterPanelCollapseUi();
  document.querySelectorAll('button[data-workspace-view]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.workspaceView === view));
  });
  updateLayoutColumns();
  requestAnimationFrame(() => App.traj.chart?.resize());
  if (persist) saveDashboardPreferences();
}

function noteDashboardStats(transport) {
  Dashboard.lastStatsAt = Date.now();
  Dashboard.statsTransport = transport;
}

function renderDashboardRuntimeBadge() {
  const fresh = hasFreshDashboardStats();
  const state = !fresh ? 'unknown' : !App.isRunning ? 'stopped' : App.isPaused ? 'paused' : 'running';
  const badgeText = !fresh ? (Dashboard.lastStatsAt === null ? 'CONNECTING' : 'UNKNOWN') : state.toUpperCase();
  const badge = $('status-badge');
  if (badge.textContent !== badgeText) badge.textContent = badgeText;
  badge.className = `status-badge ${state}`;
}

function updateDashboardHealth() {
  const age = Dashboard.lastStatsAt === null ? null : Date.now() - Dashboard.lastStatsAt;
  const fresh = hasFreshDashboardStats();
  const health = $('telemetry-health');
  const text = $('telemetry-health-text');
  health.dataset.state = fresh ? 'live' : age === null ? 'waiting' : 'stale';
  const healthText = fresh
    ? (Dashboard.statsTransport === 'ws' ? (App.wsAlive ? 'Live telemetry' : 'Reconnecting · last telemetry') : 'Live · HTTP fallback')
    : age === null ? 'Waiting for telemetry' : 'Telemetry stale · retrying';
  if (text.textContent !== healthText) text.textContent = healthText;
  $('telemetry-age').textContent = age === null ? 'Awaiting first update'
    : age < 2000 ? 'Updated just now' : `Last update ${Math.floor(age / 1000)}s ago`;

  // A stale transport must not leave a green RUNNING badge looking authoritative.
  renderDashboardRuntimeBadge();
  $('zmq-indicator').classList.toggle('telemetry-stale', !fresh);

  const robot = App.config?.robots?.type;
  const mode = App.config?.rdm?.mode;
  const context = [robot, typeof mode === 'string' ? mode : null].filter(Boolean).join(' / ');
  $('workspace-context').textContent = context || 'Waiting for configuration';
  for (const [id, label] of [['ws-indicator', 'Visual server'], ['zmq-indicator', 'Model server']]) {
    const dot = $(id);
    const connected = dot.classList.contains('connected') && (id !== 'zmq-indicator' || fresh);
    dot.title = `${label}: ${connected ? 'connected' : 'disconnected or awaiting data'}`;
    dot.setAttribute('aria-label', dot.title);
  }
  updateChartReadout();
}

function updateChartReadout() {
  const t = App.traj;
  const datasets = t.chart?.data.datasets || [];
  const hasData = datasets.some((dataset, index) => t.chart.isDatasetVisible(index) &&
    dataset.data?.some(point => Number.isFinite(point.y)));
  $('chart-empty').classList.toggle('hidden', hasData);
  const latestDataAt = Math.max(0, ...[...t.source].map(source => t.lastReceivedAt?.[source] || 0));
  const streamFresh = latestDataAt > 0 && Date.now() - latestDataAt < TELEMETRY_STALE_MS;
  const displayState = t.paused ? 'Display paused' : !App.isObserveRunning ? 'Observation stopped'
    : !App.camWsAlive ? 'Stream disconnected' : !streamFresh ? 'Waiting for stream' : 'Live display';
  const summary = `${t.selectedJoints.size} joints · ${t.source.size} sources · ${displayState}`;
  if ($('chart-summary').textContent !== summary) $('chart-summary').textContent = summary;
  let title = 'Ready to visualize';
  let description = 'Start observation, then play the trajectory stream.';
  if (!t.source.size || !t.selectedJoints.size) {
    title = 'Choose what to plot';
    description = 'Select at least one source and one joint above.';
  } else if (datasets.length && datasets.every((_, index) => !t.chart.isDatasetVisible(index))) {
    title = 'All curves are hidden';
    description = 'Use the legend to show a curve again.';
  } else if (t.paused) {
    title = 'Trajectory display paused';
    description = 'Press Play above to display incoming joint positions.';
  } else if (App.isObserveRunning) {
    title = 'Waiting for joint data';
    description = 'Selected trajectories will appear when data arrives.';
  }
  $('chart-empty-title').textContent = title;
  $('chart-empty-description').textContent = description;
}

function setupDashboard() {
  const preferences = readDashboardPreferences();
  Dashboard.legend = preferences.legend === true;
  setWorkspaceView(preferences.view, false);
  document.querySelectorAll('button[data-workspace-view]').forEach(button => {
    button.addEventListener('click', () => setWorkspaceView(button.dataset.workspaceView));
  });
  const setLegend = () => {
    $('btn-chart-legend').setAttribute('aria-pressed', String(Dashboard.legend));
    if (App.traj.chart) {
      App.traj.chart.options.plugins.legend.display = Dashboard.legend;
      App.traj.chart.update('none');
    }
  };
  setLegend();
  $('btn-chart-legend').addEventListener('click', () => {
    Dashboard.legend = !Dashboard.legend;
    setLegend();
    saveDashboardPreferences();
  });
  $('btn-chart-reset').addEventListener('click', () => {
    App.traj.chart?.resetZoom();
    refreshUnifiedChart();
  });
  for (let i = 0; i < 3; i++) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-xs cam-expand';
    button.textContent = '⤢';
    button.title = `Expand ${CAM_NAMES[i]} camera`;
    button.setAttribute('aria-label', button.title);
    button.setAttribute('aria-haspopup', 'dialog');
    button.setAttribute('aria-controls', 'camera-expanded');
    button.setAttribute('aria-expanded', 'false');
    button.addEventListener('click', () => toggleExpandedCamera(i));
    $(`cam-row-${i}`).insertBefore(button, $(`btn-cam-${i}`));
    $(`cam-img-${i}`).alt = `${CAM_NAMES[i]} camera feed`;
  }
  document.querySelectorAll('button[title]').forEach(button => {
    if (!button.textContent.trim() && !button.hasAttribute('aria-label')) {
      button.setAttribute('aria-label', button.title);
    }
  });
  $('conf-search').setAttribute('aria-label', 'Filter configuration keys');
  $('traj-window-span-slider').setAttribute('aria-label', 'Trajectory window in seconds');
  updateDashboardHealth();
  setInterval(updateDashboardHealth, 1000);
}
