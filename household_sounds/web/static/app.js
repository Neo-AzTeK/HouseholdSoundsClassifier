const labels = ['voices', 'machines', 'impacts'];

// Model thresholds per label
let thresholds = {
  voices: 0.55,
  machines: 0.64,
  impacts: 0.61,
};

// DOM Elements: Navigation Tabs
const tabBtnLive = document.getElementById('tab-btn-live');
const tabBtnStudio = document.getElementById('tab-btn-studio');
const tabLive = document.getElementById('tab-live');
const tabStudio = document.getElementById('tab-studio');

// DOM Elements: Live Classifier
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statusBadge = document.getElementById('status-badge');
const latencyBadge = document.getElementById('latency-badge');
const updatedText = document.getElementById('updated-text');
const outputPre = document.getElementById('json-output');
const levelBar = document.getElementById('level-bar');
const backendNote = document.getElementById('backend-status-note');
const radioCpu = document.getElementById('radio-cpu');
const radioNpu = document.getElementById('radio-npu');
const segmentNpu = document.getElementById('segment-npu');
const silenceGateCheck = document.getElementById('silence-gate');
const agcToggle = document.getElementById('agc-toggle');

// Source elements
const sourceBrowser = document.getElementById('source-browser');
const sourceThingy = document.getElementById('source-thingy');
const browserControls = document.getElementById('browser-controls');
const thingyControls = document.getElementById('thingy-controls');
const thingyNodeSelect = document.getElementById('thingy-node-select');
const thingyLinkBadge = document.getElementById('thingy-link-badge');
const thingyRateVal = document.getElementById('thingy-rate-val');
const thingySimBtn = document.getElementById('thingy-sim-btn');

// Direct BLE elements
const bleStatusBadge = document.getElementById('ble-status-badge');
const bleDeviceName = document.getElementById('ble-device-name');
const bleDetails = document.getElementById('ble-details');
const bleAddress = document.getElementById('ble-address');
const bleRssi = document.getElementById('ble-rssi');
const bleConnectBtn = document.getElementById('ble-connect-btn');
const bleDisconnectBtn = document.getElementById('ble-disconnect-btn');

// Threshold tuning elements
const saveThresholdsBtn = document.getElementById('save-thresholds-btn');
const resetThresholdsBtn = document.getElementById('reset-thresholds-btn');
const thresholdStatusMsg = document.getElementById('threshold-status-msg');

// Dataset Studio elements
const studioSourceThingy = document.getElementById('studio-source-thingy');
const studioSourceBrowser = document.getElementById('studio-source-browser');
const studioRecBadge = document.getElementById('studio-rec-badge');
const studioRecordBtn = document.getElementById('studio-record-btn');
const studioStopBtn = document.getElementById('studio-stop-btn');
const studioTimer = document.getElementById('studio-timer');
const studioWaveformSection = document.getElementById('studio-waveform-section');
const studioTakeLabel = document.getElementById('studio-take-label');
const waveformCanvas = document.getElementById('waveform-canvas');
const waveformMidTime = document.getElementById('waveform-mid-time');
const waveformEndTime = document.getElementById('waveform-end-time');
const studioAudio = document.getElementById('studio-audio');
const timeStartInput = document.getElementById('time-start');
const timeEndInput = document.getElementById('time-end');
const timeDurationDisplay = document.getElementById('time-duration-display');
const btnQuick1s = document.getElementById('btn-quick-1s');
const studioAddSegmentBtn = document.getElementById('studio-add-segment-btn');
const studioAnnotateMsg = document.getElementById('studio-annotate-msg');
const studioDatasetTotal = document.getElementById('studio-dataset-total');
const studioLabelCounts = document.getElementById('studio-label-counts');
const studioSegmentsList = document.getElementById('studio-segments-list');
const studioTrainBtn = document.getElementById('studio-train-btn');
const trainEpochsInput = document.getElementById('train-epochs');
const trainStatusBadge = document.getElementById('train-status-badge');
const trainProgressWrap = document.getElementById('train-progress-wrap');
const trainProgressBar = document.getElementById('train-progress-bar');
const trainProgressText = document.getElementById('train-progress-text');

// State variables
let currentSource = 'browser';
let socket = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let captureNode = null;
let muteNode = null;
let running = false;
let ready = false;
let lastScores = null;

// Studio state
let studioRecording = false;
let studioRecordTimer = null;
let studioRecordStartTime = 0;
let currentTake = null; // { recording_id, filename, duration_s, waveform }
let selectionStart = 0.0;
let selectionEnd = 1.0;
let isDraggingWaveform = false;
let dragStartX = 0;

// ==============================================================================
// Tab Navigation
// ==============================================================================
tabBtnLive.addEventListener('click', () => {
  tabBtnLive.classList.add('active');
  tabBtnStudio.classList.remove('active');
  tabLive.style.display = 'block';
  tabStudio.style.display = 'none';
});

tabBtnStudio.addEventListener('click', () => {
  tabBtnStudio.classList.add('active');
  tabBtnLive.classList.remove('active');
  tabStudio.style.display = 'block';
  tabLive.style.display = 'none';
  fetchDatasetSummary();
});

// ==============================================================================
// Live Status & Backends
// ==============================================================================
function setStatus(text, type = '') {
  statusBadge.textContent = text;
  statusBadge.className = 'badge';
  if (type === 'live') statusBadge.classList.add('badge-live');
  else if (type === 'error') statusBadge.classList.add('badge-error');
  else if (type === 'info') statusBadge.classList.add('badge-info');
}

function updateBackendsUI(backends) {
  if (!backends) return;
  const active = backends.active || 'cpu';
  if (active === 'cpu') {
    radioCpu.checked = true;
    radioCpu.closest('.segment').classList.add('active');
    radioNpu.closest('.segment').classList.remove('active');
  } else {
    radioNpu.checked = true;
    radioNpu.closest('.segment').classList.add('active');
    radioCpu.closest('.segment').classList.remove('active');
  }

  if (backends.thresholds) {
    applyThresholds(backends.thresholds);
  }

  if (backends.npu && !backends.npu.available) {
    radioNpu.disabled = true;
    segmentNpu.classList.add('disabled');
    segmentNpu.title = backends.npu.error || 'NPU unavailable';
    backendNote.textContent = `Active: ${backends.cpu.name} · (NPU unavailable: ${backends.npu.error || 'HEF not found'})`;
  } else if (backends.npu && backends.npu.available) {
    radioNpu.disabled = false;
    segmentNpu.classList.remove('disabled');
    segmentNpu.title = 'Hailo-10H NPU ready';
    backendNote.textContent = `Active: ${active === 'cpu' ? backends.cpu.name : backends.npu.name}`;
  }
}

async function fetchBackends() {
  try {
    const res = await fetch('/api/backends');
    if (res.ok) {
      const data = await res.json();
      updateBackendsUI(data);
    }
  } catch (err) {
    console.error('Failed to query backends:', err);
  }
}

// ==============================================================================
// Direct BLE Connection Management
// ==============================================================================
async function fetchBleStatus() {
  try {
    const res = await fetch('/api/ble/status');
    if (!res.ok) return;
    const st = await res.json();

    bleStatusBadge.className = 'badge';
    if (st.connected) {
      bleStatusBadge.textContent = `Connected (${st.packet_rate} pkts/s)`;
      bleStatusBadge.classList.add('badge-live');
      bleDeviceName.textContent = st.name || 'Thingy:53';
      bleDetails.style.display = 'block';
      bleAddress.textContent = st.address || '--';
      bleRssi.textContent = st.rssi !== null ? `RSSI: ${st.rssi} dBm` : 'RSSI: --';
      bleConnectBtn.style.display = 'none';
      bleDisconnectBtn.style.display = 'inline-block';
    } else if (st.state === 'scanning') {
      bleStatusBadge.textContent = 'Scanning...';
      bleStatusBadge.classList.add('badge-info');
      bleDeviceName.textContent = 'Looking for "Thingy ranging"...';
      bleDetails.style.display = 'none';
      bleConnectBtn.style.display = 'none';
      bleDisconnectBtn.style.display = 'inline-block';
    } else if (st.state === 'connecting') {
      bleStatusBadge.textContent = 'Connecting...';
      bleStatusBadge.classList.add('badge-info');
      bleDeviceName.textContent = st.address || 'Thingy:53';
      bleConnectBtn.style.display = 'none';
      bleDisconnectBtn.style.display = 'inline-block';
    } else if (st.state === 'firmware_mismatch') {
      bleStatusBadge.textContent = 'Milestone 1 Firmware';
      bleStatusBadge.className = 'badge badge-error';
      bleDeviceName.textContent = st.name ? `${st.name} (Needs Milestone 2)` : 'Needs Milestone 2';
      bleDetails.style.display = 'block';
      bleAddress.textContent = st.error || 'Milestone 1 firmware has no microphone driver.';
      bleRssi.textContent = st.rssi !== null ? `RSSI: ${st.rssi} dBm` : '';
      bleConnectBtn.style.display = 'inline-block';
      bleDisconnectBtn.style.display = 'none';
    } else {
      bleStatusBadge.textContent = 'BLE Disconnected';
      bleDeviceName.textContent = st.error ? `Error: ${st.error}` : 'Not connected';
      bleDetails.style.display = 'none';
      bleConnectBtn.style.display = 'inline-block';
      bleDisconnectBtn.style.display = 'none';
    }
  } catch (err) {
    // Ignore transient network errors
  }
}

bleConnectBtn.addEventListener('click', async () => {
  bleConnectBtn.disabled = true;
  bleStatusBadge.textContent = 'Initiating scan...';
  try {
    await fetch('/api/ble/connect', { method: 'POST' });
    setTimeout(fetchBleStatus, 500);
  } catch (err) {
    console.error('BLE connect error:', err);
  } finally {
    bleConnectBtn.disabled = false;
  }
});

bleDisconnectBtn.addEventListener('click', async () => {
  bleDisconnectBtn.disabled = true;
  try {
    await fetch('/api/ble/disconnect', { method: 'POST' });
    setTimeout(fetchBleStatus, 300);
  } catch (err) {
    console.error('BLE disconnect error:', err);
  } finally {
    bleDisconnectBtn.disabled = false;
  }
});

// ==============================================================================
// Per-Class Threshold Controls
// ==============================================================================
function applyThresholds(newThresholds) {
  if (!newThresholds) return;
  for (const [label, val] of Object.entries(newThresholds)) {
    if (labels.includes(label)) {
      thresholds[label] = Number(val);
      const row = document.querySelector(`.score-row[data-label="${label}"]`);
      if (row) {
        const pct = Math.round(val * 100);
        const thValEl = row.querySelector('.th-val');
        const marker = row.querySelector('.threshold-marker');
        const slider = row.querySelector('.th-slider');
        if (thValEl) thValEl.textContent = `${pct}%`;
        if (marker) marker.style.left = `${pct}%`;
        if (slider) slider.value = pct;
      }
    }
  }
}

async function fetchThresholds() {
  try {
    const res = await fetch('/api/thresholds');
    if (res.ok) {
      const data = await res.json();
      applyThresholds(data.thresholds);
    }
  } catch (err) {
    console.error('Failed to load thresholds:', err);
  }
}

// Hook sliders
document.querySelectorAll('.th-slider').forEach(slider => {
  slider.addEventListener('input', (e) => {
    const label = e.target.getAttribute('data-label');
    const val = Number(e.target.value) / 100.0;
    thresholds[label] = val;
    const row = document.querySelector(`.score-row[data-label="${label}"]`);
    if (row) {
      const thValEl = row.querySelector('.th-val');
      const marker = row.querySelector('.threshold-marker');
      if (thValEl) thValEl.textContent = `${e.target.value}%`;
      if (marker) marker.style.left = `${e.target.value}%`;
    }
    // Re-evaluate active state immediately
    if (lastScores) render(lastScores);
  });
});

saveThresholdsBtn.addEventListener('click', async () => {
  thresholdStatusMsg.textContent = 'Saving...';
  try {
    const res = await fetch('/api/thresholds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thresholds: thresholds, persist: true }),
    });
    if (res.ok) {
      thresholdStatusMsg.textContent = '✅ Saved permanently to model checkpoint!';
      setTimeout(() => { thresholdStatusMsg.textContent = ''; }, 3500);
    }
  } catch (err) {
    thresholdStatusMsg.textContent = '❌ Failed to save';
  }
});

resetThresholdsBtn.addEventListener('click', async () => {
  thresholdStatusMsg.textContent = 'Resetting...';
  try {
    const res = await fetch('/api/thresholds/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persist: true }),
    });
    if (res.ok) {
      const data = await res.json();
      applyThresholds(data.thresholds);
      thresholdStatusMsg.textContent = '✅ Reset to calibrated defaults!';
      setTimeout(() => { thresholdStatusMsg.textContent = ''; }, 3000);
    }
  } catch (err) {
    thresholdStatusMsg.textContent = '❌ Failed to reset';
  }
});

// ==============================================================================
// Live Score Rendering
// ==============================================================================
function render(data) {
  const scores = data.scores || data;
  lastScores = scores;

  for (const label of labels) {
    const score = Math.max(0, Math.min(1, Number(scores[label]) || 0));
    const row = document.querySelector(`[data-label="${label}"]`);
    if (!row) continue;

    const fill = row.querySelector('.bar-fill');
    const percentOutput = row.querySelector('.score-percent');

    fill.style.width = `${(score * 100).toFixed(1)}%`;
    percentOutput.textContent = `${Math.round(score * 100)}%`;

    // Compare with this class's specific threshold
    const th = thresholds[label] || 0.50;
    if (score >= th) {
      row.classList.add('active');
    } else {
      row.classList.remove('active');
    }
  }

  if (data.rms !== undefined && currentSource === 'thingy') {
    const pct = Math.min(100, Math.round(data.rms * 800));
    levelBar.style.width = `${pct}%`;
  }

  if (data.latency_ms !== undefined) {
    const backendLabel = data.backend_id ? data.backend_id.toUpperCase() : 'CPU';
    const sourceLabel = data.node_id !== undefined ? `Node ${data.node_id} · ` : '';
    latencyBadge.textContent = `${sourceLabel}${data.latency_ms} ms (${backendLabel})`;
  }

  outputPre.textContent = JSON.stringify(data, null, 2);
  updatedText.textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

// ==============================================================================
// Audio Streaming & WebSocket
// ==============================================================================
async function fetchThingyStatus() {
  try {
    const res = await fetch('/api/thingy/status');
    if (!res.ok) return;
    const data = await res.json();
    const nodes = data.nodes || {};

    let totalRate = 0;
    const currentOptions = new Set(Array.from(thingyNodeSelect.options).map(o => o.value));

    for (const [nodeId, info] of Object.entries(nodes)) {
      totalRate += info.packet_rate || 0;
      if (!currentOptions.has(nodeId)) {
        const opt = document.createElement('option');
        opt.value = nodeId;
        opt.textContent = `Node ${nodeId}`;
        thingyNodeSelect.appendChild(opt);
      }
    }

    thingyRateVal.textContent = `${totalRate.toFixed(1)} pkts/s`;
    if (totalRate > 0) {
      thingyLinkBadge.textContent = `Active (${Object.keys(nodes).length} nodes)`;
      thingyLinkBadge.className = 'badge badge-live';
    } else if (Object.keys(nodes).length > 0) {
      thingyLinkBadge.textContent = 'Idle (no recent pkts)';
      thingyLinkBadge.className = 'badge';
    } else {
      thingyLinkBadge.textContent = 'Waiting for packets...';
      thingyLinkBadge.className = 'badge';
    }
  } catch (err) {
    // Ignore transient network errors
  }
}

async function stop() {
  running = false;
  ready = false;
  const closingSocket = socket;
  socket = null;
  if (closingSocket?.readyState === WebSocket.OPEN) {
    closingSocket.send(JSON.stringify({ type: 'stop' }));
    closingSocket.close();
  }
  sourceNode?.disconnect();
  captureNode?.disconnect();
  muteNode?.disconnect();
  mediaStream?.getTracks().forEach(track => track.stop());
  if (audioContext && audioContext.state !== 'closed') await audioContext.close();
  mediaStream = audioContext = sourceNode = captureNode = muteNode = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  levelBar.style.width = '0%';
  if (currentSource === 'browser') {
    setStatus('Stopped');
  }
}

async function startBrowserMic() {
  if (running) return;
  running = true;
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus('Requesting microphone...', 'info');

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule('/static/mic-worklet.js');
    await audioContext.resume();

    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(audioContext, 'mic-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    muteNode = audioContext.createGain();
    muteNode.gain.value = 0;

    captureNode.port.onmessage = event => {
      const samples = new Float32Array(event.data);
      let sum = 0;
      for (const sample of samples) sum += sample * sample;
      levelBar.style.width = `${Math.min(100, Math.sqrt(sum / samples.length) * 600)}%`;
      if (ready && socket?.readyState === WebSocket.OPEN && socket.bufferedAmount < 250000) {
        socket.send(event.data);
      }
    };

    connectWebSocket('browser', audioContext.sampleRate);
  } catch (error) {
    await stop();
    setStatus(`Could not start: ${error.message}`, 'error');
  }
}

function connectWebSocket(source, sampleRate = 48000) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }

  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${window.location.host}/ws/audio`;
  socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    socket.send(JSON.stringify({ type: 'start', source: source, sampleRate: sampleRate }));
  };

  socket.onmessage = event => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'ready') {
        ready = true;
        setStatus(source === 'browser' ? 'Listening (Browser Mic)' : 'Listening (Thingy Stream)', 'live');
        if (data.backends) updateBackendsUI(data.backends);
      } else if (data.type === 'prediction') {
        render(data);
      }
    } catch (e) {
      console.error('Error handling socket message:', e);
    }
  };

  socket.onerror = () => {
    setStatus('WebSocket error', 'error');
  };

  socket.onclose = () => {
    ready = false;
    if (running && currentSource === 'browser') {
      setStatus('Disconnected from server', 'error');
      stop();
    }
  };
}

// Source switcher
sourceBrowser.addEventListener('change', () => {
  if (sourceBrowser.checked) {
    currentSource = 'browser';
    sourceBrowser.closest('.segment').classList.add('active');
    sourceThingy.closest('.segment').classList.remove('active');
    browserControls.style.display = 'block';
    thingyControls.style.display = 'none';
    setStatus('Ready');
  }
});

sourceThingy.addEventListener('change', () => {
  if (sourceThingy.checked) {
    currentSource = 'thingy';
    sourceThingy.closest('.segment').classList.add('active');
    sourceBrowser.closest('.segment').classList.remove('active');
    browserControls.style.display = 'none';
    thingyControls.style.display = 'block';
    if (running) stop();
    connectWebSocket('thingy', 4031);
    fetchThingyStatus();
    fetchBleStatus();
  }
});

startButton.addEventListener('click', startBrowserMic);
stopButton.addEventListener('click', stop);

thingySimBtn.addEventListener('click', async () => {
  thingySimBtn.disabled = true;
  thingySimBtn.textContent = 'Injecting tone...';
  try {
    await fetch('/api/thingy/mock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: 0, sound: 'tone', chunks: 30 }),
    });
  } catch (err) {
    console.error('Mock stream error:', err);
  } finally {
    thingySimBtn.disabled = false;
    thingySimBtn.textContent = 'Simulate Stream (Test)';
  }
});

// Backend radio
document.querySelectorAll('input[name="backend"]').forEach(input => {
  input.addEventListener('change', async (e) => {
    const target = e.target.value;
    try {
      const res = await fetch('/api/backend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: target }),
      });
      const data = await res.json();
      if (data.backends) updateBackendsUI(data.backends);
    } catch (err) {
      console.error('Backend switch failed:', err);
    }
  });
});

silenceGateCheck.addEventListener('change', async () => {
  await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ silence_gate: silenceGateCheck.checked }),
  });
});

if (agcToggle) {
  agcToggle.addEventListener('change', async () => {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_agc: agcToggle.checked }),
    });
  });
}

// ==============================================================================
// DATASET STUDIO: RECORD, ANNOTATE, TRAIN
// ==============================================================================

// Timer helper
function updateStudioTimer() {
  const elapsed = (Date.now() - studioRecordStartTime) / 1000.0;
  const mins = Math.floor(elapsed / 60);
  const secs = (elapsed % 60).toFixed(1);
  studioTimer.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(4, '0')}`;
}

studioRecordBtn.addEventListener('click', async () => {
  const source = studioSourceThingy.checked ? 'thingy' : 'browser';
  studioRecordBtn.disabled = true;
  studioStopBtn.disabled = false;
  studioRecBadge.textContent = 'Recording Take...';
  studioRecBadge.className = 'badge badge-live';

  studioRecordStartTime = Date.now();
  studioTimer.textContent = '00:00.0';
  studioRecordTimer = setInterval(updateStudioTimer, 100);

  // If browser source, ensure mic is running
  if (source === 'browser' && !running) {
    await startBrowserMic();
  }

  const sr = source === 'thingy' ? 4031 : (audioContext?.sampleRate || 48000);
  await fetch('/api/studio/record/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: source, sample_rate: sr }),
  });
});

studioStopBtn.addEventListener('click', async () => {
  studioStopBtn.disabled = true;
  clearInterval(studioRecordTimer);
  studioRecBadge.textContent = 'Saving Take...';
  studioRecBadge.className = 'badge badge-info';

  try {
    const res = await fetch('/api/studio/record/stop', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to save take');
    }
    const data = await res.json();
    currentTake = data;
    loadWaveform(data);
    studioRecBadge.textContent = `Saved (${data.duration_s}s)`;
    studioRecBadge.className = 'badge';
  } catch (err) {
    alert(`Could not save take: ${err.message}`);
    studioRecBadge.textContent = 'Error';
    studioRecBadge.className = 'badge badge-error';
  } finally {
    studioRecordBtn.disabled = false;
  }
});

function loadWaveform(take) {
  currentTake = take;
  studioWaveformSection.style.display = 'block';
  studioTakeLabel.textContent = `${take.filename} (${take.duration_s}s)`;

  waveformEndTime.textContent = `${take.duration_s.toFixed(1)}s`;
  waveformMidTime.textContent = `${(take.duration_s / 2.0).toFixed(1)}s`;

  // Set audio source
  studioAudio.src = `/api/studio/audio/${take.filename}`;

  // Default selection: 0.0s to min(1.0s, duration)
  selectionStart = 0.0;
  selectionEnd = Math.min(1.0, take.duration_s);
  updateTimingInputs();
  drawWaveform();
}

function updateTimingInputs() {
  timeStartInput.value = selectionStart.toFixed(2);
  timeEndInput.value = selectionEnd.toFixed(2);
  timeDurationDisplay.textContent = `${(selectionEnd - selectionStart).toFixed(2)} s`;
}

function drawWaveform() {
  if (!currentTake || !waveformCanvas) return;
  const ctx = waveformCanvas.getContext('2d');
  const w = waveformCanvas.width;
  const h = waveformCanvas.height;
  const env = currentTake.waveform || [];
  const duration = currentTake.duration_s || 1.0;

  // Background
  ctx.fillStyle = '#090d16';
  ctx.fillRect(0, 0, w, h);

  // Centerline
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Waveform bars
  ctx.fillStyle = '#38bdf8';
  const barWidth = w / env.length;
  for (let i = 0; i < env.length; i++) {
    const val = env[i];
    const barHeight = Math.max(2, val * (h - 10));
    const x = i * barWidth;
    const y = (h - barHeight) / 2;
    ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), barHeight);
  }

  // Draw Selection Window Overlay
  const startX = (selectionStart / duration) * w;
  const endX = (selectionEnd / duration) * w;
  const selW = Math.max(2, endX - startX);

  ctx.fillStyle = 'rgba(245, 158, 11, 0.28)';
  ctx.fillRect(startX, 0, selW, h);

  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, 0, selW, h);

  // Selection handles
  ctx.fillStyle = '#f59e0b';
  ctx.fillRect(startX - 2, 0, 4, h);
  ctx.fillRect(endX - 2, 0, 4, h);
}

// Waveform canvas interaction (Drag to select time window)
waveformCanvas.addEventListener('mousedown', (e) => {
  if (!currentTake) return;
  const rect = waveformCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const pct = Math.max(0, Math.min(1, x / rect.width));
  const t = pct * currentTake.duration_s;

  isDraggingWaveform = true;
  dragStartX = t;
  selectionStart = t;
  selectionEnd = Math.min(currentTake.duration_s, t + 1.0);
  updateTimingInputs();
  drawWaveform();
});

waveformCanvas.addEventListener('mousemove', (e) => {
  if (!isDraggingWaveform || !currentTake) return;
  const rect = waveformCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const pct = Math.max(0, Math.min(1, x / rect.width));
  const t = pct * currentTake.duration_s;

  if (t >= dragStartX) {
    selectionStart = dragStartX;
    selectionEnd = Math.max(dragStartX + 0.1, t);
  } else {
    selectionStart = Math.max(0, t);
    selectionEnd = dragStartX;
  }
  updateTimingInputs();
  drawWaveform();
});

window.addEventListener('mouseup', () => {
  if (isDraggingWaveform) {
    isDraggingWaveform = false;
  }
});

timeStartInput.addEventListener('change', () => {
  selectionStart = Math.max(0, Number(timeStartInput.value));
  if (selectionEnd <= selectionStart) selectionEnd = selectionStart + 1.0;
  updateTimingInputs();
  drawWaveform();
});

timeEndInput.addEventListener('change', () => {
  selectionEnd = Math.min(currentTake?.duration_s || 10, Number(timeEndInput.value));
  if (selectionStart >= selectionEnd) selectionStart = Math.max(0, selectionEnd - 1.0);
  updateTimingInputs();
  drawWaveform();
});

btnQuick1s.addEventListener('click', () => {
  if (!currentTake) return;
  selectionEnd = Math.min(currentTake.duration_s, selectionStart + 1.0);
  updateTimingInputs();
  drawWaveform();
});

// Save segment to dataset
studioAddSegmentBtn.addEventListener('click', async () => {
  if (!currentTake) {
    alert('Please record or load an audio take first');
    return;
  }

  // Collect checked labels
  const selectedLabels = {};
  document.querySelectorAll('input[name="studio-label"]').forEach(cb => {
    selectedLabels[cb.value] = cb.checked ? 1 : 0;
  });

  const hasAnyLabel = Object.values(selectedLabels).some(v => v === 1);
  if (!hasAnyLabel) {
    alert('Please select at least one label (e.g. Steps, Voices, etc.) for this window');
    return;
  }

  studioAnnotateMsg.textContent = 'Saving segment...';
  try {
    const res = await fetch('/api/studio/annotate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recording_id: currentTake.recording_id,
        start_s: selectionStart,
        end_s: selectionEnd,
        labels: selectedLabels,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to save segment');
    }
    studioAnnotateMsg.textContent = '✅ Segment added to dataset!';
    setTimeout(() => { studioAnnotateMsg.textContent = ''; }, 3000);
    fetchDatasetSummary();
  } catch (err) {
    studioAnnotateMsg.textContent = `❌ ${err.message}`;
  }
});

// Fetch & Render Dataset Summary
async function fetchDatasetSummary() {
  try {
    const res = await fetch('/api/studio/dataset');
    if (!res.ok) return;
    const data = await res.json();

    studioDatasetTotal.textContent = `${data.total_segments} Segments`;

    // Render label pills
    const lc = data.label_counts || {};
    studioLabelCounts.innerHTML = `
      <span class="count-pill">Voices: <strong>${lc.voices || 0}</strong></span>
      <span class="count-pill">Machines: <strong>${lc.machines || 0}</strong></span>
      <span class="count-pill">Impacts: <strong>${lc.impacts || 0}</strong></span>
    `;

    // Render segments list
    const segs = data.segments || [];
    if (segs.length === 0) {
      studioSegmentsList.innerHTML = `<div class="empty-state">No custom segments recorded yet. Record a take above and annotate segments!</div>`;
    } else {
      studioSegmentsList.innerHTML = segs.map(s => {
        const tagList = Object.entries(s.labels || {})
          .filter(([_, v]) => v === 1)
          .map(([k, _]) => `<span class="tag-badge">${k}</span>`)
          .join(' ');
        const dur = s.meta?.duration_s || 1.0;
        const start = s.meta?.start_s || 0.0;
        return `
          <div class="segment-row" data-id="${s.id}">
            <div>
              <strong>${s.id}</strong> (${start}s–${(start + dur).toFixed(1)}s)
              <div class="segment-tags" style="margin-top: 2px;">${tagList || '<em>silence</em>'}</div>
            </div>
            <button class="btn-icon" onclick="deleteSegment('${s.id}')" title="Delete segment">🗑️</button>
          </div>
        `;
      }).join('');
    }
  } catch (err) {
    console.error('Failed to query dataset summary:', err);
  }
}

window.deleteSegment = async function(segmentId) {
  if (!confirm(`Delete segment ${segmentId}?`)) return;
  try {
    await fetch(`/api/studio/segment/${segmentId}`, { method: 'DELETE' });
    fetchDatasetSummary();
  } catch (err) {
    alert(`Could not delete segment: ${err.message}`);
  }
};

// Train Model with Custom Dataset
studioTrainBtn.addEventListener('click', async () => {
  const epochs = Number(trainEpochsInput.value) || 10;
  studioTrainBtn.disabled = true;
  trainProgressWrap.style.display = 'block';
  trainProgressBar.style.width = '5%';
  trainStatusBadge.textContent = 'Training...';
  trainStatusBadge.className = 'badge badge-live';
  trainProgressText.textContent = `Starting fine-tuning (${epochs} epochs)...`;

  try {
    const res = await fetch('/api/studio/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: epochs }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || 'Training failed to start');

    // Poll status
    const pollInterval = setInterval(async () => {
      try {
        const sRes = await fetch('/api/studio/train/status');
        const st = await sRes.json();
        if (st.state === 'training') {
          const pct = Math.min(95, Math.round((st.progress_epoch / st.total_epochs) * 100));
          trainProgressBar.style.width = `${pct}%`;
          trainProgressText.textContent = st.message || `Epoch ${st.progress_epoch}/${st.total_epochs}...`;
        } else if (st.state === 'completed') {
          clearInterval(pollInterval);
          trainProgressBar.style.width = '100%';
          trainStatusBadge.textContent = 'Completed!';
          trainStatusBadge.className = 'badge badge-live';
          trainProgressText.textContent = '✅ Fine-tuning complete! Model reloaded with updated weights.';
          studioTrainBtn.disabled = false;
          fetchBackends();
        } else if (st.state === 'error') {
          clearInterval(pollInterval);
          trainStatusBadge.textContent = 'Error';
          trainStatusBadge.className = 'badge badge-error';
          trainProgressText.textContent = `❌ ${st.message}`;
          studioTrainBtn.disabled = false;
        }
      } catch (err) {
        // Continue polling
      }
    }, 1000);
  } catch (err) {
    trainProgressText.textContent = `❌ ${err.message}`;
    trainStatusBadge.textContent = 'Failed';
    trainStatusBadge.className = 'badge badge-error';
    studioTrainBtn.disabled = false;
  }
});

// Periodic background polling for BLE & nodes
setInterval(fetchBleStatus, 2000);
setInterval(fetchThingyStatus, 2500);

// Init on load
fetchBackends();
fetchThresholds();
fetchBleStatus();
