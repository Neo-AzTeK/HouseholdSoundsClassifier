const labels = ['voices', 'machines', 'collisions', 'steps', 'music'];

// DOM elements
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
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const silenceGateCheck = document.getElementById('silence-gate');

// Source elements
const sourceBrowser = document.getElementById('source-browser');
const sourceThingy = document.getElementById('source-thingy');
const browserControls = document.getElementById('browser-controls');
const thingyControls = document.getElementById('thingy-controls');
const thingyNodeSelect = document.getElementById('thingy-node-select');
const thingyLinkBadge = document.getElementById('thingy-link-badge');
const thingyRateVal = document.getElementById('thingy-rate-val');
const thingySimBtn = document.getElementById('thingy-sim-btn');

let currentSource = 'browser';
let socket = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let captureNode = null;
let muteNode = null;
let running = false;
let ready = false;
let currentThreshold = 0.50;
let lastScores = null;
let thingyPollTimer = null;

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

  // Check NPU availability
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
    console.error('Failed to query Thingy status:', err);
  }
}

async function requestBackendChange(target) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'set_backend', backend: target }));
  } else {
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
  }
}

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

    if (score >= currentThreshold) {
      row.classList.add('active');
    } else {
      row.classList.remove('active');
    }
  }

  // Update input level bar if RMS provided
  if (data.rms !== undefined && currentSource === 'thingy') {
    const pct = Math.min(100, Math.round(data.rms * 800));
    levelBar.style.width = `${pct}%`;
  }

  // Update telemetry and latency
  if (data.latency_ms !== undefined) {
    const backendLabel = data.backend_id ? data.backend_id.toUpperCase() : 'CPU';
    const sourceLabel = data.node_id !== undefined ? `Node ${data.node_id} · ` : '';
    latencyBadge.textContent = `${sourceLabel}${data.latency_ms} ms (${backendLabel})`;
  }

  outputPre.textContent = JSON.stringify(data, null, 2);
  updatedText.textContent = `Updated ${new Date().toLocaleTimeString()}`;
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

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${protocol}//${location.host}/ws/audio`);

  socket.onopen = () => {
    socket.send(JSON.stringify({ type: 'start', source, sampleRate }));
  };

  socket.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'ready') {
      ready = true;
      if (source === 'browser' && sourceNode && captureNode && muteNode && audioContext) {
        sourceNode.connect(captureNode);
        captureNode.connect(muteNode);
        muteNode.connect(audioContext.destination);
        setStatus(`Mic Active · ${msg.sampleRate.toLocaleString()} Hz`, 'live');
      } else if (source === 'thingy') {
        setStatus('Listening to Thingy Stream', 'live');
      }
      if (msg.backends) updateBackendsUI(msg.backends);
    } else if (msg.type === 'backend_changed') {
      if (msg.backends) updateBackendsUI(msg.backends);
    } else if (msg.type === 'error') {
      if (source === 'browser') stop().then(() => setStatus(`Server error: ${msg.message}`, 'error'));
    } else if (msg.scores || labels.every(label => typeof msg[label] === 'number')) {
      // Check node filtering in Thingy mode
      if (currentSource === 'thingy' && msg.source === 'thingy') {
        const filterNode = thingyNodeSelect.value;
        if (filterNode === 'all' || filterNode === String(msg.node_id)) {
          render(msg);
        }
      } else if (currentSource === 'browser' && msg.source === 'browser') {
        render(msg);
      }
    }
  };

  socket.onerror = () => {
    if (source === 'browser') stop().then(() => setStatus('Connection error', 'error'));
  };
  socket.onclose = () => {
    if (running && source === 'browser') stop();
  };
}

function switchSource(newSource) {
  currentSource = newSource;
  if (newSource === 'browser') {
    sourceBrowser.closest('.segment').classList.add('active');
    sourceThingy.closest('.segment').classList.remove('active');
    browserControls.style.display = 'block';
    thingyControls.style.display = 'none';
    if (thingyPollTimer) {
      clearInterval(thingyPollTimer);
      thingyPollTimer = null;
    }
    stop();
    setStatus('Ready');
  } else {
    sourceThingy.closest('.segment').classList.add('active');
    sourceBrowser.closest('.segment').classList.remove('active');
    browserControls.style.display = 'none';
    thingyControls.style.display = 'block';
    stop(); // stop browser mic
    setStatus('Connecting to Thingy stream...', 'info');
    connectWebSocket('thingy', 4031);
    fetchThingyStatus();
    thingyPollTimer = setInterval(fetchThingyStatus, 2000);
  }
}

// Event Listeners
startButton.addEventListener('click', startBrowserMic);
stopButton.addEventListener('click', stop);

sourceBrowser.addEventListener('change', () => {
  if (sourceBrowser.checked) switchSource('browser');
});
sourceThingy.addEventListener('change', () => {
  if (sourceThingy.checked) switchSource('thingy');
});

thingySimBtn.addEventListener('click', async () => {
  thingySimBtn.disabled = true;
  thingySimBtn.textContent = 'Injecting audio...';
  try {
    await fetch('/api/thingy/mock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: 0, sound: 'tone', chunks: 25 }),
    });
  } catch (err) {
    console.error('Mock stream error:', err);
  } finally {
    setTimeout(() => {
      thingySimBtn.disabled = false;
      thingySimBtn.textContent = 'Simulate Stream (Test)';
    }, 1000);
  }
});

radioCpu.addEventListener('change', () => {
  if (radioCpu.checked) requestBackendChange('cpu');
});

radioNpu.addEventListener('change', () => {
  if (radioNpu.checked) requestBackendChange('npu');
});

thresholdSlider.addEventListener('input', e => {
  const val = Number(e.target.value);
  currentThreshold = val / 100.0;
  thresholdVal.textContent = `${val}%`;
  if (lastScores) render({ scores: lastScores });
});

silenceGateCheck.addEventListener('change', e => {
  const enabled = e.target.checked;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'set_silence_gate', enabled }));
  } else {
    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ silence_gate: enabled }),
    }).catch(console.error);
  }
});

// Check for insecure context (e.g. HTTP across LAN)
function checkSecureContext() {
  const isLocalhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  if (!window.isSecureContext && !isLocalhost) {
    const warning = document.getElementById('https-warning');
    const link = document.getElementById('https-link');
    if (warning && link) {
      link.href = `https://${location.hostname}:${location.port || 8000}${location.pathname}`;
      warning.style.display = 'block';
    }
  }
}

// Initial load
fetchBackends();
checkSecureContext();
