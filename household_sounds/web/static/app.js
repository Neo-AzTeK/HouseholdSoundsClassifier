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

  // Update telemetry and latency
  if (data.latency_ms !== undefined) {
    const backendLabel = data.backend_id ? data.backend_id.toUpperCase() : 'CPU';
    latencyBadge.textContent = `${data.latency_ms} ms (${backendLabel})`;
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
  setStatus('Stopped');
}

async function start() {
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

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${location.host}/ws/audio`);

    socket.onopen = () => {
      socket.send(JSON.stringify({ type: 'start', sampleRate: audioContext.sampleRate }));
    };

    socket.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'ready') {
        ready = true;
        sourceNode.connect(captureNode);
        captureNode.connect(muteNode);
        muteNode.connect(audioContext.destination);
        setStatus(`Listening · ${msg.sampleRate.toLocaleString()} Hz`, 'live');
        if (msg.backends) updateBackendsUI(msg.backends);
      } else if (msg.type === 'backend_changed') {
        if (msg.backends) updateBackendsUI(msg.backends);
      } else if (msg.type === 'error') {
        stop().then(() => setStatus(`Server error: ${msg.message}`, 'error'));
      } else if (msg.scores || labels.every(label => typeof msg[label] === 'number')) {
        render(msg);
      }
    };

    socket.onerror = () => stop().then(() => setStatus('Connection error', 'error'));
    socket.onclose = () => { if (running) stop(); };

  } catch (error) {
    await stop();
    setStatus(`Could not start: ${error.message}`, 'error');
  }
}

// Event Listeners
startButton.addEventListener('click', start);
stopButton.addEventListener('click', stop);

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

// Initial load
fetchBackends();
