const labels = ['voices', 'machines', 'collisions', 'steps', 'music'];
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statusText = document.getElementById('status');
const statusDot = document.getElementById('status-dot');
const updated = document.getElementById('updated');
const output = document.getElementById('json-output');
const levelBar = document.getElementById('level-bar');

let socket = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let captureNode = null;
let muteNode = null;
let running = false;
let ready = false;

function setStatus(message, kind = '') {
  statusText.textContent = message;
  statusDot.className = `status-dot ${kind}`;
}

function render(scores) {
  for (const label of labels) {
    const score = Math.max(0, Math.min(1, Number(scores[label]) || 0));
    const row = document.querySelector(`[data-label="${label}"]`);
    row.querySelector('.fill').style.width = `${(score * 100).toFixed(1)}%`;
    row.querySelector('output').textContent = `${Math.round(score * 100)}%`;
  }
  output.textContent = JSON.stringify(Object.fromEntries(labels.map(label => [label, scores[label]])), null, 2);
  updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
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
  setStatus('Requesting microphone…');
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule('/static/mic-worklet.js');
    await audioContext.resume();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(audioContext, 'mic-capture', {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]
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
    socket.onopen = () => socket.send(JSON.stringify({ type: 'start', sampleRate: audioContext.sampleRate }));
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.type === 'ready') {
        ready = true;
        sourceNode.connect(captureNode);
        captureNode.connect(muteNode);
        muteNode.connect(audioContext.destination);
        setStatus(`Listening · ${message.sampleRate.toLocaleString()} Hz`, 'live');
      } else if (message.type === 'error') {
        stop().then(() => setStatus(`Server error: ${message.message}`, 'error'));
      } else if (labels.every(label => typeof message[label] === 'number')) {
        render(message);
      }
    };
    socket.onerror = () => stop().then(() => setStatus('Connection error', 'error'));
    socket.onclose = () => { if (running) stop(); };
  } catch (error) {
    await stop();
    setStatus(`Could not start: ${error.message}`, 'error');
  }
}

startButton.addEventListener('click', start);
stopButton.addEventListener('click', stop);
