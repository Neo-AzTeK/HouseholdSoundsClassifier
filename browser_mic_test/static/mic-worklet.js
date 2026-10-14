class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Float32Array(4096);
    this.offset = 0;
  }

  process(inputs, outputs) {
    const channels = inputs[0];
    const output = outputs[0];
    for (const channel of output) channel.fill(0);
    if (!channels || channels.length === 0) return true;
    const length = channels[0].length;
    for (let i = 0; i < length; i++) {
      let mono = 0;
      for (const channel of channels) mono += channel[i];
      this.chunk[this.offset++] = mono / channels.length;
      if (this.offset === this.chunk.length) {
        this.port.postMessage(this.chunk.buffer, [this.chunk.buffer]);
        this.chunk = new Float32Array(4096);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCaptureProcessor);
