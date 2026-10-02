/**
 * NTRO Automated Signal Analyzer - Web Frontend Controller (SIH26147)
 * High-performance HTML5 Canvas Visualizations:
 * 1. 2D Time-Frequency Waterfall Spectrogram
 * 2. Power Spectral Density (PSD / FFT) with Peak Tracker
 * 3. I/Q Constellation Diagram with Persistence and Slicing Guides
 * 4. Time Domain Waveform & Eye Diagram
 */

// State
let currentAnalysis = null;
let activeTimeMode = 'time'; // 'time' or 'eye'
let selectedColormap = 'turbo';

// DOM Elements
const canvasWaterfall = document.getElementById('canvas-waterfall');
const canvasPsd = document.getElementById('canvas-psd');
const canvasConstellation = document.getElementById('canvas-constellation');
const canvasTimeDomain = document.getElementById('canvas-timedomain');

// Init
window.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initModals();
  initColormapSelector();
  initCanvasResize();

  // Initial load check
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      if (data.has_analysis_result) {
        runAnalysis();
      } else {
        // Synthesize initial demo signal (QPSK, 20 dB SNR, 2500 Hz CFO, CCSDS preamble)
        synthesizeDefaultDemo();
      }
    })
    .catch(() => {
      synthesizeDefaultDemo();
    });
});

function initCanvasResize() {
  const resizeAll = () => {
    [canvasWaterfall, canvasPsd, canvasConstellation, canvasTimeDomain].forEach(c => {
      if (!c) return;
      const rect = c.parentElement.getBoundingClientRect();
      c.width = rect.width * (window.devicePixelRatio || 1);
      c.height = rect.height * (window.devicePixelRatio || 1);
    });
    if (currentAnalysis) {
      renderAllVisuals(currentAnalysis.visuals);
    }
  };
  window.addEventListener('resize', resizeAll);
  setTimeout(resizeAll, 100);
}

// ---------------- API & Pipeline Operations ----------------

function synthesizeDefaultDemo() {
  fetch('/api/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modulation: 'QPSK',
      snr_db: 22.0,
      cfo_hz: 2500.0,
      sample_rate: 1000000,
      symbol_rate: 100000,
      num_symbols: 3000,
      preamble: 'CCSDS'
    })
  })
  .then(r => r.json())
  .then(data => {
    if (data.result) {
      updateUI(data.result);
    }
  })
  .catch(err => console.error("Synthesis error:", err));
}

function runAnalysis() {
  const manualMod = 'auto';
  const manualFec = document.getElementById('cfg-fec')?.value || 'auto';
  const manualInterleave = document.getElementById('cfg-interleaver')?.value || 'none';

  fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      manual_mod: manualMod,
      manual_fec: manualFec,
      manual_interleave: manualInterleave,
      target_preamble: 'auto'
    })
  })
  .then(r => r.json())
  .then(data => {
    if (data.result) {
      updateUI(data.result);
    }
  })
  .catch(err => console.error("Analysis error:", err));
}

// ---------------- UI Update Controller ----------------

function updateUI(res) {
  currentAnalysis = res;
  const p = res.parameters;
  const amc = res.amc;
  const framing = res.framing;

  // 1. Telemetry Ribbon
  document.getElementById('val-modulation').textContent = amc.predicted_modulation || '--';
  const confPct = Math.round((amc.confidence || 0) * 100);
  document.getElementById('val-confidence').textContent = `${confPct}%`;
  document.getElementById('bar-confidence').style.width = `${confPct}%`;

  document.getElementById('val-baud').innerHTML = `${Math.round(p.estimated_baud_rate).toLocaleString()} <small>Baud</small>`;
  document.getElementById('val-sps').textContent = `${p.samples_per_symbol}`;
  document.getElementById('val-fs').textContent = `${(p.sample_rate / 1e6).toFixed(2)} MHz`;

  document.getElementById('val-cfo').innerHTML = `${p.estimated_cfo_hz >= 0 ? '+' : ''}${p.estimated_cfo_hz} <small>Hz</small>`;
  document.getElementById('badge-cfo-marker').textContent = `CFO: ${p.estimated_cfo_hz} Hz`;

  document.getElementById('val-snr').innerHTML = `${p.estimated_snr_db} <small>dB</small>`;
  const snr = p.estimated_snr_db;
  const quality = snr > 20 ? 'EXCELLENT' : (snr > 12 ? 'GOOD' : (snr > 5 ? 'MODERATE' : 'POOR / NOISY'));
  document.getElementById('val-snr-quality').textContent = quality;

  document.getElementById('val-obw').innerHTML = `${(p.obw_99_hz / 1000).toFixed(1)} <small>kHz</small>`;
  document.getElementById('val-bw3db').textContent = `${(p.bandwidth_3db_hz / 1000).toFixed(1)} kHz`;

  const bestPre = res.preambles.best_preamble || 'NONE';
  document.getElementById('val-preamble').textContent = bestPre;
  document.getElementById('val-frame-count').textContent = framing.num_frames_detected;
  document.getElementById('val-entropy').textContent = `${framing.overall_entropy} bits/byte`;

  // 2. Tab 1: AMC Table & Likelihood Bars
  updateAmcDetails(amc);

  // 3. Tab 2: Demodulator
  document.getElementById('demod-engine-name').textContent = `${res.active_configuration.modulation} Demodulator`;
  document.getElementById('demod-symbols-count').textContent = res.demodulation.num_symbols.toLocaleString();
  document.getElementById('demod-bits-count').textContent = res.demodulation.num_bits_recovered.toLocaleString();
  const bitSnippet = res.demodulation.bit_sample_snippet.slice(0, 128).join('');
  document.getElementById('demod-bits-preview').textContent = bitSnippet || '--';

  // 4. Tab 3 & 4: Interleaver and FEC Callouts
  document.getElementById('interleave-status-callout').innerHTML = `
    Active Scheme: <strong>${res.active_configuration.interleaving.toUpperCase()}</strong><br>
    De-interleaving Status: <strong>Applied / Synchronized</strong>
  `;
  document.getElementById('fec-results-callout').innerHTML = `
    FEC Engine: <strong>${res.fec_status.fec_applied || 'Bypass'}</strong><br>
    Status: <strong>${res.fec_status.status || 'Decoded'}</strong>
  `;

  // 5. Tab 5: Framing & Hex Dump
  updateFramingDetails(res);

  // 6. Canvases
  renderAllVisuals(res.visuals);
}

function updateAmcDetails(amc) {
  const feats = amc.features || {};
  const tbody = document.getElementById('amc-features-tbody');
  tbody.innerHTML = `
    <tr><td>C₂₀ (2nd Moment)</td><td><strong>${feats.c20 ?? '--'}</strong></td><td>~1.0 (BPSK), ~0.0 (QPSK/QAM)</td><td>BPSK phase linearity indicator</td></tr>
    <tr><td>C₄₀ (4th Moment)</td><td><strong>${feats.c40 ?? '--'}</strong></td><td>~1.6 (BPSK), ~0.8 (QPSK)</td><td>Trellis rotation fourth cumulant</td></tr>
    <tr><td>C₄₂ (Power Cumulant)</td><td><strong>${feats.c42 ?? '--'}</strong></td><td>~1.6 (BPSK), ~0.8 (QPSK), ~0.53 (16QAM)</td><td>Energy variance distribution</td></tr>
    <tr><td>γ_max (Spectral Peak)</td><td><strong>${feats.gamma_max ?? '--'}</strong></td><td>High (ASK/BPSK), Low (CW/FSK)</td><td>Max normalized amplitude peak</td></tr>
    <tr><td>σ_aa (Amp Std Dev)</td><td><strong>${feats.sigma_aa ?? '--'}</strong></td><td>&lt;0.3 (PSK/FSK), &gt;0.35 (QAM)</td><td>Normalized envelope variance</td></tr>
    <tr><td>σ_af (Freq Std Dev)</td><td><strong>${feats.sigma_af ?? '--'}</strong></td><td>&gt;0.05 (FSK), &lt;0.02 (PSK)</td><td>Instantaneous frequency hopping</td></tr>
    <tr><td>FSK Tone Ratio</td><td><strong>${feats.fsk_fdev_ratio ?? '--'}</strong></td><td>&ge;6.0 (FSK), &lt;4.0 (PSK/QAM)</td><td>Tone histogram peak prominence</td></tr>
  `;

  const barsContainer = document.getElementById('amc-prob-bars');
  barsContainer.innerHTML = '';
  const probs = amc.probabilities || {};
  for (const [mod, prob] of Object.entries(probs)) {
    const pct = Math.round(prob * 100);
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <div class="prob-label">${mod}</div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${pct}%"></div>
      </div>
      <div class="prob-pct">${pct}%</div>
    `;
    barsContainer.appendChild(item);
  }
}

function updateFramingDetails(res) {
  const framing = res.framing;
  const preambles = res.preambles;

  const preList = document.getElementById('preambles-summary-list');
  preList.innerHTML = '';
  for (const [k, v] of Object.entries(preambles.detected_preambles || {})) {
    const div = document.createElement('div');
    div.className = 'callout';
    div.innerHTML = `Pattern: <strong>${v.name}</strong> • Matches Found: <strong>${v.num_matches}</strong> • First Offset: <code>bit ${v.match_indices[0]}</code>`;
    preList.appendChild(div);
  }

  const tbody = document.getElementById('table-frames-body');
  if (framing.frames && framing.frames.length > 0) {
    tbody.innerHTML = framing.frames.map(f => `
      <tr>
        <td>#${f.frame_index + 1}</td>
        <td><code>${f.bit_offset}</code></td>
        <td>${f.frame_length_bits} bits</td>
        <td><code>${f.header_hex || '--'}</code></td>
        <td>${f.payload_entropy}</td>
      </tr>
    `).join('');
  } else {
    tbody.innerHTML = `<tr><td colspan="5">Continuous bitstream / no discrete sync frames.</td></tr>`;
  }

  document.getElementById('hex-editor-container').textContent = framing.hex_dump || 'No payload bytes available.';
}

// ---------------- High-FPS Canvas Rendering ----------------

function renderAllVisuals(v) {
  if (!v) return;
  renderWaterfall(v.waterfall);
  renderPsd(v.psd);
  renderConstellation(v.constellation);
  renderTimeDomain(v.time_domain);
}

// 1. Waterfall Spectrogram
function renderWaterfall(wf) {
  if (!wf || !wf.intensity) return;
  const ctx = canvasWaterfall.getContext('2d');
  const w = canvasWaterfall.width;
  const h = canvasWaterfall.height;

  ctx.clearRect(0, 0, w, h);

  const grid = wf.intensity; // [freq_bins, time_bins]
  const nFreq = grid.length;
  const nTime = grid[0].length;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;

  // Render heatmap onto image buffer
  for (let y = 0; y < h; y++) {
    const timeIdx = Math.floor((y / h) * nTime);
    for (let x = 0; x < w; x++) {
      const freqIdx = Math.floor((x / w) * nFreq);
      const val = grid[freqIdx][timeIdx] || 0.0;
      const rgb = getColormapRGB(val, selectedColormap);

      const pIdx = (y * w + x) * 4;
      data[pIdx] = rgb[0];
      data[pIdx + 1] = rgb[1];
      data[pIdx + 2] = rgb[2];
      data[pIdx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // Center line marker
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, h);
  ctx.stroke();
  ctx.setLineDash([]);
}

function getColormapRGB(val, colormap) {
  // val is normalized [0, 1]
  val = Math.max(0, Math.min(1, val));
  if (colormap === 'viridis') {
    return [
      Math.floor(255 * (0.2 + 0.8 * val * val)),
      Math.floor(255 * (0.1 + 0.9 * val)),
      Math.floor(255 * (0.4 + 0.6 * (1 - val)))
    ];
  } else if (colormap === 'inferno') {
    return [
      Math.floor(255 * Math.min(1, val * 1.5)),
      Math.floor(255 * Math.max(0, (val - 0.2) * 1.2)),
      Math.floor(255 * Math.max(0, (val - 0.6) * 2.5))
    ];
  } else if (colormap === 'cyan') {
    return [
      Math.floor(val * 40),
      Math.floor(val * 240),
      Math.floor(val * 255)
    ];
  } else {
    // Turbo
    const r = Math.sin(val * Math.PI * 1.5);
    const g = Math.sin((val + 0.3) * Math.PI);
    const b = Math.cos(val * Math.PI * 1.5);
    return [
      Math.floor(255 * Math.max(0, r)),
      Math.floor(255 * Math.max(0, g)),
      Math.floor(255 * Math.max(0, b))
    ];
  }
}

// 2. Power Spectral Density (PSD)
function renderPsd(psd) {
  if (!psd || !psd.frequencies) return;
  const ctx = canvasPsd.getContext('2d');
  const w = canvasPsd.width;
  const h = canvasPsd.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= w; x += w / 8) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y <= h; y += h / 6) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const freqs = psd.frequencies;
  const powers = psd.power_db;
  const minDb = Math.min(...powers) - 5;
  const maxDb = Math.max(...powers) + 5;
  const dbRange = maxDb - minDb || 1;

  // Draw PSD curve
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 2;
  ctx.beginPath();

  for (let i = 0; i < powers.length; i++) {
    const x = (i / (powers.length - 1)) * w;
    const y = h - ((powers[i] - minDb) / dbRange) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill gradient under PSD
  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
  gradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Peak marker
  const maxIdx = powers.indexOf(Math.max(...powers));
  const peakX = (maxIdx / (powers.length - 1)) * w;
  const peakY = h - ((powers[maxIdx] - minDb) / dbRange) * (h - 20) - 10;

  ctx.fillStyle = '#ff1744';
  ctx.beginPath();
  ctx.arc(peakX, peakY, 4, 0, 2 * Math.PI);
  ctx.fill();

  ctx.fillStyle = '#f1f5f9';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText(`Peak: ${powers[maxIdx].toFixed(1)} dB`, peakX + 8, peakY - 4);
}

// 3. Constellation Diagram
function renderConstellation(c) {
  if (!c || !c.i) return;
  const ctx = canvasConstellation.getContext('2d');
  const w = canvasConstellation.width;
  const h = canvasConstellation.height;

  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const scale = Math.min(w, h) * 0.35;

  // Axes Crosshairs
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();

  // Reference Circles
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.arc(cx, cy, scale, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);

  // Ideal Constellation Points Overlay
  const showIdeal = document.getElementById('check-ideal-grid')?.checked;
  if (showIdeal && currentAnalysis) {
    const mod = currentAnalysis.active_configuration.modulation;
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 1;

    let idealPoints = [];
    if (mod.includes('QPSK')) {
      const q = 1 / Math.sqrt(2);
      idealPoints = [[q, q], [q, -q], [-q, q], [-q, -q]];
    } else if (mod.includes('BPSK')) {
      idealPoints = [[1, 0], [-1, 0]];
    } else if (mod.includes('8PSK')) {
      for (let k = 0; k < 8; k++) {
        idealPoints.push([Math.cos(2*Math.PI*k/8), Math.sin(2*Math.PI*k/8)]);
      }
    } else if (mod.includes('16QAM')) {
      const vals = [-3, -1, 1, 3].map(v => v / Math.sqrt(10));
      for (const vi of vals) {
        for (const vq of vals) idealPoints.push([vi, vq]);
      }
    }

    idealPoints.forEach(([ix, iy]) => {
      ctx.strokeRect(cx + ix * scale - 4, cy - iy * scale - 4, 8, 8);
    });
  }

  // Draw constellation points with glow
  const isGlow = document.getElementById('check-persistence')?.checked;
  ctx.fillStyle = isGlow ? 'rgba(0, 240, 255, 0.5)' : '#00f0ff';

  for (let k = 0; k < c.i.length; k++) {
    const px = cx + c.i[k] * scale;
    const py = cy - c.q[k] * scale;
    ctx.fillRect(px - 1, py - 1, 2.5, 2.5);
  }
}

// 4. Time Domain / Eye Diagram
function renderTimeDomain(td) {
  if (!td || !td.i) return;
  const ctx = canvasTimeDomain.getContext('2d');
  const w = canvasTimeDomain.width;
  const h = canvasTimeDomain.height;

  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();

  if (activeTimeMode === 'time') {
    // Waveform mode: In-Phase (Cyan) and Quadrature (Amber)
    const n = td.i.length;
    const stepX = w / n;

    // I(t)
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const x = k * stepX;
      const y = cy - td.i[k] * (cy * 0.7);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Q(t)
    ctx.strokeStyle = '#ffab00';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const x = k * stepX;
      const y = cy - td.q[k] * (cy * 0.7);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Legend
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#00f0ff';
    ctx.fillText('— In-Phase I(t)', 10, 16);
    ctx.fillStyle = '#ffab00';
    ctx.fillText('— Quadrature Q(t)', 120, 16);

  } else {
    // Eye Diagram mode: overlay overlapping slices of 2 symbol periods
    const sps = Math.max(4, Math.round(currentAnalysis?.parameters?.samples_per_symbol || 10));
    const eyeSpan = sps * 2;
    const nTraces = Math.floor(td.i.length / eyeSpan);

    ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
    ctx.lineWidth = 1.2;

    for (let tr = 0; tr < Math.min(nTraces, 30); tr++) {
      ctx.beginPath();
      for (let s = 0; s < eyeSpan; s++) {
        const idx = tr * eyeSpan + s;
        const x = (s / eyeSpan) * w;
        const y = cy - td.i[idx] * (cy * 0.75);
        if (s === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#00f0ff';
    ctx.fillText('Eye Diagram (2-Symbol Overlap)', 10, 16);
  }
}

// ---------------- Event Handlers ----------------

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId)?.classList.add('active');
    });
  });

  // Time vs Eye toggles
  document.getElementById('btn-toggle-time')?.addEventListener('click', (e) => {
    activeTimeMode = 'time';
    e.target.classList.add('active');
    document.getElementById('btn-toggle-eye')?.classList.remove('active');
    if (currentAnalysis) renderTimeDomain(currentAnalysis.visuals.time_domain);
  });

  document.getElementById('btn-toggle-eye')?.addEventListener('click', (e) => {
    activeTimeMode = 'eye';
    e.target.classList.add('active');
    document.getElementById('btn-toggle-time')?.classList.remove('active');
    if (currentAnalysis) renderTimeDomain(currentAnalysis.visuals.time_domain);
  });

  // Run Auto Analysis
  document.getElementById('btn-run-pipeline')?.addEventListener('click', runAnalysis);
  document.getElementById('btn-apply-fec')?.addEventListener('click', runAnalysis);
  document.getElementById('btn-apply-interleaver')?.addEventListener('click', runAnalysis);

  // Export JSON Report
  document.getElementById('btn-export-json')?.addEventListener('click', () => {
    window.location.href = '/api/export/json';
  });
  document.getElementById('btn-download-payload')?.addEventListener('click', () => {
    window.location.href = '/api/export/payload';
  });
}

function initColormapSelector() {
  const sel = document.getElementById('select-colormap');
  sel?.addEventListener('change', (e) => {
    selectedColormap = e.target.value;
    if (currentAnalysis) renderWaterfall(currentAnalysis.visuals.waterfall);
  });

  document.getElementById('check-ideal-grid')?.addEventListener('change', () => {
    if (currentAnalysis) renderConstellation(currentAnalysis.visuals.constellation);
  });
  document.getElementById('check-persistence')?.addEventListener('change', () => {
    if (currentAnalysis) renderConstellation(currentAnalysis.visuals.constellation);
  });
}

function initModals() {
  // Modal: Synthetic
  const modalSynth = document.getElementById('modal-synthetic');
  document.getElementById('btn-open-synthetic')?.addEventListener('click', () => {
    modalSynth.classList.add('open');
  });
  document.getElementById('btn-close-synthetic')?.addEventListener('click', () => {
    modalSynth.classList.remove('open');
  });
  document.getElementById('btn-cancel-synthetic')?.addEventListener('click', () => {
    modalSynth.classList.remove('open');
  });

  // Range slider labels
  const rangeSnr = document.getElementById('synth-snr');
  rangeSnr?.addEventListener('input', (e) => {
    document.getElementById('label-snr').textContent = `${e.target.value} dB`;
  });
  const rangeCfo = document.getElementById('synth-cfo');
  rangeCfo?.addEventListener('input', (e) => {
    document.getElementById('label-cfo').textContent = `${e.target.value} Hz`;
  });

  document.getElementById('btn-submit-synthetic')?.addEventListener('click', () => {
    const mod = document.getElementById('synth-mod').value;
    const snr = parseFloat(document.getElementById('synth-snr').value);
    const cfo = parseFloat(document.getElementById('synth-cfo').value);
    const pre = document.getElementById('synth-preamble').value;
    const fs = parseFloat(document.getElementById('synth-fs').value);
    const rs = parseFloat(document.getElementById('synth-rs').value);

    modalSynth.classList.remove('open');

    fetch('/api/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modulation: mod,
        snr_db: snr,
        cfo_hz: cfo,
        preamble: pre,
        sample_rate: fs,
        symbol_rate: rs,
        num_symbols: 3000
      })
    })
    .then(r => r.json())
    .then(data => {
      if (data.result) updateUI(data.result);
    })
    .catch(err => console.error("Error synthesizing:", err));
  });

  // Modal: Upload
  const modalUpload = document.getElementById('modal-upload');
  document.getElementById('btn-open-upload')?.addEventListener('click', () => {
    modalUpload.classList.add('open');
  });
  document.getElementById('btn-close-upload')?.addEventListener('click', () => {
    modalUpload.classList.remove('open');
  });
  document.getElementById('btn-cancel-upload')?.addEventListener('click', () => {
    modalUpload.classList.remove('open');
  });

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  let selectedFile = null;

  dropzone?.addEventListener('click', () => fileInput.click());
  fileInput?.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      selectedFile = e.target.files[0];
      document.getElementById('file-selected-name').textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
    }
  });

  dropzone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      selectedFile = e.dataTransfer.files[0];
      document.getElementById('file-selected-name').textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
    }
  });

  document.getElementById('btn-submit-upload')?.addEventListener('click', () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const arrayBuffer = e.target.result;
      const base64 = btoa(new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), ''));
      const fs = parseFloat(document.getElementById('upload-fs').value);
      const fmt = document.getElementById('upload-format').value;

      modalUpload.classList.remove('open');

      fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: selectedFile.name,
          content_base64: base64,
          sample_rate: fs,
          format_hint: fmt
        })
      })
      .then(r => r.json())
      .then(data => {
        if (data.result) updateUI(data.result);
      })
      .catch(err => console.error("Upload error:", err));
    };
    reader.readAsArrayBuffer(selectedFile);
  });
}
