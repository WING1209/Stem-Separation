// スマホ判定 (簡易チェック)
const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && window.innerWidth <= 900;
if (!isMobile) {
  document.getElementById("app").style.display = "none";
  document.getElementById("pc-block").style.display = "block";
  throw new Error("Non-mobile access blocked");
}

const TARGET_SAMPLE_RATE = 44100;
const MAX_SECONDS = 30;
const MAX_SAMPLES = TARGET_SAMPLE_RATE * MAX_SECONDS;
const STEM_KEYS = ["drums", "bass", "other", "vocals", "guitar", "piano"];

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");

let selectedFile = null;
const worker = new Worker("worker.js", { type: "module" });

// 画面にログを追記するヘルパー
function logStatus(msg) {
  const now = new Date().toLocaleTimeString("ja-JP");
  statusEl.textContent = `[${now}] ${msg}\n` + statusEl.textContent;
  statusEl.style.whiteSpace = "pre-line";
}

logStatus("ページ初期化完了（Worker準備OK）");

function isAllowedAudio(file) {
  if (!file) return false;
  const nameOk = /\.(mp3|wav)$/i.test(file.name);
  const typeOk = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav"].includes(
    file.type.toLowerCase()
  );
  return nameOk || typeOk;
}

function resetDownloadLinks() {
  STEM_KEYS.forEach((k) => {
    const link = document.getElementById(`dl-${k}`);
    if (link) {
      if (link.href && link.href.startsWith("blob:")) {
        URL.revokeObjectURL(link.href);
      }
      link.classList.add("disabled");
      link.removeAttribute("href");
    }
  });
}

async function normalizeTo44100Stereo(decodedBuffer) {
  const numChannels = decodedBuffer.numberOfChannels;
  if (numChannels < 1) {
    throw new Error("有効な音声チャンネルが見つかりませんでした。");
  }

  const clampedDuration = Math.min(decodedBuffer.duration, MAX_SECONDS);
  const targetLength = Math.min(Math.ceil(clampedDuration * TARGET_SAMPLE_RATE), MAX_SAMPLES);
  const sourceLength = Math.min(
    decodedBuffer.length,
    Math.ceil(clampedDuration * decodedBuffer.sampleRate)
  );

  const offlineCtx = new OfflineAudioContext(2, targetLength, TARGET_SAMPLE_RATE);
  const inputForRender = offlineCtx.createBuffer(
    Math.min(numChannels, 2),
    sourceLength,
    decodedBuffer.sampleRate
  );

  if (numChannels === 1) {
    const mono = decodedBuffer.getChannelData(0).subarray(0, sourceLength);
    inputForRender.copyToChannel(mono, 0);
    inputForRender.copyToChannel(mono, 1);
  } else {
    const ch0 = decodedBuffer.getChannelData(0).subarray(0, sourceLength);
    const ch1 = decodedBuffer.getChannelData(1).subarray(0, sourceLength);
    inputForRender.copyToChannel(ch0, 0);
    inputForRender.copyToChannel(ch1, 1);
  }

  const source = offlineCtx.createBufferSource();
  source.buffer = inputForRender;
  source.connect(offlineCtx.destination);
  source.start(0);

  return await offlineCtx.startRendering();
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  resetDownloadLinks();

  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    fileSub.textContent = "対応形式: MP3 / WAV (44100Hz・ステレオ / 先頭30秒まで)";
    logStatus("エラー: MP3 または WAVファイルのみ対応しています。");
    return;
  }

  selectedFile = file;
  runBtn.disabled = false;
  const mb = (file.size / 1024 / 1024).toFixed(2);
  fileSub.textContent = `${file.name} (${mb} MB)`;
  logStatus(`選択完了: ${file.name} (${mb} MB)`);
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  resetDownloadLinks();
  logStatus("1. 音声ファイル読み込み開始...");

  try {
    let arrayBuffer = await selectedFile.arrayBuffer();
    logStatus(`2. ArrayBuffer取得完了 (${(arrayBuffer.byteLength / 1024 / 1024).toFixed(2)} MB)`);

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    let audioBuffer;

    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      logStatus(`3. デコード完了 (${audioBuffer.duration.toFixed(1)}秒 / ${audioBuffer.sampleRate}Hz)`);
    } finally {
      arrayBuffer = null;
      if (typeof audioCtx.close === "function") {
        await audioCtx.close().catch(() => {});
      }
    }

    let normalizedBuffer = await normalizeTo44100Stereo(audioBuffer);
    audioBuffer = null;
    logStatus("4. 44100Hzステレオ変換完了");

    const left = new Float32Array(normalizedBuffer.getChannelData(0));
    const right = new Float32Array(normalizedBuffer.getChannelData(1));
    const sampleRate = normalizedBuffer.sampleRate;
    normalizedBuffer = null;

    logStatus("5. Workerへ音声データを送信 -> モデル読込待ち");
    worker.postMessage(
      {
        sampleRate,
        left,
        right
      },
      [left.buffer, right.buffer]
    );
  } catch (err) {
    logStatus(`読み込み・変換エラー: ${err.message}`);
    runBtn.disabled = false;
  }
});

function encodeWav(chL, chR, sampleRate) {
  const numChannels = 2;
  const length = chL.length;
  const buffer = new ArrayBuffer(44 + length * numChannels * 2);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + length * numChannels * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, length * numChannels * 2, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sL = Math.max(-1, Math.min(1, chL[i]));
    const sR = Math.max(-1, Math.min(1, chR[i]));
    view.setInt16(offset, sL < 0 ? sL * 0x8000 : sL * 0x7fff, true);
    view.setInt16(offset + 2, sR < 0 ? sR * 0x8000 : sR * 0x7fff, true);
    offset += 4;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

worker.onmessage = async (e) => {
  const { type, message, stems, sampleRate } = e.data;
  if (type === "progress") {
    logStatus(`Worker: ${message}`);
  } else if (type === "done") {
    for (let i = 0; i < STEM_KEYS.length; i++) {
      const k = STEM_KEYS[i];
      const link = document.getElementById(`dl-${k}`);
      if (link && stems[k]) {
        logStatus(`WAV生成中... (${i + 1}/${STEM_KEYS.length})`);
        const blob = encodeWav(stems[k].left, stems[k].right, sampleRate);
        link.href = URL.createObjectURL(blob);
        link.classList.remove("disabled");
        stems[k] = null;
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    logStatus("完了! 6パートそれぞれのWAVを保存できます。");
    runBtn.disabled = false;
  } else if (type === "error") {
    logStatus(`分離エラー: ${message || "処理中にエラーが発生しました。"}`);
    runBtn.disabled = false;
  }
};

worker.onerror = (err) => {
  logStatus(`ワーカーエラー: ${err.message || "予期しないエラーが発生しました。"}`);
  runBtn.disabled = false;
};
