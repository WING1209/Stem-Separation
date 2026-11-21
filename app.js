// スマホ判定 (簡易チェック)
const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && window.innerWidth <= 900;
if (!isMobile) {
  document.getElementById("app").style.display = "none";
  document.getElementById("pc-block").style.display = "block";
  throw new Error("Non-mobile access blocked");
}

const TARGET_SAMPLE_RATE = 44100;
const STEM_KEYS = ["drums", "bass", "other", "vocals", "guitar", "piano"];

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");

let selectedFile = null;
const worker = new Worker("worker.js", { type: "module" });

// MP3 / WAV のみ許可
function isAllowedAudio(file) {
  if (!file) return false;
  const nameOk = /\.(mp3|wav)$/i.test(file.name);
  const typeOk = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav"].includes(
    file.type.toLowerCase()
  );
  return nameOk || typeOk;
}

// ダウンロードリンクを初期状態(無効)に戻し、既存の ObjectURL を解放
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

// 任意のデコード済み AudioBuffer を 44100Hz 2ch (ステレオ) へ自動変換
async function normalizeTo44100Stereo(decodedBuffer) {
  const numChannels = decodedBuffer.numberOfChannels;
  if (numChannels < 1) {
    throw new Error("有効な音声チャンネルが見つかりませんでした。");
  }

  const needsResample = decodedBuffer.sampleRate !== TARGET_SAMPLE_RATE;
  const needsChannelConvert = numChannels !== 2;

  if (!needsResample && !needsChannelConvert) {
    return decodedBuffer;
  }

  const targetLength = Math.ceil(decodedBuffer.duration * TARGET_SAMPLE_RATE);
  const offlineCtx = new OfflineAudioContext(2, targetLength, TARGET_SAMPLE_RATE);

  const inputForRender = offlineCtx.createBuffer(
    Math.min(numChannels, 2),
    decodedBuffer.length,
    decodedBuffer.sampleRate
  );

  if (numChannels === 1) {
    const mono = decodedBuffer.getChannelData(0);
    inputForRender.copyToChannel(mono, 0);
    inputForRender.copyToChannel(mono, 1);
  } else {
    inputForRender.copyToChannel(decodedBuffer.getChannelData(0), 0);
    inputForRender.copyToChannel(decodedBuffer.getChannelData(1), 1);
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
    fileSub.textContent = "対応形式: MP3 / WAV (44100Hz・ステレオ)";
    statusEl.textContent = "エラー: MP3 または WAVファイルのみ対応しています。";
    return;
  }

  selectedFile = file;
  runBtn.disabled = false;
  const mb = (file.size / 1024 / 1024).toFixed(2);
  fileSub.textContent = `${file.name} (${mb} MB)`;
  statusEl.textContent = `選択完了: ${file.name} - 「6 パート分離を実行」を押してください。`;
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  resetDownloadLinks();
  statusEl.textContent = "音声データを読み込み中...";

  try {
    let arrayBuffer = await selectedFile.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    let audioBuffer;

    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    } finally {
      arrayBuffer = null;
      if (typeof audioCtx.close === "function") {
        await audioCtx.close().catch(() => {});
      }
    }

    const needsConvert =
      audioBuffer.sampleRate !== TARGET_SAMPLE_RATE || audioBuffer.numberOfChannels !== 2;
    if (needsConvert) {
      statusEl.textContent = `44100Hz・ステレオに変換中... (元: ${audioBuffer.sampleRate}Hz / ${audioBuffer.numberOfChannels}ch)`;
    }

    let normalizedBuffer = await normalizeTo44100Stereo(audioBuffer);
    audioBuffer = null;

    const left = new Float32Array(normalizedBuffer.getChannelData(0));
    const right = new Float32Array(normalizedBuffer.getChannelData(1));
    const sampleRate = normalizedBuffer.sampleRate;
    normalizedBuffer = null;

    statusEl.textContent = "バックグラウンドで6パート分離中...";
    worker.postMessage(
      {
        sampleRate,
        left,
        right
      },
      [left.buffer, right.buffer]
    );
  } catch (err) {
    statusEl.textContent = `読み込み・変換エラー: ${err.message}`;
    runBtn.disabled = false;
  }
});

// Float32Array (ステレオ 2ch) を WAV Blob に変換
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
  view.setUint16(20, 1, true); // PCM
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

worker.onmessage = (e) => {
  const { type, message, stems, sampleRate } = e.data;
  if (type === "progress") {
    statusEl.textContent = message;
  } else if (type === "done") {
    STEM_KEYS.forEach((k) => {
      const link = document.getElementById(`dl-${k}`);
      if (link && stems[k]) {
        const blob = encodeWav(stems[k].left, stems[k].right, sampleRate);
        link.href = URL.createObjectURL(blob);
        link.classList.remove("disabled");
      }
    });
    statusEl.textContent = "完了! 6パートそれぞれのWAVを保存できます。";
    runBtn.disabled = false;
  } else if (type === "error") {
    statusEl.textContent = `分離エラー: ${message || "処理中にエラーが発生しました。"}`;
    runBtn.disabled = false;
  }
};

worker.onerror = (err) => {
  statusEl.textContent = `ワーカーエラー: ${err.message || "予期しないエラーが発生しました。"}`;
  runBtn.disabled = false;
};
