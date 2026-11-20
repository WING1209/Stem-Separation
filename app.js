// スマホ判定（簡易チェック）
const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && window.innerWidth <= 900;
if (!isMobile) {
  document.getElementById("app").style.display = "none";
  document.getElementById("pc-block").style.display = "block";
  throw new Error("Non-mobile access blocked");
}

const fileInput = document.getElementById("audioFile");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");

let selectedFile = null;
const worker = new Worker("worker.js", { type: "module" });

// MP3 / WAV のみ許可
function isAllowedAudio(file) {
  if (!file) return false;
  const nameOk = /\.(mp3|wav)$/i.test(file.name);
  const typeOk = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav"].includes(file.type.toLowerCase());
  return nameOk || typeOk;
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    statusEl.textContent = "エラー: MP3またはWAVファイルのみ対応しています。";
    return;
  }
  selectedFile = file;
  runBtn.disabled = false;
  statusEl.textContent = `選択中: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  statusEl.textContent = "音声データを読み込み中...";

  const arrayBuffer = await selectedFile.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // 左右チャンネル（モノラル時は複製）をFloat32ArrayとしてWorkerへ転送
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : new Float32Array(left);

  statusEl.textContent = "バックグラウンドで分離処理中...";
  worker.postMessage({
    sampleRate: audioBuffer.sampleRate,
    left,
    right
  }, [left.buffer, right.buffer]);
});

// Float32Array(ステレオ2ch) を WAV Blob に変換
function encodeWav(chL, chR, sampleRate) {
  const numChannels = 2;
  const length = chL.length;
  const buffer = new ArrayBuffer(44 + length * numChannels * 2);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
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
    const keys = ["vocals", "bass", "drums", "other"];
    keys.forEach((k) => {
      const link = document.getElementById(`dl-${k}`);
      const blob = encodeWav(stems[k].left, stems[k].right, sampleRate);
      link.href = URL.createObjectURL(blob);
      link.classList.remove("disabled");
    });
    statusEl.textContent = "完了！各パートのWAVを保存できます。";
    runBtn.disabled = false;
  }
};