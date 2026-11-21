// スマホ判定（簡易チェック）
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
  const typeOk = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav"].includes(file.type.toLowerCase());
  return nameOk || typeOk;
}

// ダウンロードリンクを初期状態（無効）に戻す
function resetDownloadLinks() {
  STEM_KEYS.forEach((k) => {
    const link = document.getElementById(`dl-${k}`);
    if (link) {
      link.classList.add("disabled");
      link.removeAttribute("href");
    }
  });
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  resetDownloadLinks();

  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    fileSub.textContent = "対応形式：MP3 / WAV（44100Hz・ステレオ）";
    statusEl.textContent = "エラー: MP3またはWAVファイルのみ対応しています。";
    return;
  }

  selectedFile = file;
  runBtn.disabled = false;
  const mb = (file.size / 1024 / 1024).toFixed(2);
  fileSub.textContent = `${file.name} (${mb} MB)`;
  statusEl.textContent = `選択完了: ${file.name} — 「6パート分離を実行」を押してください。`;
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  resetDownloadLinks();
  statusEl.textContent = "音声データを読み込み中...";

  try {
    const arrayBuffer = await selectedFile.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    if (audioBuffer.sampleRate !== TARGET_SAMPLE_RATE) {
      statusEl.textContent = "サンプリングレートがちがいます";
      runBtn.disabled = false;
      return;
    }

    if (audioBuffer.numberOfChannels < 2) {
      statusEl.textContent = "エラー: ステレオ（2ch）音源のみ対応しています。";
      runBtn.disabled = false;
      return;
    }

    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.getChannelData(1);

    statusEl.textContent = "バックグラウンドで6パート分離中...";
    worker.postMessage({
      sampleRate: audioBuffer.sampleRate,
      left,
      right
    }, [left.buffer, right.buffer]);
  } catch (err) {
    statusEl.textContent = `読み込みエラー: ${err.message}`;
    runBtn.disabled = false;
  }
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
    statusEl.textContent = "完了！6パートそれぞれのWAVを保存できます。";
    runBtn.disabled = false;
  }
};
