import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

// ご提示いただいたHugging Faceの直接ダウンロードURL
const MODEL_URL = "https://huggingface.co/jackjiangxinfa/demucs-onnx/resolve/main/model.onnx";

// 1回あたりの入力サンプル数（44100Hz × 6秒 = 264600 想定。モデル仕様に合わせて調整）
const CHUNK_SIZE = 264600;

// モデルの入出力テンソル名（必要に応じて実際のモデル名に合わせて変更）
const INPUT_NAME = "input";
const OUTPUT_MAP = {
  vocals: "vocals",
  bass:   "bass",
  drums:  "drums",
  other:  "other",
  guitar: "guitar",
  piano:  "piano"
};

let session = null;

async function initModel() {
  if (session) return session;
  postMessage({ type: "progress", message: "AIモデル（約304MB）を読み込み中..." });

  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"]
  });
  return session;
}

function createStemBuffers(totalLength) {
  const keys = ["vocals", "bass", "drums", "other"];
  const result = {};
  for (const k of keys) {
    result[k] = {
      left: new Float32Array(totalLength),
      right: new Float32Array(totalLength)
    };
  }
  return result;
}

// テンソル [1, 2, chunkLen] を指定範囲に加算するヘルパー
function addTensorToBuffer(tensor, targetLeft, targetRight, startOffset, currentLen, chunkLen) {
  if (!tensor) return;
  const data = tensor.data;
  for (let i = 0; i < currentLen; i++) {
    targetLeft[startOffset + i] += data[i];
    targetRight[startOffset + i] += data[chunkLen + i];
  }
}

// 音割れ防止（-1.0 〜 +1.0 に収める）
function clampBuffer(buf) {
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] > 1) buf[i] = 1;
    else if (buf[i] < -1) buf[i] = -1;
  }
}

self.onmessage = async (e) => {
  try {
    const { sampleRate, left, right } = e.data;
    const model = await initModel();

    // 実際にモデルが持つ入力名・出力名を確認（未設定時のフォールバック用）
    const inputName = model.inputNames.includes(INPUT_NAME) ? INPUT_NAME : model.inputNames[0];
    const availableOutputs = model.outputNames;
    const fetches = Object.values(OUTPUT_MAP).filter((name) => name && availableOutputs.includes(name));

    const totalLength = left.length;
    const stems = createStemBuffers(totalLength);

    for (let start = 0; start < totalLength; start += CHUNK_SIZE) {
      const end = Math.min(start + CHUNK_SIZE, totalLength);
      const currentLen = end - start;

      const pct = Math.floor((start / totalLength) * 100);
      postMessage({ type: "progress", message: `6パート推論→4パート統合中... (${pct}%)` });

      // 入力配列 [1, 2, CHUNK_SIZE]（ステレオ2ch・末尾0埋め）
      const inputData = new Float32Array(2 * CHUNK_SIZE);
      for (let i = 0; i < currentLen; i++) {
        inputData[i] = left[start + i];
        inputData[CHUNK_SIZE + i] = right[start + i];
      }

      const inputTensor = new ort.Tensor("float32", inputData, [1, 2, CHUNK_SIZE]);
      const feeds = { [inputName]: inputTensor };
      const results = await model.run(feeds, fetches.length > 0 ? fetches : undefined);

      // パターンA: 6つの別々の出力テンソルとして返る場合
      if (fetches.length > 0) {
        addTensorToBuffer(results[OUTPUT_MAP.vocals], stems.vocals.left, stems.vocals.right, start, currentLen, CHUNK_SIZE);
        addTensorToBuffer(results[OUTPUT_MAP.bass],   stems.bass.left,   stems.bass.right,   start, currentLen, CHUNK_SIZE);
        addTensorToBuffer(results[OUTPUT_MAP.drums],  stems.drums.left,  stems.drums.right,  start, currentLen, CHUNK_SIZE);

        // ④その他に other + guitar + piano を合算
        addTensorToBuffer(results[OUTPUT_MAP.other],  stems.other.left,  stems.other.right,  start, currentLen, CHUNK_SIZE);
        addTensorToBuffer(results[OUTPUT_MAP.guitar], stems.other.left,  stems.other.right,  start, currentLen, CHUNK_SIZE);
        addTensorToBuffer(results[OUTPUT_MAP.piano],  stems.other.left,  stems.other.right,  start, currentLen, CHUNK_SIZE);
      } else {
        // パターンB: 1つの出力テンソル [1, 6, 2, CHUNK_SIZE] で返る場合のフォールバック
        const firstOut = results[availableOutputs[0]];
        if (!firstOut) {
          throw new Error(`出力テンソルが見つかりません (outputs: ${availableOutputs.join(", ")})`);
        }
        const dims = firstOut.dims; // 期待例: [1, 6, 2, samples]
        const numStems = dims[1] || 6;
        const actualChunk = dims[dims.length - 1] || CHUNK_SIZE;
        const copyLen = Math.min(currentLen, actualChunk);
        const chStride = actualChunk;
        const stemStride = 2 * chStride;
        const data = firstOut.data;

        // 一般的順 [drums, bass, other, vocals, guitar, piano] を想定した振り分け
        for (let s = 0; s < numStems; s++) {
          const baseL = s * stemStride;
          const baseR = baseL + chStride;

          let target = stems.other;
          if (s === 0) target = stems.drums;
          else if (s === 1) target = stems.bass;
          else if (s === 3) target = stems.vocals;

          for (let i = 0; i < copyLen; i++) {
            target.left[start + i] += data[baseL + i];
            target.right[start + i] += data[baseR + i];
          }
        }
      }

      inputTensor.dispose();
      for (const name of Object.keys(results)) {
        if (results[name]) results[name].dispose();
      }
    }

    // 4パートそれぞれの音割れ防止
    for (const key of ["vocals", "bass", "drums", "other"]) {
      clampBuffer(stems[key].left);
      clampBuffer(stems[key].right);
    }

    postMessage({ type: "progress", message: "WAVファイルを生成中..." });

    postMessage(
      {
        type: "done",
        sampleRate,
        stems
      },
      [
        stems.vocals.left.buffer, stems.vocals.right.buffer,
        stems.bass.left.buffer,   stems.bass.right.buffer,
        stems.drums.left.buffer,  stems.drums.right.buffer,
        stems.other.left.buffer,  stems.other.right.buffer
      ]
    );
  } catch (err) {
    postMessage({
      type: "progress",
      message: `推論エラー: ${err.message}（inputs: ${session ? session.inputNames.join(",") : "-"} / outputs: ${session ? session.outputNames.join(",") : "-"}）`
    });
  }
};
