import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

const MODEL_URL =
  "https://github.com/WING1209/Stem-Separation/releases/download/v1.0-fp16/model_fp16.onnx";

// モデルが要求する入力サンプル数 (343980)
const CHUNK_SIZE = 343980;
const INPUT_NAME = "input";
const OUTPUT_NAME = "output";
const STEM_KEYS = ["drums", "bass", "other", "vocals", "guitar", "piano"];

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

let session = null;

async function initModel() {
  if (session) return session;

  postMessage({ type: "progress", message: "AIモデルをダウンロード中..." });
  const response = await fetch(MODEL_URL);
  if (!response.ok) {
    throw new Error(`モデルの取得に失敗しました (HTTP ${response.status})`);
  }

  postMessage({ type: "progress", message: "モデルデータをメモリへ展開中..." });
  const buffer = await response.arrayBuffer();
  let modelBytes = new Uint8Array(buffer);

  postMessage({
    type: "progress",
    message: `AIセッションを初期化中... (${(modelBytes.byteLength / 1024 / 1024).toFixed(1)} MB)`
  });

  try {
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "disabled",
      enableCpuMemArena: false,
      enableMemPattern: false
    });
  } finally {
    modelBytes = null;
  }

  postMessage({ type: "progress", message: "AIモデルの準備が完了しました!" });
  return session;
}

function createStemBuffers(totalLength) {
  const result = {};
  for (const k of STEM_KEYS) {
    result[k] = {
      left: new Float32Array(totalLength),
      right: new Float32Array(totalLength)
    };
  }
  return result;
}

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

    const inputName = model.inputNames.includes(INPUT_NAME)
      ? INPUT_NAME
      : model.inputNames[0];
    const outputName = model.outputNames.includes(OUTPUT_NAME)
      ? OUTPUT_NAME
      : model.outputNames[0];

    const totalLength = left.length;
    const stems = createStemBuffers(totalLength);

    for (let start = 0; start < totalLength; start += CHUNK_SIZE) {
      const end = Math.min(start + CHUNK_SIZE, totalLength);
      const currentLen = end - start;
      const pct = Math.floor((start / totalLength) * 100);

      postMessage({
        type: "progress",
        message: `6 パート分離を計算中... (${pct}%)`
      });

      const inputData = new Float32Array(2 * CHUNK_SIZE);
      for (let i = 0; i < currentLen; i++) {
        inputData[i] = left[start + i];
        inputData[CHUNK_SIZE + i] = right[start + i];
      }

      const inputTensor = new ort.Tensor("float32", inputData, [1, 2, CHUNK_SIZE]);
      const feeds = { [inputName]: inputTensor };
      const results = await model.run(feeds, [outputName]);
      const outTensor = results[outputName];

      if (!outTensor) {
        inputTensor.dispose();
        throw new Error(`出力 "${outputName}" が取得できませんでした`);
      }

      const dims = outTensor.dims;
      const numSamples = dims[dims.length - 1] || CHUNK_SIZE;
      const numChannels = dims[dims.length - 2] || 2;
      const numStems =
        dims.length >= 4 ? dims[1] : dims.length === 3 ? dims[0] : 1;

      const copyLen = Math.min(currentLen, numSamples);
      const chStride = numSamples;
      const stemStride = numChannels * chStride;
      const data = outTensor.data;

      for (let s = 0; s < Math.min(numStems, STEM_KEYS.length); s++) {
        const key = STEM_KEYS[s];
        const baseL = s * stemStride;
        const baseR = numChannels > 1 ? baseL + chStride : baseL;
        for (let i = 0; i < copyLen; i++) {
          stems[key].left[start + i] = data[baseL + i];
          stems[key].right[start + i] = data[baseR + i];
        }
      }

      inputTensor.dispose();
      outTensor.dispose();
    }

    for (const key of STEM_KEYS) {
      clampBuffer(stems[key].left);
      clampBuffer(stems[key].right);
    }

    postMessage({ type: "progress", message: "6 パート分のWAVデータを転送中..." });

    const transferBuffers = [];
    for (const key of STEM_KEYS) {
      transferBuffers.push(stems[key].left.buffer, stems[key].right.buffer);
    }

    postMessage({ type: "done", sampleRate, stems }, transferBuffers);
  } catch (err) {
    postMessage({
      type: "error",
      message: `推論エラー: ${err.message}`
    });
  }
};
