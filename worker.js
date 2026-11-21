import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

// iPhone / WebKit 向け省メモリ設定
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

// ボーカル抽出モデル (FP16軽量版)
const MODEL_URL = "https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx";

// モデルが要求する入力サンプル数 (343980)
const CHUNK_SIZE = 343980;
const INPUT_NAME = "input";
const OUTPUT_NAME = "output";

let session = null;

async function initModel() {
  if (session) return session;
  postMessage({ type: "progress", message: "軽量AIモデルを読み込み中..." });
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    enableMemPattern: false,
    enableCpuMemArena: false
  });
  return session;
}

function clampVal(v) {
  return v > 1 ? 1 : (v < -1 ? -1 : v);
}

self.onmessage = async (e) => {
  let { sampleRate, left, right } = e.data;
  try {
    const model = await initModel();
    const inputName = model.inputNames.includes(INPUT_NAME) ? INPUT_NAME : model.inputNames[0];
    const outputName = model.outputNames.includes(OUTPUT_NAME) ? OUTPUT_NAME : model.outputNames[0];

    const totalLength = left.length;
    const vocalsL = new Float32Array(totalLength);
    const vocalsR = new Float32Array(totalLength);
    const instL = new Float32Array(totalLength);
    const instR = new Float32Array(totalLength);

    const inputData = new Float32Array(2 * CHUNK_SIZE);

    for (let start = 0; start < totalLength; start += CHUNK_SIZE) {
      const end = Math.min(start + CHUNK_SIZE, totalLength);
      const currentLen = end - start;
      const pct = Math.floor((start / totalLength) * 100);
      postMessage({ type: "progress", message: `2パート分離を計算中... (${pct}%)` });

      inputData.fill(0);
      for (let i = 0; i < currentLen; i++) {
        inputData[i] = left[start + i];
        inputData[CHUNK_SIZE + i] = right[start + i];
      }

      const inputTensor = new ort.Tensor("float32", inputData, [1, 2, CHUNK_SIZE]);
      let outTensor = null;

      try {
        const feeds = { [inputName]: inputTensor };
        const results = await model.run(feeds, [outputName]);
        outTensor = results[outputName];

        if (!outTensor) {
          throw new Error(`出力 "${outputName}" が取得できませんでした`);
        }

        const dims = outTensor.dims;
        const numSamples = dims[dims.length - 1] || CHUNK_SIZE;
        const numChannels = dims[dims.length - 2] || 2;
        const copyLen = Math.min(currentLen, numSamples);
        const data = outTensor.data;

        const baseL = 0;
        const baseR = numChannels > 1 ? numSamples : 0;

        for (let i = 0; i < copyLen; i++) {
          const vL = clampVal(data[baseL + i]);
          const vR = clampVal(data[baseR + i]);
          vocalsL[start + i] = vL;
          vocalsR[start + i] = vR;

          // 元音源から歌を差し引いて「それ以外」を生成
          instL[start + i] = clampVal(left[start + i] - vL);
          instR[start + i] = clampVal(right[start + i] - vR);
        }
      } finally {
        inputTensor.dispose();
        if (outTensor) outTensor.dispose();
      }
    }

    // 入力バッファを解放
    left = null;
    right = null;

    postMessage({ type: "progress", message: "2パート分のWAVファイルを生成中..." });

    const stems = {
      vocals: { left: vocalsL, right: vocalsR },
      instrumental: { left: instL, right: instR }
    };

    const transferBuffers = [
      vocalsL.buffer,
      vocalsR.buffer,
      instL.buffer,
      instR.buffer
    ];

    postMessage(
      {
        type: "done",
        sampleRate,
        stems
      },
      transferBuffers
    );
  } catch (err) {
    postMessage({
      type: "error",
      message: err.message || String(err)
    });
  }
};
