import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

const MODEL_URL = "https://huggingface.co/jackjiangxinfa/demucs-onnx/resolve/main/model.onnx";

// モデルが要求する正確な入力サンプル数（343980）
const CHUNK_SIZE = 343980;
const INPUT_NAME = "input";
const OUTPUT_NAME = "output";

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

    const inputName = model.inputNames.includes(INPUT_NAME) ? INPUT_NAME : model.inputNames[0];
    const outputName = model.outputNames.includes(OUTPUT_NAME) ? OUTPUT_NAME : model.outputNames[0];

    const totalLength = left.length;
    const stems = createStemBuffers(totalLength);

    for (let start = 0; start < totalLength; start += CHUNK_SIZE) {
      const end = Math.min(start + CHUNK_SIZE, totalLength);
      const currentLen = end - start;

      const pct = Math.floor((start / totalLength) * 100);
      postMessage({ type: "progress", message: `4パート分離を計算中... (${pct}%)` });

      // [1, 2, 343980] 入力テンソル作成（末尾は0埋め）
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
        throw new Error(`出力 "${outputName}" が取得できませんでした`);
      }

      // 出力形状の解釈（例: [1, 6, 2, 343980] または [6, 2, 343980]）
      const dims = outTensor.dims;
      const numSamples = dims[dims.length - 1] || CHUNK_SIZE;
      const numChannels = dims[dims.length - 2] || 2;
      const numStems = dims.length >= 4 ? dims[1] : (dims.length === 3 ? dims[0] : 1);

      const copyLen = Math.min(currentLen, numSamples);
      const chStride = numSamples;
      const stemStride = numChannels * chStride;
      const data = outTensor.data;

      // 6ステム標準順 [0: drums, 1: bass, 2: other, 3: vocals, 4: guitar, 5: piano] を想定
      // vocals / bass / drums 以外はすべて「④その他（stems.other）」に加算
      for (let s = 0; s < numStems; s++) {
        const baseL = s * stemStride;
        const baseR = numChannels > 1 ? baseL + chStride : baseL;

        let target = stems.other;
        if (numStems >= 6) {
          if (s === 0) target = stems.drums;
          else if (s === 1) target = stems.bass;
          else if (s === 3) target = stems.vocals;
          else target = stems.other; // 2:other, 4:guitar, 5:piano
        } else if (numStems === 4) {
          // 4ステム順 [0: drums, 1: bass, 2: other, 3: vocals] を想定
          if (s === 0) target = stems.drums;
          else if (s === 1) target = stems.bass;
          else if (s === 3) target = stems.vocals;
          else target = stems.other;
        }

        for (let i = 0; i < copyLen; i++) {
          target.left[start + i] += data[baseL + i];
          target.right[start + i] += data[baseR + i];
        }
      }

      inputTensor.dispose();
      outTensor.dispose();
    }

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
      message: `推論エラー: ${err.message}`
    });
  }
};
