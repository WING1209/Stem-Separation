import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

const MODEL_URL =
  "https://huggingface.co/jackjiangxinfa/demucs-onnx/resolve/main/model.onnx";

// モデルが要求する入力サンプル数（343980）
const CHUNK_SIZE = 343980;
const INPUT_NAME = "input";
const OUTPUT_NAME = "output";

// 6 パートの並び順（標準的な 6 ステム構成: drums, bass, other, vocals, guitar, piano）
const STEM_KEYS = ["drums", "bass", "other", "vocals", "guitar", "piano"];

let session = null;

async function initModel() {
  if (session) return session;

  postMessage({
    type: "progress",
    message: "AI モデル（約 304MB）を読み込み中..."
  });

  // 必要に応じてスレッド数を制御（環境依存のトラブル切り分け用）
  // ort.env.wasm.numThreads = 1;

  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"]
  });

  console.log("[Demucs Worker] inputNames:", session.inputNames);
  console.log("[Demucs Worker] outputNames:", session.outputNames);

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
    const v = buf[i];
    if (Number.isNaN(v)) {
      buf[i] = 0;
    } else if (v > 1) {
      buf[i] = 1;
    } else if (v < -1) {
      buf[i] = -1;
    }
  }
}

/**
 * 出力テンソル形状 [batch, stems, channels, samples] を前提に値を読み出す
 * もし形状が異なる場合は console.warn で検知できるようにする
 */
function extractStemSample(data, dims, stemIdx, channelIdx, sampleIdx) {
  if (dims.length === 4) {
    // [B, S, C, T]
    const [, numStems, numChannels, numSamples] = dims;
    const idx =
      ((0 * numStems + stemIdx) * numChannels + channelIdx) * numSamples +
      sampleIdx;
    return data[idx];
  }
  if (dims.length === 3) {
    // [S, C, T] （バッチ次元なし想定）
    const [numStems, numChannels, numSamples] = dims;
    const idx = (stemIdx * numChannels + channelIdx) * numSamples + sampleIdx;
    return data[idx];
  }
  throw new Error(`未対応の出力テンソル形状です: [${dims.join(", ")}]`);
}

self.onmessage = async (e) => {
  try {
    const { sampleRate, left, right } = e.data ?? {};

    if (!(left instanceof Float32Array) || !(right instanceof Float32Array)) {
      throw new Error("left / right は Float32Array で渡してください");
    }
    if (left.length === 0 || left.length !== right.length) {
      throw new Error("left と right の長さが不正です");
    }

    const model = await initModel();
    const inputName = model.inputNames.includes(INPUT_NAME)
      ? INPUT_NAME
      : model.inputNames[0];
    const outputName = model.outputNames.includes(OUTPUT_NAME)
      ? OUTPUT_NAME
      : model.outputNames[0];

    const totalLength = left.length;
    const stems = createStemBuffers(totalLength);

    // ゼロ埋め用の入力バッファをループ外で確保して GC 負荷を軽減
    const inputData = new Float32Array(2 * CHUNK_SIZE);

    for (let start = 0; start < totalLength; start += CHUNK_SIZE) {
      const end = Math.min(start + CHUNK_SIZE, totalLength);
      const currentLen = end - start;
      const pct = Math.floor((start / totalLength) * 100);

      postMessage({
        type: "progress",
        message: `6 パート分離を計算中... (${pct}%)`
      });

      // 残りサンプルが CHUNK_SIZE 未満のときは後半を 0 埋め
      inputData.fill(0);
      for (let i = 0; i < currentLen; i++) {
        inputData[i] = left[start + i];
        inputData[CHUNK_SIZE + i] = right[start + i];
      }

      const inputTensor = new ort.Tensor("float32", inputData, [
        1,
        2,
        CHUNK_SIZE
      ]);

      let outTensor = null;
      try {
        const feeds = { [inputName]: inputTensor };
        const results = await model.run(feeds, [outputName]);
        outTensor = results[outputName];

        if (!outTensor) {
          throw new Error(`出力 "${outputName}" が取得できませんでした`);
        }

        const dims = outTensor.dims;
        if (start === 0) {
          console.log("[Demucs Worker] output dims:", dims);
        }

        const numSamples = dims[dims.length - 1] ?? CHUNK_SIZE;
        const numChannels = dims[dims.length - 2] ?? 2;
        const numStems =
          dims.length >= 4 ? dims[1] : dims.length === 3 ? dims[0] : 1;

        if (numStems !== STEM_KEYS.length) {
          console.warn(
            `[Demucs Worker] 出力ステム数(${numStems})が STEM_KEYS(${STEM_KEYS.length}) と一致しません`
          );
        }

        const copyLen = Math.min(currentLen, numSamples);
        const data = outTensor.data;
        const targetStems = Math.min(numStems, STEM_KEYS.length);

        for (let s = 0; s < targetStems; s++) {
          const key = STEM_KEYS[s];
          const outL = stems[key].left;
          const outR = stems[key].right;

          for (let i = 0; i < copyLen; i++) {
            outL[start + i] = extractStemSample(data, dims, s, 0, i);
            outR[start + i] =
              numChannels > 1
                ? extractStemSample(data, dims, s, 1, i)
                : outL[start + i];
          }
        }
      } finally {
        inputTensor.dispose();
        outTensor?.dispose();
      }
    }

    for (const key of STEM_KEYS) {
      clampBuffer(stems[key].left);
      clampBuffer(stems[key].right);
    }

    postMessage({
      type: "progress",
      message: "6 パート分の WAV ファイルを生成中..."
    });

    // Transferable の重複を防止
    const transferSet = new Set();
    for (const key of STEM_KEYS) {
      transferSet.add(stems[key].left.buffer);
      transferSet.add(stems[key].right.buffer);
    }

    postMessage(
      {
        type: "done",
        sampleRate,
        stems
      },
      Array.from(transferSet)
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[Demucs Worker] Error:", err);
    postMessage({
      type: "error",
      message: `推論エラー: ${message}`
    });
  }
};
