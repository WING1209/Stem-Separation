import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

// ※ここに用意した4Stem分離用ONNXモデルのURLを指定します
const MODEL_URL = "./models/stem_4ch.onnx";
let session = null;

async function initModel() {
  if (session) return session;
  postMessage({ type: "progress", message: "AIモデルを読み込み中..." });
  // スマホ環境に合わせてWASMを基本とし、必要に応じ調整
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"]
  });
  return session;
}

self.onmessage = async (e) => {
  try {
    const { sampleRate, left, right } = e.data;
    await initModel();

    postMessage({ type: "progress", message: "4パート分離を計算中..." });

    // TODO: モデルの入力仕様（例: [1, 2, samples]）に合わせてテンソルを作成し推論を実行
    // ここでは動作確認用のダミー出力（元音をそのまま4枠に返却）を返しています
    const dummy = {
      vocals: { left: new Float32Array(left), right: new Float32Array(right) },
      bass:   { left: new Float32Array(left), right: new Float32Array(right) },
      drums:  { left: new Float32Array(left), right: new Float32Array(right) },
      other:  { left: new Float32Array(left), right: new Float32Array(right) }
    };

    postMessage({
      type: "done",
      sampleRate,
      stems: dummy
    }, [
      dummy.vocals.left.buffer, dummy.vocals.right.buffer,
      dummy.bass.left.buffer,   dummy.bass.right.buffer,
      dummy.drums.left.buffer,  dummy.drums.right.buffer,
      dummy.other.left.buffer,  dummy.other.right.buffer
    ]);
  } catch (err) {
    postMessage({ type: "progress", message: `エラー: ${err.message}` });
  }
};