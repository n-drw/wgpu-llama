import init, { Model } from "./build/m.js";

Error.stackTraceLimit = 50;
async function fetchArrayBuffer(url) {
  const cacheName = "llama3-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}
class Llama2C {
  static instance = {};

  static async getInstance(weightsURL, modelID, tokenizerURL, useWgpu) {
    // load individual modelID only once
    if (!this.instance[modelID + useWgpu]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });

      const [weightsArrayU8, tokenizerArrayU8] = await Promise.all([
        fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
      ]);

      this.instance[modelID + useWgpu] = new Model(weightsArrayU8, tokenizerArrayU8, useWgpu);
    }
    return this.instance[modelID + useWgpu];
  }
}

let controller = null;
self.addEventListener("message", (event) => {
  if (event.data.command === "start") {
    controller = new AbortController();
    generate(event.data);
  } else if (event.data.command === "abort") {
    controller.abort();
  }
});

async function generate(data) {
  const {
    weightsURL,
    modelID,
    tokenizerURL,
    prompt,
    temp,
    top_p,
    repeatPenalty,
    seed,
    maxSeqLen,
    useWgpu
  } = data;
  try {
    self.postMessage({ status: "loading", message: "Starting llama3" });

    const model = await Llama2C.getInstance(weightsURL, modelID, tokenizerURL, useWgpu);

    self.postMessage({ status: "loading", message: "Initializing model" });
    const firstToken = await model.init_with_prompt(
      prompt,
      temp,
      top_p,
      repeatPenalty,
      seed
    );

    const seq_len = model.get_seq_len();

    let sentence = firstToken;
    let maxTokens = maxSeqLen ? maxSeqLen : seq_len - prompt.length - 1;
    let startTime = performance.now();
    let tokensCount = 0;
    while (tokensCount < maxTokens) {
      await new Promise(async (resolve) => {
        if (controller && controller.signal.aborted) {
          self.postMessage({
            status: "aborted",
            message: "Aborted",
            output: prompt + sentence,
          });
          return;
        }
        const token = await model.next_token();
        const tokensSec =
          ((tokensCount + 1) / (performance.now() - startTime)) * 1000;

        sentence += token;
        self.postMessage({
          status: "generating",
          message: "Generating token",
          token: token,
          sentence: sentence,
          totalTime: performance.now() - startTime,
          tokensSec,
          prompt: prompt,
        });
        setTimeout(resolve, 0);
      });
      tokensCount++;
    }
    self.postMessage({
      status: "complete",
      message: "complete",
      output: prompt + sentence,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
}
