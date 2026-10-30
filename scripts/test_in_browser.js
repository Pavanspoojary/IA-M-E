const { spawn } = require('child_process');

async function main() {
  const sampleName = process.argv[2] || 'distribution_marks_sample_1.jpg';
  const targetIA = process.argv[3] || 'ALL';

  const chromeProc = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
    '--headless',
    '--remote-debugging-port=9229',
    '--disable-gpu',
    'http://localhost:8000/'
  ]);

  await new Promise(r => setTimeout(r, 2000));

  try {
    const pages = await fetch('http://localhost:9229/json').then(r => r.json());
    const page = pages.find(p => p.url.includes('8000'));
    if (!page) {
      console.error('Target page not found in Chrome DevTools.');
      return;
    }

    const ws = new WebSocket(page.webSocketDebuggerUrl);

    await new Promise((resolve, reject) => {
      ws.onopen = () => {
        ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
        ws.send(JSON.stringify({
          id: 2,
          method: 'Runtime.evaluate',
          params: {
            expression: `(async () => {
              try {
                for (let i = 0; i < 60; i++) {
                  if (typeof window.BrowserExtractionEngine !== "undefined") break;
                  await new Promise(r => setTimeout(r, 100));
                }
                if (typeof window.BrowserExtractionEngine === "undefined") {
                  return { error: "window.BrowserExtractionEngine timed out" };
                }
                const engine = new window.BrowserExtractionEngine();
                const initOk = await engine.init();
                if (!initOk) return { error: "Engine init failed" };
                
                const resp = await fetch("/sample_data/${sampleName}");
                if (!resp.ok) return { error: "Failed to fetch image: " + resp.statusText };
                const blob = await resp.blob();
                const file = new File([blob], "${sampleName}", { type: "image/jpeg" });
                
                const res = await engine.processDocument(file, "${targetIA}");
                const out = {};
                for (const [ia, marks] of Object.entries(res.results_by_ia)) {
                  out[ia] = marks
                    .filter(m => !m.is_blank || m.numeric_val > 0)
                    .map(m => ({ id: m.id, detected: m.detected_str, num: m.numeric_val, conf: m.confidence, status: m.status }));
                }
                return { out, totals: res.summary_totals };
              } catch(err) {
                return { error: err.message, stack: err.stack };
              }
            })()`,
            awaitPromise: true,
            returnByValue: true
          }
        }));
      };

      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data);
        if (data.id === 2) {
          const val = data.result && data.result.result ? data.result.result.value : data.result;
          console.log(`=== In-Browser Extraction Results for ${sampleName} (${targetIA}) ===`);
          console.log(JSON.stringify(val, null, 2));
          resolve();
        }
      };

      ws.onerror = reject;
    });

    ws.close();
  } finally {
    chromeProc.kill();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
