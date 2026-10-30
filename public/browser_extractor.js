/**
 * In-Browser Neural Extraction Engine for IA Bluebook Marks.
 * Pure Client-Side Execution powered by ONNX Runtime Web (WASM).
 * Strict Zero Emojis Policy Enforced.
 */

(function (global) {
  'use strict';

  const DOM_ROWS = [
    { id: '1a', label: '1. a', y1: 96, y2: 141, max: 10.0 },
    { id: '1b', label: '1. b', y1: 141, y2: 189, max: 10.0 },
    { id: '1c', label: '1. c', y1: 189, y2: 236, max: 10.0 },
    { id: '2a', label: '2. a', y1: 236, y2: 283, max: 10.0 },
    { id: '2b', label: '2. b', y1: 283, y2: 331, max: 10.0 },
    { id: '2c', label: '2. c', y1: 331, y2: 378, max: 10.0 },
    { id: '3a', label: '3. a', y1: 378, y2: 425, max: 10.0 },
    { id: '3b', label: '3. b', y1: 425, y2: 472, max: 10.0 },
    { id: '3c', label: '3. c', y1: 472, y2: 520, max: 10.0 },
    { id: '4a', label: '4. a', y1: 520, y2: 567, max: 10.0 },
    { id: '4b', label: '4. b', y1: 567, y2: 614, max: 10.0 },
    { id: '4c', label: '4. c', y1: 614, y2: 662, max: 10.0 },
    { id: 'total', label: 'Total Marks', y1: 662, y2: 709, max: 50.0 },
  ];

  const DOM_COLUMNS = {
    'IA 1': { colRange: [0, 200], marksX: [95, 199] },
    'IA 2': { colRange: [199, 400], marksX: [294, 399] },
    'IA 3': { colRange: [399, 600], marksX: [495, 599] },
  };

  const LEGACY_ROWS = [
    { id: '1a', label: 'Q1 (a)', y: 235, max: 5.0 },
    { id: '1b', label: 'Q1 (b)', y: 307, max: 5.0 },
    { id: '2a', label: 'Q2 (a)', y: 379, max: 5.0 },
    { id: '2b', label: 'Q2 (b)', y: 449, max: 5.0 },
    { id: '3a', label: 'Q3 (a)', y: 518, max: 5.0 },
    { id: '3b', label: 'Q3 (b)', y: 588, max: 5.0 },
    { id: '3b_alt', label: 'Q3 (b-alt)', y: 658, max: 5.0 },
    { id: '4a', label: 'Q4 (a)', y: 729, max: 5.0 },
    { id: '4b', label: 'Q4 (b)', y: 800, max: 5.0 },
    { id: 'total', label: 'Total', y: 872, max: 50.0 },
  ];

  const LEGACY_COLUMNS = {
    'IA 1': { colRange: [220, 410], marksX: [308, 396] },
    'IA 2': { colRange: [410, 600], marksX: [497, 585] },
    'IA 3': { colRange: [600, 790], marksX: [686, 774] },
  };

  class BrowserExtractionEngine {
    constructor() {
      this.session = null;
      this.modelLoaded = false;
      this.initPromise = null;
    }

    /**
     * Initialize ONNX Runtime Web session.
     */
    async init() {
      if (this.modelLoaded && this.session) return true;
      if (this.initPromise) return this.initPromise;

      this.initPromise = (async () => {
        if (typeof ort === 'undefined') {
          console.error('onnxruntime-web is not loaded.');
          return false;
        }

        try {
          ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
          ort.env.wasm.numThreads = 1;

          const possibleUrls = [
            'models/digit_model.onnx',
            '/models/digit_model.onnx',
            './models/digit_model.onnx',
            '../models/digit_model.onnx',
          ];

          let session = null;
          let lastErr = null;

          for (const url of possibleUrls) {
            try {
              session = await ort.InferenceSession.create(url, {
                executionProviders: ['wasm'],
              });
              if (session) {
                console.log(`ONNX digit model loaded successfully from ${url}`);
                break;
              }
            } catch (err) {
              lastErr = err;
            }
          }

          if (!session) {
            console.error('Failed to load ONNX model from paths:', lastErr);
            return false;
          }

          this.session = session;
          this.modelLoaded = true;
          return true;
        } catch (e) {
          console.error('Error during ONNX init:', e);
          return false;
        }
      })();

      return this.initPromise;
    }

    /**
     * Render PDF file to array of HTML5 canvases.
     */
    async renderPdfToCanvases(file) {
      if (typeof pdfjsLib === 'undefined') {
        throw new Error('pdf.js library is not loaded.');
      }
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      const canvases = [];

      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        const viewport = page.getViewport({ scale: 2.0 });

        const canvas = document.createElement('canvas');
        canvas.width = Math.round(viewport.width);
        canvas.height = Math.round(viewport.height);

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        await page.render({ canvasContext: ctx, viewport: viewport }).promise;
        canvases.push(canvas);
      }

      return canvases;
    }

    /**
     * Load image file into an HTML5 canvas.
     */
    async loadImageToCanvas(file) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
          URL.revokeObjectURL(url);
          const canvas = document.createElement('canvas');
          canvas.width = img.naturalWidth || img.width;
          canvas.height = img.naturalHeight || img.height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          ctx.drawImage(img, 0, 0);
          resolve(canvas);
        };
        img.onerror = (e) => {
          URL.revokeObjectURL(url);
          reject(new Error('Failed to load image file.'));
        };
        img.src = url;
      });
    }

    /**
     * Detect and rectify marks distribution table to standard 600x780 canvas.
     */
    rectifyTable(sourceCanvas) {
      const srcW = sourceCanvas.width;
      const srcH = sourceCanvas.height;
      const srcCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
      const imgData = srcCtx.getImageData(0, 0, srcW, srcH);
      const data = imgData.data;

      // Scan rows and columns for table borders
      // VTU distribution tables occupy center ~80% width and ~60% height
      let minX = Math.round(srcW * 0.05);
      let maxX = Math.round(srcW * 0.95);
      let minY = Math.round(srcH * 0.05);
      let maxY = Math.round(srcH * 0.95);

      // Detect dark grid lines (luminance < 120)
      const rowDarkCounts = new Int32Array(srcH);
      const colDarkCounts = new Int32Array(srcW);

      for (let y = 0; y < srcH; y += 4) {
        const rowOffset = y * srcW * 4;
        for (let x = 0; x < srcW; x += 4) {
          const idx = rowOffset + x * 4;
          const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
          if (lum < 110) {
            rowDarkCounts[y]++;
            colDarkCounts[x]++;
          }
        }
      }

      // Find top/bottom table boundaries with dense horizontal lines
      for (let y = Math.round(srcH * 0.05); y < Math.round(srcH * 0.4); y += 4) {
        if (rowDarkCounts[y] > srcW * 0.1) {
          minY = Math.max(0, y - 10);
          break;
        }
      }
      for (let y = Math.round(srcH * 0.95); y > Math.round(srcH * 0.6); y -= 4) {
        if (rowDarkCounts[y] > srcW * 0.1) {
          maxY = Math.min(srcH, y + 10);
          break;
        }
      }

      for (let x = Math.round(srcW * 0.05); x < Math.round(srcW * 0.35); x += 4) {
        if (colDarkCounts[x] > srcH * 0.08) {
          minX = Math.max(0, x - 10);
          break;
        }
      }
      for (let x = Math.round(srcW * 0.95); x > Math.round(srcW * 0.65); x -= 4) {
        if (colDarkCounts[x] > srcH * 0.08) {
          maxX = Math.min(srcW, x + 10);
          break;
        }
      }

      const tableW = maxX - minX;
      const tableH = maxY - minY;

      const destCanvas = document.createElement('canvas');
      destCanvas.width = 600;
      destCanvas.height = 780;
      const destCtx = destCanvas.getContext('2d', { willReadFrequently: true });

      if (tableW > srcW * 0.4 && tableH > srcH * 0.4) {
        destCtx.drawImage(sourceCanvas, minX, minY, tableW, tableH, 0, 0, 600, 780);
      } else {
        destCtx.drawImage(sourceCanvas, 0, 0, srcW, srcH, 0, 0, 600, 780);
      }

      return destCanvas;
    }

    /**
     * Crop sub-canvas.
     */
    cropCanvas(sourceCanvas, x, y, w, h) {
      const c = document.createElement('canvas');
      c.width = Math.max(1, w);
      c.height = Math.max(1, h);
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(sourceCanvas, x, y, w, h, 0, 0, w, h);
      return c;
    }

    /**
     * Extract red-pen strokes and check strikeout status.
     */
    processCellInk(cellCanvas) {
      const w = cellCanvas.width;
      const h = cellCanvas.height;
      const ctx = cellCanvas.getContext('2d', { willReadFrequently: true });
      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data;

      // Stroke mask (1 for stroke, 0 for background)
      const mask = new Uint8Array(w * h);
      let redCount = 0;
      let darkCount = 0;

      let minX = w,
        maxX = -1,
        minY = h,
        maxY = -1;

      // Scan inner region (shave 2px border to ignore table lines)
      for (let y = 3; y < h - 3; y++) {
        const row = y * w;
        for (let x = 3; x < w - 3; x++) {
          const idx = (row + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          // Red ink isolation: red channel significantly exceeds green and blue
          const diff = r - Math.max(g, b);
          if (diff > 10) {
            mask[row + x] = 1;
            redCount++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          } else {
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            if (lum < 100) darkCount++;
          }
        }
      }

      // If red ink is insufficient, check for dark pen ink
      if (redCount < 20 && darkCount >= 30) {
        for (let y = 4; y < h - 4; y++) {
          const row = y * w;
          for (let x = 4; x < w - 4; x++) {
            const idx = (row + x) * 4;
            const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            if (lum < 95) {
              mask[row + x] = 1;
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
          }
        }
      }

      const totalStrokes = redCount >= 20 ? redCount : darkCount;

      if (totalStrokes < 18 || minX > maxX || minY > maxY) {
        return { isBlank: true, isStrikeout: false, mask, minX, maxX, minY, maxY, w, h };
      }

      const rw = maxX - minX + 1;
      const rh = maxY - minY + 1;
      const ratio = rw / Math.max(1, rh);

      // Strikeout detection rule (w/h >= 2.5 or flat horizontal line)
      const isStrikeout =
        ratio >= 2.5 || (rh <= 12 && rw >= 18) || (rw > w * 0.35 && rh <= 14);

      return {
        isBlank: false,
        isStrikeout,
        mask,
        minX,
        maxX,
        minY,
        maxY,
        rw,
        rh,
        w,
        h,
      };
    }

    /**
     * Segment digit strokes into distinct character crops.
     */
    segmentDigits(cellInfo) {
      const { mask, minX, maxX, minY, maxY, rw, rh, w, h } = cellInfo;
      const digitCrops = [];

      // Check for two digits side-by-side (e.g. 15, 20, 24, 25)
      // If width is substantial compared to height (aspect ratio > 1.15)
      if (rw > 26 && rw / Math.max(1, rh) > 1.15) {
        // Calculate vertical projection across x
        const colHist = new Int32Array(rw);
        for (let x = minX; x <= maxX; x++) {
          let c = 0;
          for (let y = minY; y <= maxY; y++) {
            if (mask[y * w + x]) c++;
          }
          colHist[x - minX] = c;
        }

        // Find valley / split point in the middle 60%
        let minVal = 9999;
        let splitX = -1;
        const startX = Math.round(rw * 0.3);
        const endX = Math.round(rw * 0.7);

        for (let i = startX; i <= endX; i++) {
          if (colHist[i] < minVal) {
            minVal = colHist[i];
            splitX = minX + i;
          }
        }

        if (splitX > minX + 5 && splitX < maxX - 5 && minVal <= Math.max(2, rh * 0.25)) {
          digitCrops.push({ minX: minX, maxX: splitX - 1, minY, maxY });
          digitCrops.push({ minX: splitX + 1, maxX: maxX, minY, maxY });
          return digitCrops;
        }
      }

      digitCrops.push({ minX, maxX, minY, maxY });
      return digitCrops;
    }

    /**
     * Normalize a segmented digit stroke to 28x28 Float32Array (MNIST format).
     */
    create28x28Tensor(cellInfo, cropBounds) {
      const { mask, w, h } = cellInfo;
      const { minX, maxX, minY, maxY } = cropBounds;
      const dw = maxX - minX + 1;
      const dh = maxY - minY + 1;

      // 28x28 normalized canvas
      const normCanvas = document.createElement('canvas');
      normCanvas.width = 28;
      normCanvas.height = 28;
      const nctx = normCanvas.getContext('2d', { willReadFrequently: true });
      nctx.fillStyle = '#000000';
      nctx.fillRect(0, 0, 28, 28);

      // Create cropped digit canvas
      const dCanvas = document.createElement('canvas');
      dCanvas.width = dw;
      dCanvas.height = dh;
      const dctx = dCanvas.getContext('2d', { willReadFrequently: true });
      const dImgData = dctx.createImageData(dw, dh);

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const srcIdx = y * w + x;
          const dstIdx = ((y - minY) * dw + (x - minX)) * 4;
          if (mask[srcIdx]) {
            dImgData.data[dstIdx] = 255;
            dImgData.data[dstIdx + 1] = 255;
            dImgData.data[dstIdx + 2] = 255;
            dImgData.data[dstIdx + 3] = 255;
          } else {
            dImgData.data[dstIdx] = 0;
            dImgData.data[dstIdx + 1] = 0;
            dImgData.data[dstIdx + 2] = 0;
            dImgData.data[dstIdx + 3] = 255;
          }
        }
      }
      dctx.putImageData(dImgData, 0, 0);

      // Scale to fit within 20x20 maintaining aspect ratio
      const scale = Math.min(20 / dw, 20 / dh);
      const scaledW = Math.max(1, Math.round(dw * scale));
      const scaledH = Math.max(1, Math.round(dh * scale));
      const targetX = Math.round((28 - scaledW) / 2);
      const targetY = Math.round((28 - scaledH) / 2);

      nctx.drawImage(dCanvas, 0, 0, dw, dh, targetX, targetY, scaledW, scaledH);

      // Extract Float32Array
      const finalImgData = nctx.getImageData(0, 0, 28, 28).data;
      const tensorData = new Float32Array(28 * 28);

      for (let i = 0; i < 28 * 28; i++) {
        tensorData[i] = finalImgData[i * 4] / 255.0;
      }

      return tensorData;
    }

    /**
     * Run neural classification on a 28x28 digit tensor.
     */
    async classifyDigit(tensorData) {
      if (!this.session) {
        return { digit: '0', confidence: 0.5 };
      }

      const inputTensor = new ort.Tensor('float32', tensorData, [1, 1, 28, 28]);
      const results = await this.session.run({ input: inputTensor });
      const output = results.output.data;

      // Softmax
      let maxLogit = -Infinity;
      for (let i = 0; i < 10; i++) {
        if (output[i] > maxLogit) maxLogit = output[i];
      }

      let expSum = 0;
      const exps = new Float32Array(10);
      for (let i = 0; i < 10; i++) {
        exps[i] = Math.exp(output[i] - maxLogit);
        expSum += exps[i];
      }

      let bestIdx = 0;
      let bestProb = 0;
      for (let i = 0; i < 10; i++) {
        const p = exps[i] / expSum;
        if (p > bestProb) {
          bestProb = p;
          bestIdx = i;
        }
      }

      return {
        digit: String(bestIdx),
        confidence: Math.round(bestProb * 1000) / 1000,
      };
    }

    /**
     * Process a single cell and determine mark value.
     */
    async recognizeCell(cellCanvas, maxMarks) {
      const cellInfo = this.processCellInk(cellCanvas);

      if (cellInfo.isBlank) {
        return {
          detectedStr: '-',
          numericVal: 0.0,
          confidence: 1.0,
          status: 'Blank',
          isBlank: true,
        };
      }

      if (cellInfo.isStrikeout) {
        return {
          detectedStr: '-',
          numericVal: 0.0,
          confidence: 0.95,
          status: 'Valid',
          isBlank: false,
        };
      }

      // Segment character strokes
      const digitBoxes = this.segmentDigits(cellInfo);
      const tokens = [];
      let totalConf = 0;

      for (const box of digitBoxes) {
        const tensorData = this.create28x28Tensor(cellInfo, box);
        const res = await this.classifyDigit(tensorData);
        tokens.push(res.digit);
        totalConf += res.confidence;
      }

      const valStr = tokens.join('');
      const numVal = parseFloat(valStr) || 0.0;
      const avgConf = tokens.length > 0 ? totalConf / tokens.length : 0.0;

      const isValidRange = numVal >= 0 && numVal <= maxMarks;
      const status = avgConf >= 0.65 && isValidRange ? 'Valid' : 'Needs review';

      return {
        detectedStr: valStr,
        numericVal: numVal,
        confidence: Math.round(avgConf * 1000) / 1000,
        status,
        isBlank: false,
      };
    }

    /**
     * Extract evaluation sheet from canvas.
     */
    async extractFromCanvas(canvas, targetIA = 'ALL') {
      await this.init();

      const rectified = this.rectifyTable(canvas);

      const targetClean = String(targetIA).trim().toUpperCase();
      let selectedIAs = [];
      if (targetClean.includes('ALL')) {
        selectedIAs = ['IA 1', 'IA 2', 'IA 3'];
      } else if (targetClean.includes('IA 2')) {
        selectedIAs = ['IA 2'];
      } else if (targetClean.includes('IA 3')) {
        selectedIAs = ['IA 3'];
      } else {
        selectedIAs = ['IA 1'];
      }

      // Create Overview Column Crop
      let overviewCanvas;
      if (selectedIAs.length === 1) {
        const range = DOM_COLUMNS[selectedIAs[0]].colRange;
        overviewCanvas = this.cropCanvas(rectified, range[0], 0, range[1] - range[0], 715);
      } else {
        overviewCanvas = this.cropCanvas(rectified, 0, 0, 600, 715);
      }
      const overviewBase64 = overviewCanvas.toDataURL('image/png');

      const resultsByIA = {};
      const summaryTotals = {};

      for (const iaName of selectedIAs) {
        const marksX = DOM_COLUMNS[iaName].marksX;
        const marksList = [];
        let calcSum = 0;
        let detectedTotal = null;

        for (const row of DOM_ROWS) {
          const cellCanvas = this.cropCanvas(
            rectified,
            marksX[0] + 6,
            row.y1 + 5,
            marksX[1] - marksX[0] - 12,
            row.y2 - row.y1 - 10
          );

          const cellB64 = cellCanvas.toDataURL('image/png');
          const markRes = await this.recognizeCell(cellCanvas, row.max);

          if (row.id === 'total') {
            detectedTotal = markRes.isBlank ? null : markRes.numericVal;
          } else if (!markRes.isBlank && markRes.numericVal > 0) {
            calcSum += markRes.numericVal;
          }

          marksList.push({
            id: row.id,
            label: row.label,
            detected_str: markRes.detectedStr,
            numeric_val: markRes.numericVal,
            verified_val: markRes.numericVal,
            confidence: markRes.confidence,
            status: markRes.status,
            is_blank: markRes.isBlank,
            crop_base64: cellB64,
          });
        }

        resultsByIA[iaName] = marksList;
        summaryTotals[iaName] = {
          calculated_sum: Math.round(calcSum * 10) / 10,
          detected_total: detectedTotal,
          matches:
            detectedTotal !== null && Math.abs(calcSum - detectedTotal) < 0.01,
        };
      }

      return {
        template_type: 'DISTRIBUTION_OF_MARKS',
        column_crop_base64: overviewBase64,
        target_ia: targetIA,
        selected_ias: selectedIAs,
        results_by_ia: resultsByIA,
        summary_totals: summaryTotals,
        total_sheets: 1,
        sheets: [
          {
            sheet_index: 1,
            page_number: 1,
            sheet_label: 'Sheet 1 (Page 1)',
            template_type: 'DISTRIBUTION_OF_MARKS',
            column_crop_base64: overviewBase64,
            target_ia: targetIA,
            selected_ias: selectedIAs,
            results_by_ia: resultsByIA,
            summary_totals: summaryTotals,
          },
        ],
      };
    }

    /**
     * Process multi-page PDF or single image file completely in browser.
     */
    async processDocument(file, targetIA = 'ALL') {
      const isPdf =
        file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');

      let canvases = [];
      if (isPdf) {
        canvases = await this.renderPdfToCanvases(file);
      } else {
        const c = await this.loadImageToCanvas(file);
        canvases = [c];
      }

      const allSheets = [];
      let firstOverviewB64 = '';

      for (let i = 0; i < canvases.length; i++) {
        const sheetRes = await this.extractFromCanvas(canvases[i], targetIA);
        const sheetObj = sheetRes.sheets[0];
        sheetObj.sheet_index = i + 1;
        sheetObj.page_number = i + 1;
        sheetObj.sheet_label = `Sheet ${i + 1} (Page ${i + 1})`;
        allSheets.push(sheetObj);

        if (i === 0) {
          firstOverviewB64 = sheetRes.column_crop_base64;
        }
      }

      return {
        template_type: 'DISTRIBUTION_OF_MARKS',
        column_crop_base64: firstOverviewB64,
        target_ia: targetIA,
        selected_ias: allSheets[0].selected_ias,
        results_by_ia: allSheets[0].results_by_ia,
        summary_totals: allSheets[0].summary_totals,
        total_sheets: allSheets.length,
        sheets: allSheets,
      };
    }

    /**
     * Export data to CSV string and download via Blob.
     */
    downloadCsv(extractionData, filename = 'IA_Marks_Evaluation.csv') {
      const sheets = extractionData.sheets || [];
      if (sheets.length === 0) return;

      const qHeaders = [
        '1.a', '1.b', '1.c',
        '2.a', '2.b', '2.c',
        '3.a', '3.b', '3.c',
        '4.a', '4.b', '4.c',
      ];
      const headerLine = ['Sheet', 'Assessment', ...qHeaders, 'Total', 'Status'].join(',');

      const lines = [headerLine];

      sheets.forEach((s) => {
        const sLabel = `"${s.sheet_label || `Sheet ${s.sheet_index}`}"`;
        const iaKeys = Object.keys(s.results_by_ia || {});

        iaKeys.forEach((ia) => {
          const marksList = s.results_by_ia[ia] || [];
          const markMap = {};
          marksList.forEach((m) => {
            markMap[m.label] = m;
          });

          const rowVals = [sLabel, `"${ia}"`];

          qHeaders.forEach((qh) => {
            const m = markMap[qh];
            if (!m || m.is_blank || m.verified_val === null || m.verified_val === 0) {
              rowVals.push('-');
            } else {
              rowVals.push(String(m.verified_val));
            }
          });

          const totMark = marksList.find((m) => m.id === 'total');
          const totVal = totMark ? totMark.verified_val : 0;
          rowVals.push(String(totVal));

          const summary = (s.summary_totals && s.summary_totals[ia]) || {};
          const status = summary.matches ? 'Verified Match' : 'Discrepancy';
          rowVals.push(`"${status}"`);

          lines.push(rowVals.join(','));
        });
      });

      const csvContent = '\ufeff' + lines.join('\r\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      this._triggerDownload(blob, filename);
    }

    /**
     * Export data to XLSX Excel file and download.
     */
    downloadXlsx(extractionData, filename = 'IA_Marks_Evaluation.xlsx') {
      if (typeof XLSX === 'undefined') {
        throw new Error('SheetJS (xlsx) library is not loaded.');
      }

      const wb = XLSX.utils.book_new();
      const sheets = extractionData.sheets || [];
      const qHeaders = [
        '1.a', '1.b', '1.c',
        '2.a', '2.b', '2.c',
        '3.a', '3.b', '3.c',
        '4.a', '4.b', '4.c',
      ];

      // Summary Sheet
      const summaryRows = [
        ['Continuous Internal Assessment (IA) Marks Evaluation Ledger'],
        ['Generated via In-Browser Neural Extraction Engine'],
        [],
        ['Sheet', 'Assessment', ...qHeaders, 'Total', 'Status'],
      ];

      sheets.forEach((s) => {
        const sLabel = s.sheet_label || `Sheet ${s.sheet_index}`;
        const iaKeys = Object.keys(s.results_by_ia || {});

        iaKeys.forEach((ia) => {
          const marksList = s.results_by_ia[ia] || [];
          const markMap = {};
          marksList.forEach((m) => {
            markMap[m.label] = m;
          });

          const row = [sLabel, ia];
          qHeaders.forEach((qh) => {
            const m = markMap[qh];
            if (!m || m.is_blank || m.verified_val === null || m.verified_val === 0) {
              row.push('-');
            } else {
              row.push(m.verified_val);
            }
          });

          const totMark = marksList.find((m) => m.id === 'total');
          row.push(totMark ? totMark.verified_val : 0);

          const summary = (s.summary_totals && s.summary_totals[ia]) || {};
          row.push(summary.matches ? 'Verified Match' : 'Discrepancy');
          summaryRows.push(row);
        });
      });

      const wsSummary = XLSX.utils.aoa_to_sheet(summaryRows);
      XLSX.utils.book_append_sheet(wb, wsSummary, 'IA Marks Summary');

      // Individual Sheets
      sheets.forEach((s) => {
        const sheetName = `Sheet ${s.sheet_index}`.substring(0, 31);
        const detailRows = [
          [`Assessment Marks Detail - ${s.sheet_label}`],
          [],
          ['Question', 'Assessment', 'Detected', 'Verified Mark', 'Status'],
        ];

        const iaKeys = Object.keys(s.results_by_ia || {});
        iaKeys.forEach((ia) => {
          const marksList = s.results_by_ia[ia] || [];
          marksList.forEach((m) => {
            detailRows.push([
              m.label,
              ia,
              m.detected_str,
              m.verified_val,
              m.status,
            ]);
          });
        });

        const wsDetail = XLSX.utils.aoa_to_sheet(detailRows);
        XLSX.utils.book_append_sheet(wb, wsDetail, sheetName);
      });

      XLSX.writeFile(wb, filename);
    }

    _triggerDownload(blob, filename) {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    }
  }

  global.BrowserExtractionEngine = BrowserExtractionEngine;
})(typeof window !== 'undefined' ? window : this);
