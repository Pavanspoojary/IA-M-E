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
     * Ensure OpenCV.js runtime is fully initialized.
     */
    async ensureOpenCVReady() {
      if (typeof cv !== 'undefined' && cv.Mat && cv.getPerspectiveTransform) {
        return true;
      }
      if (typeof cv !== 'undefined') {
        return new Promise((resolve) => {
          let resolved = false;
          const orig = cv['onRuntimeInitialized'];
          cv['onRuntimeInitialized'] = () => {
            if (typeof orig === 'function') orig();
            if (!resolved) {
              resolved = true;
              resolve(true);
            }
          };
          const interval = setInterval(() => {
            if (typeof cv !== 'undefined' && cv.Mat && cv.getPerspectiveTransform) {
              clearInterval(interval);
              if (!resolved) {
                resolved = true;
                resolve(true);
              }
            }
          }, 50);
          setTimeout(() => {
            clearInterval(interval);
            if (!resolved) {
              resolved = true;
              resolve(false);
            }
          }, 4000);
        });
      }
      return false;
    }

    /**
     * Initialize ONNX Runtime Web session.
     */
    async init() {
      if (this.modelLoaded && this.session) return true;
      if (this.initPromise) return this.initPromise;

      this.initPromise = (async () => {
        // Initialize OpenCV.js
        await this.ensureOpenCVReady();

        if (typeof ort === 'undefined') {
          console.error('onnxruntime-web is not loaded.');
          return false;
        }

        try {
          ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
          ort.env.wasm.numThreads = 1;

          const MODEL_VERSION = '20261031_v8';
          const possibleUrls = [
            `models/digit_model.onnx?v=${MODEL_VERSION}`,
            `/models/digit_model.onnx?v=${MODEL_VERSION}`,
            `./models/digit_model.onnx?v=${MODEL_VERSION}`,
            `../models/digit_model.onnx?v=${MODEL_VERSION}`,
          ];

          let session = null;
          let lastErr = null;

          for (const url of possibleUrls) {
            try {
              const resp = await fetch(url);
              if (!resp.ok) continue;
              const buffer = await resp.arrayBuffer();
              if (!buffer || buffer.byteLength < 1000) continue;

              session = await ort.InferenceSession.create(buffer);
              if (session) {
                console.log(`ONNX digit model loaded successfully from ${url} (${buffer.byteLength} bytes)`);
                break;
              }
            } catch (err) {
              lastErr = err;
            }
          }

          if (!session) {
            console.error('Failed to load ONNX model from paths:', lastErr);
            this.initPromise = null;
            return false;
          }

          this.session = session;
          this.modelLoaded = true;
          return true;
        } catch (e) {
          console.error('Error during ONNX init:', e);
          this.initPromise = null;
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
     * Order 4 polygon points as [top-left, top-right, bottom-right, bottom-left].
     */
    orderPoints(pts) {
      let tl = pts[0], br = pts[0], tr = pts[0], bl = pts[0];
      let minSum = pts[0].x + pts[0].y;
      let maxSum = pts[0].x + pts[0].y;
      let minDiff = pts[0].y - pts[0].x;
      let maxDiff = pts[0].y - pts[0].x;

      for (let i = 1; i < pts.length; i++) {
        const p = pts[i];
        const s = p.x + p.y;
        const d = p.y - p.x;
        if (s < minSum) {
          minSum = s;
          tl = p;
        }
        if (s > maxSum) {
          maxSum = s;
          br = p;
        }
        if (d < minDiff) {
          minDiff = d;
          tr = p;
        }
        if (d > maxDiff) {
          maxDiff = d;
          bl = p;
        }
      }
      return [tl, tr, br, bl];
    }

    /**
     * Perspective warp table using OpenCV.js 4-corner detection.
     */
    rectifyTableWithOpenCV(sourceCanvas) {
      if (typeof cv === 'undefined' || !cv.Mat || !cv.getPerspectiveTransform) {
        return null;
      }
      let src = null, small = null, gray = null, blur = null, thresh = null;
      let contours = null, hierarchy = null, srcTri = null, dstTri = null, M = null, dst = null;

      try {
        src = cv.imread(sourceCanvas);
        const origW = src.cols;
        const origH = src.rows;

        // Downscale for high-speed contour search if image is very large
        const scale = Math.min(1.0, 1200.0 / Math.max(origW, origH));
        let workMat = src;
        if (scale < 0.99) {
          small = new cv.Mat();
          cv.resize(src, small, new cv.Size(Math.round(origW * scale), Math.round(origH * scale)), 0, 0, cv.INTER_AREA);
          workMat = small;
        }

        const w = workMat.cols;
        const h = workMat.rows;

        gray = new cv.Mat();
        cv.cvtColor(workMat, gray, cv.COLOR_RGBA2GRAY);
        blur = new cv.Mat();
        cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
        thresh = new cv.Mat();
        cv.adaptiveThreshold(blur, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 4);

        contours = new cv.MatVector();
        hierarchy = new cv.Mat();
        cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let bestPts = null;
        let maxArea = 0;
        const minArea = w * h * 0.18;

        // 1. Try 4-point polygon approximation
        for (let i = 0; i < contours.size(); i++) {
          const c = contours.get(i);
          const area = cv.contourArea(c);
          if (area > maxArea && area > minArea) {
            const peri = cv.arcLength(c, true);
            for (const eps of [0.02, 0.025, 0.03, 0.015]) {
              const approx = new cv.Mat();
              cv.approxPolyDP(c, approx, eps * peri, true);
              if (approx.rows === 4) {
                maxArea = area;
                bestPts = [
                  { x: approx.data32S[0] / scale, y: approx.data32S[1] / scale },
                  { x: approx.data32S[2] / scale, y: approx.data32S[3] / scale },
                  { x: approx.data32S[4] / scale, y: approx.data32S[5] / scale },
                  { x: approx.data32S[6] / scale, y: approx.data32S[7] / scale },
                ];
                approx.delete();
                break;
              }
              approx.delete();
            }
          }
        }

        // 2. Fallback to minAreaRect if no exact 4-vertex polygon
        if (!bestPts) {
          for (let i = 0; i < contours.size(); i++) {
            const c = contours.get(i);
            const area = cv.contourArea(c);
            if (area > maxArea && area > minArea) {
              const rotRect = cv.minAreaRect(c);
              const pts = cv.RotatedRect.points(rotRect);
              maxArea = area;
              bestPts = [
                { x: pts[0].x / scale, y: pts[0].y / scale },
                { x: pts[1].x / scale, y: pts[1].y / scale },
                { x: pts[2].x / scale, y: pts[2].y / scale },
                { x: pts[3].x / scale, y: pts[3].y / scale },
              ];
            }
          }
        }

        if (!bestPts) {
          return null;
        }

        const ordered = this.orderPoints(bestPts);
        const destW = 600;
        const destH = 780;

        srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
          ordered[0].x, ordered[0].y,
          ordered[1].x, ordered[1].y,
          ordered[2].x, ordered[2].y,
          ordered[3].x, ordered[3].y,
        ]);
        dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
          0, 0,
          destW - 1, 0,
          destW - 1, destH - 1,
          0, destH - 1,
        ]);

        M = cv.getPerspectiveTransform(srcTri, dstTri);
        dst = new cv.Mat();
        const dsize = new cv.Size(destW, destH);
        cv.warpPerspective(src, dst, M, dsize);

        const destCanvas = document.createElement('canvas');
        destCanvas.width = destW;
        destCanvas.height = destH;
        cv.imshow(destCanvas, dst);

        return destCanvas;
      } catch (err) {
        console.warn('OpenCV perspective rectification failed:', err);
        return null;
      } finally {
        if (src) src.delete();
        if (small) small.delete();
        if (gray) gray.delete();
        if (blur) blur.delete();
        if (thresh) thresh.delete();
        if (contours) contours.delete();
        if (hierarchy) hierarchy.delete();
        if (srcTri) srcTri.delete();
        if (dstTri) dstTri.delete();
        if (M) M.delete();
        if (dst) dst.delete();
      }
    }

    /**
     * Pure JavaScript fallback for table boundary detection.
     */
    rectifyTablePureJS(sourceCanvas) {
      const srcW = sourceCanvas.width;
      const srcH = sourceCanvas.height;
      const srcCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
      const imgData = srcCtx.getImageData(0, 0, srcW, srcH);
      const data = imgData.data;

      let minX = Math.round(srcW * 0.05);
      let maxX = Math.round(srcW * 0.95);
      let minY = Math.round(srcH * 0.12);
      let maxY = Math.round(srcH * 0.95);

      const rowDarkCounts = new Int32Array(srcH);
      const colDarkCounts = new Int32Array(srcW);

      for (let y = 0; y < srcH; y += 3) {
        const rowOffset = y * srcW * 4;
        for (let x = 0; x < srcW; x += 3) {
          const idx = rowOffset + x * 4;
          const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
          if (lum < 105) {
            rowDarkCounts[y]++;
            colDarkCounts[x]++;
          }
        }
      }

      const thresholdDark = (srcW / 3) * 0.35;
      for (let y = Math.round(srcH * 0.12); y < Math.round(srcH * 0.45); y += 3) {
        if (rowDarkCounts[y] > thresholdDark) {
          minY = y;
          break;
        }
      }

      for (let y = Math.round(srcH * 0.95); y > Math.round(srcH * 0.60); y -= 3) {
        if (rowDarkCounts[y] > thresholdDark) {
          maxY = y;
          break;
        }
      }

      const thresholdCol = (srcH / 3) * 0.30;
      for (let x = Math.round(srcW * 0.04); x < Math.round(srcW * 0.35); x += 3) {
        if (colDarkCounts[x] > thresholdCol) {
          minX = x;
          break;
        }
      }

      for (let x = Math.round(srcW * 0.96); x > Math.round(srcW * 0.65); x -= 3) {
        if (colDarkCounts[x] > thresholdCol) {
          maxX = x;
          break;
        }
      }

      const tableW = maxX - minX;
      const tableH = maxY - minY;

      const destCanvas = document.createElement('canvas');
      destCanvas.width = 600;
      destCanvas.height = 780;
      const destCtx = destCanvas.getContext('2d', { willReadFrequently: true });

      if (tableW > srcW * 0.45 && tableH > srcH * 0.40) {
        destCtx.drawImage(sourceCanvas, minX, minY, tableW, tableH, 0, 0, 600, 780);
      } else {
        const cropY = Math.round(srcH * 0.16);
        const cropH = Math.round(srcH * 0.76);
        const cropX = Math.round(srcW * 0.08);
        const cropW = Math.round(srcW * 0.84);
        destCtx.drawImage(sourceCanvas, cropX, cropY, cropW, cropH, 0, 0, 600, 780);
      }

      return destCanvas;
    }

    /**
     * Detect and rectify marks distribution table to standard 600x780 canvas.
     */
    async rectifyTable(sourceCanvas) {
      await this.ensureOpenCVReady();
      const cvWarped = this.rectifyTableWithOpenCV(sourceCanvas);
      if (cvWarped) {
        return cvWarped;
      }
      return this.rectifyTablePureJS(sourceCanvas);
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
     * Dynamically detect table horizontal and vertical grid lines to calibrate row and column bounds.
     * Prevents row misalignment when images are cropped, scaled, or single-column.
     */
    detectTableGrid(rectifiedCanvas) {
      if (typeof cv === 'undefined' || !cv.Mat || !cv.imread) {
        return { rows: DOM_ROWS, columns: DOM_COLUMNS, isSingleColumn: false };
      }
      let mat = null, gray = null, bw = null, hKernel = null, hLines = null, vKernel = null, vLines = null;
      try {
        mat = cv.imread(rectifiedCanvas);
        const W = mat.cols;
        const H = mat.rows;
        gray = new cv.Mat();
        cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
        bw = new cv.Mat();
        cv.adaptiveThreshold(gray, bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 5);

        // 1. Detect horizontal grid lines
        hKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(30, 1));
        hLines = new cv.Mat();
        cv.morphologyEx(bw, hLines, cv.MORPH_OPEN, hKernel);

        const hProj = new Float32Array(H);
        const xMin = Math.round(W * 0.05);
        const xMax = Math.round(W * 0.95);
        for (let y = 0; y < H; y++) {
          let sum = 0;
          const rowStart = y * W;
          for (let x = xMin; x < xMax && x < W; x++) {
            if (hLines.data[rowStart + x] > 0) sum++;
          }
          hProj[y] = sum;
        }

        // Find yStart and yEnd peaks
        let yStart = 93;
        let maxS = -1;
        for (let y = 80; y <= 110 && y < H; y++) {
          if (hProj[y] > maxS) {
            maxS = hProj[y];
            yStart = y;
          }
        }
        let yEnd = 709;
        let maxE = -1;
        for (let y = 680; y <= 725 && y < H; y++) {
          if (hProj[y] > maxE) {
            maxE = hProj[y];
            yEnd = y;
          }
        }
        const avgRowH = (yEnd - yStart) / 13.0;

        const rowPeaks = [yStart];
        const peakThreshold = Math.max(25, (xMax - xMin) * 0.08);
        for (let k = 1; k < 13; k++) {
          const expY = Math.round(yStart + k * avgRowH);
          const wLow = Math.max(0, expY - 10);
          const wHigh = Math.min(H - 1, expY + 10);
          let bestY = expY;
          let maxP = -1;
          for (let y = wLow; y <= wHigh; y++) {
            if (hProj[y] > maxP) {
              maxP = hProj[y];
              bestY = y;
            }
          }
          rowPeaks.push(maxP > peakThreshold ? bestY : expY);
        }
        rowPeaks.push(yEnd);

        const rowKeys = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c', 'total'];
        const newRows = [];
        for (let r = 0; r < rowKeys.length; r++) {
          newRows.push({
            id: rowKeys[r],
            label: DOM_ROWS[r].label,
            y1: rowPeaks[r],
            y2: Math.max(rowPeaks[r] + 35, rowPeaks[r + 1]),
            max: DOM_ROWS[r].max,
          });
        }

        // 2. Detect vertical grid lines
        vKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 30));
        vLines = new cv.Mat();
        cv.morphologyEx(bw, vLines, cv.MORPH_OPEN, vKernel);

        const vProj = new Float32Array(W);
        for (let x = 0; x < W; x++) {
          let sum = 0;
          for (let y = 150; y < 650 && y < H; y++) {
            if (vLines.data[y * W + x] > 0) sum++;
          }
          vProj[x] = sum;
        }

        const isSingleColumn = (W / H < 0.45);
        let dynamicCols = DOM_COLUMNS;

        if (isSingleColumn) {
          const midX = Math.round(W * 0.46);
          const marksRange = [Math.max(0, midX + 2), W - 4];
          dynamicCols = {
            'IA 1': { colRange: [0, W], marksX: marksRange },
            'IA 2': { colRange: [0, W], marksX: marksRange },
            'IA 3': { colRange: [0, W], marksX: marksRange },
          };
        } else {
          const expectedV = [0, 95, 199, 295, 399, 495, 599];
          const vSnapped = [];
          for (const expX of expectedV) {
            const wLow = Math.max(0, expX - 12);
            const wHigh = Math.min(W - 1, expX + 12);
            let bestX = expX;
            let maxVal = -1;
            for (let x = wLow; x <= wHigh; x++) {
              if (vProj[x] > maxVal) {
                maxVal = vProj[x];
                bestX = x;
              }
            }
            vSnapped.push(maxVal > 50 ? bestX : expX);
          }
          dynamicCols = {
            'IA 1': { colRange: [vSnapped[0], vSnapped[2]], marksX: [vSnapped[1], vSnapped[2]] },
            'IA 2': { colRange: [vSnapped[2], vSnapped[4]], marksX: [vSnapped[3], vSnapped[4]] },
            'IA 3': { colRange: [vSnapped[4], vSnapped[6]], marksX: [vSnapped[5], vSnapped[6]] },
          };
        }

        return { rows: newRows, columns: dynamicCols, isSingleColumn };
      } catch (e) {
        console.warn('Grid line detection error, using defaults:', e);
        return { rows: DOM_ROWS, columns: DOM_COLUMNS, isSingleColumn: false };
      } finally {
        if (mat) mat.delete();
        if (gray) gray.delete();
        if (bw) bw.delete();
        if (hKernel) hKernel.delete();
        if (hLines) hLines.delete();
        if (vKernel) vKernel.delete();
        if (vLines) vLines.delete();
      }
    }

    /**
     * Extract handwritten strokes and check strikeout status.
     * Uses OpenCV.js connected component analysis when available with adaptive background thresholding.
     */
    processCellInk(cellCanvas, row) {
      const w = cellCanvas.width;
      const h = cellCanvas.height;

      // 1. High-accuracy OpenCV.js pipeline
      if (typeof cv !== 'undefined' && cv.Mat && cv.adaptiveThreshold && cv.connectedComponentsWithStats) {
        const inner = document.createElement('canvas');
        inner.width = Math.max(1, w - 6);
        inner.height = Math.max(1, h - 4);
        const ictx = inner.getContext('2d', { willReadFrequently: true });
        ictx.drawImage(cellCanvas, 3, 2, inner.width, inner.height, 0, 0, inner.width, inner.height);
        const idata = ictx.getImageData(0, 0, inner.width, inner.height).data;
        const iw = inner.width;
        const ih = inner.height;
        let redCount = 0;
        const isRed = new Uint8Array(iw * ih);

        for (let i = 0; i < iw * ih; i++) {
          const r = idata[i * 4];
          const g = idata[i * 4 + 1];
          const b = idata[i * 4 + 2];
          const diff = r - Math.max(g, b);
          if (diff >= 3.0 || r >= g + 2.0) {
            isRed[i] = 1;
            redCount++;
          }
        }

        const mask = new Uint8Array(w * h);
        if (redCount < 10) {
          // Check central dark ink (blue/black pen)
          let darkCount = 0;
          for (let y = Math.floor(ih * 0.15); y < Math.floor(ih * 0.85); y++) {
            const rOff = y * iw * 4;
            for (let x = Math.floor(iw * 0.15); x < Math.floor(iw * 0.85); x++) {
              const idx = rOff + x * 4;
              const lum = 0.299 * idata[idx] + 0.587 * idata[idx + 1] + 0.114 * idata[idx + 2];
              if (lum < 110) darkCount++;
            }
          }
          if (darkCount < 12) {
            return { isBlank: true, isStrikeout: false, mask, minX: 0, maxX: 0, minY: 0, maxY: 0, rw: 0, rh: 0, w, h };
          }
        }

        let mat = null, gray = null, blur = null, th = null, labels = null, stats = null, centroids = null;
        try {
          mat = cv.imread(inner);
          gray = new cv.Mat();
          cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
          blur = new cv.Mat();
          cv.GaussianBlur(gray, blur, new cv.Size(3, 3), 0);
          th = new cv.Mat();
          cv.adaptiveThreshold(blur, th, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 8);

          if (redCount >= 10) {
            for (let i = 0; i < iw * ih; i++) {
              if (!isRed[i]) th.data[i] = 0;
            }
          }

          labels = new cv.Mat();
          stats = new cv.Mat();
          centroids = new cv.Mat();

          // Strip any horizontal grid lines at top (cy <= 3) and bottom (cy >= ih - 4)
          for (let y = 0; y < ih; y++) {
            if (y <= 3 || y >= ih - 4) {
              let run = 0;
              for (let x = 0; x < iw; x++) {
                if (th.data[y * iw + x] > 0) run++;
              }
              if (run >= Math.min(18, Math.round(iw * 0.35))) {
                for (let x = 0; x < iw; x++) th.data[y * iw + x] = 0;
              }
            }
          }
          // Also strip vertical grid lines at left (cx <= 3) and right (cx >= iw - 4)
          for (let x = 0; x < iw; x++) {
            if (x <= 3 || x >= iw - 4) {
              let run = 0;
              for (let y = 0; y < ih; y++) {
                if (th.data[y * iw + x] > 0) run++;
              }
              if (run >= Math.min(15, Math.round(ih * 0.40))) {
                for (let y = 0; y < ih; y++) th.data[y * iw + x] = 0;
              }
            }
          }

          const numL = cv.connectedComponentsWithStats(th, labels, stats, centroids);

          for (let l = 1; l < numL; l++) {
            const cx = stats.intAt(l, cv.CC_STAT_LEFT);
            const cy = stats.intAt(l, cv.CC_STAT_TOP);
            const cw = stats.intAt(l, cv.CC_STAT_WIDTH);
            const ch = stats.intAt(l, cv.CC_STAT_HEIGHT);
            const area = stats.intAt(l, cv.CC_STAT_AREA);

            const isEdgeShaving = (
              ((cy <= 4 || (cy + ch) >= ih - 4) && ch <= 4 && area <= 30) ||
              ((cx <= 4 || (cx + cw) >= iw - 4) && cw <= 5 && area <= 30) ||
              (cy <= 2 && ch <= 2) ||
              ((cy + ch) >= ih - 2 && ch <= 2) ||
              (cx <= 2 && cw <= 2) ||
              ((cx + cw) >= iw - 2 && cw <= 2)
            );
            const isHBorder = (cy <= 3 || (cy + ch) >= ih - 3) && (ch <= 6 && (cw >= 25 || cw / Math.max(1, ch) >= 3.0));
            const isVBorder = (cx <= 2 || (cx + cw) >= iw - 2) && (ch >= 20 && ch / Math.max(1, cw) >= 2.5);
            const isSpeck = area < 8;

            if (isEdgeShaving || isHBorder || isVBorder || isSpeck) {
              for (let i = 0; i < iw * ih; i++) {
                if (labels.intAt(Math.floor(i / iw), i % iw) === l) {
                  th.data[i] = 0;
                }
              }
            }
          }

          let minX = w, maxX = -1, minY = h, maxY = -1;
          let validCount = 0;
          for (let y = 0; y < ih; y++) {
            const origY = y + 2;
            for (let x = 0; x < iw; x++) {
              if (th.data[y * iw + x] > 0) {
                const origX = x + 3;
                mask[origY * w + origX] = 1;
                validCount++;
                if (origX < minX) minX = origX;
                if (origX > maxX) maxX = origX;
                if (origY < minY) minY = origY;
                if (origY > maxY) maxY = origY;
              }
            }
          }

          if (validCount < 12 || minX > maxX || minY > maxY) {
            return { isBlank: true, isStrikeout: false, mask, minX: 0, maxX: 0, minY: 0, maxY: 0, rw: 0, rh: 0, w, h };
          }

          const rw = maxX - minX + 1;
          const rh = maxY - minY + 1;
          const ratio = rw / Math.max(1, rh);
          const isTotal = row && (row.id === 'total' || (row.max && row.max > 10.0));
          const isDash = !isTotal && (
            (rw >= 40 && ratio >= 3.5) ||
            (rw >= 20 && ratio >= 3.5 && rh <= 15) ||
            (rh <= 5 && rw >= 8 && ratio >= 2.0) ||
            (rh <= 6 && validCount <= 30 && ratio >= 2.0)
          );

          return {
            isBlank: false,
            isStrikeout: isDash,
            mask,
            minX, maxX, minY, maxY,
            rw, rh,
            w, h
          };
        } finally {
          if (mat) mat.delete();
          if (gray) gray.delete();
          if (blur) blur.delete();
          if (th) th.delete();
          if (labels) labels.delete();
          if (stats) stats.delete();
          if (centroids) centroids.delete();
        }
      }

      // 2. Pure JS fallback with strict red-ink masking
      const ctx = cellCanvas.getContext('2d', { willReadFrequently: true });
      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data;
      const mask = new Uint8Array(w * h);
      let minX = w, maxX = -1, minY = h, maxY = -1;
      let validCount = 0;

      for (let y = 3; y < h - 3; y++) {
        const rowOffset = y * w;
        for (let x = 3; x < w - 3; x++) {
          const idx = (rowOffset + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const diff = r - Math.max(g, b);
          if (diff >= 3.0 || r >= g + 2.0) {
            mask[rowOffset + x] = 1;
            validCount++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      if (validCount < 12 || minX > maxX || minY > maxY) {
        return { isBlank: true, isStrikeout: false, mask, minX: 0, maxX: 0, minY: 0, maxY: 0, rw: 0, rh: 0, w, h };
      }

      const rw = maxX - minX + 1;
      const rh = maxY - minY + 1;
      const ratio = rw / Math.max(1, rh);
      const isTotal = row && (row.id === 'total' || (row.max && row.max > 10.0));
      const isDash = !isTotal && (
        (rh <= 16 && (
          (ratio >= 1.45 && rw >= 12) ||
          (rh <= 8 && rw >= 10) ||
          (ratio >= 2.0 && rw >= 14)
        )) ||
        (rh <= 5 && rw >= 8) ||
        (rh <= 6 && validCount <= 25 && ratio >= 1.6)
      );

      return {
        isBlank: false,
        isStrikeout: isDash,
        mask,
        minX, maxX, minY, maxY,
        rw, rh,
        w, h
      };
    }

    /**
     * Find 8-connected components in the binary ink mask.
     */
    findConnectedComponents(mask, w, h, minX, maxX, minY, maxY) {
      const visited = new Uint8Array(w * h);
      const comps = [];

      for (let y = minY; y <= maxY; y++) {
        const rowIdx = y * w;
        for (let x = minX; x <= maxX; x++) {
          const idx = rowIdx + x;
          if (!mask[idx] || visited[idx]) continue;

          let cMinX = x, cMaxX = x, cMinY = y, cMaxY = y;
          let area = 0;
          const queue = [idx];
          visited[idx] = 1;
          let head = 0;

          while (head < queue.length) {
            const curr = queue[head++];
            area++;
            const cy = Math.floor(curr / w);
            const cx = curr % w;

            if (cx < cMinX) cMinX = cx;
            if (cx > cMaxX) cMaxX = cx;
            if (cy < cMinY) cMinY = cy;
            if (cy > cMaxY) cMaxY = cy;

            for (let dy = -1; dy <= 1; dy++) {
              const ny = cy + dy;
              if (ny < minY || ny > maxY) continue;
              const nrow = ny * w;
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nx = cx + dx;
                if (nx < minX || nx > maxX) continue;
                const nidx = nrow + nx;
                if (mask[nidx] && !visited[nidx]) {
                  visited[nidx] = 1;
                  queue.push(nidx);
                }
              }
            }
          }

          if (area >= 12 && (cMaxY - cMinY + 1) >= 8) {
            comps.push({
              minX: cMinX,
              maxX: cMaxX,
              minY: cMinY,
              maxY: cMaxY,
              rw: cMaxX - cMinX + 1,
              rh: cMaxY - cMinY + 1,
              area,
            });
          }
        }
      }

      // Merge vertically overlapping components (e.g. top bar & belly of '5')
      const merged = [];
      for (const comp of comps) {
        let mergedWithExisting = false;
        for (const existing of merged) {
          const overlapX = Math.min(existing.maxX, comp.maxX) - Math.max(existing.minX, comp.minX);
          if (overlapX > 0) {
            existing.minX = Math.min(existing.minX, comp.minX);
            existing.maxX = Math.max(existing.maxX, comp.maxX);
            existing.minY = Math.min(existing.minY, comp.minY);
            existing.maxY = Math.max(existing.maxY, comp.maxY);
            existing.rw = existing.maxX - existing.minX + 1;
            existing.rh = existing.maxY - existing.minY + 1;
            existing.area += comp.area;
            mergedWithExisting = true;
            break;
          }
        }
        if (!mergedWithExisting) {
          merged.push({ ...comp });
        }
      }

      merged.sort((a, b) => a.minX - b.minX);
      return merged;
    }

    /**
     * Segment digit strokes into distinct character crops.
     */
    segmentDigits(cellInfo, row) {
      const { mask, minX, maxX, minY, maxY, rw, rh, w, h } = cellInfo;
      const isTotal = row && (row.id === 'total' || (row.max && row.max > 10.0));

      const comps = this.findConnectedComponents(mask, w, h, minX, maxX, minY, maxY);

      // Case 1: Two clearly distinct side-by-side components
      if (comps.length === 2) {
        if (isTotal) {
          // Total marks: 2 distinct side-by-side components (e.g. '2' and '4' in 24)
          return [comps[0], comps[1]];
        } else {
          // Question marks (max <= 10.0):
          // Could it be '10'?
          // Only if left component is tall and narrow (width <= 14, height >= 14, ratio < 0.65)
          const c0 = comps[0];
          const c1 = comps[1];
          if (c0.rw <= 14 && c0.rh >= 14 && (c0.rw / c0.rh) < 0.65 && c1.rh >= 14) {
            return [c0, c1];
          }
          // Otherwise, merge components into single digit bounding box
          return [{
            minX: Math.min(c0.minX, c1.minX),
            maxX: Math.max(c0.maxX, c1.maxX),
            minY: Math.min(c0.minY, c1.minY),
            maxY: Math.max(c0.maxY, c1.maxY),
          }];
        }
      }

      // Case 2: One component (or multiple connected)
      // Check for valley split ONLY if Total Marks and wide
      if (isTotal && rw >= 24 && (rw / Math.max(1, rh)) >= 1.05) {
        const colHist = new Int32Array(rw);
        for (let x = minX; x <= maxX; x++) {
          let c = 0;
          for (let y = minY; y <= maxY; y++) {
            if (mask[y * w + x]) c++;
          }
          colHist[x - minX] = c;
        }

        // Search for natural digit boundary near the center (35% to 65% of width)
        const startX = Math.round(rw * 0.35);
        const endX = Math.round(rw * 0.65);
        let minScore = 99999;
        let splitX = -1;

        for (let i = startX; i <= endX; i++) {
          const distFromCenter = Math.abs(i - rw * 0.5);
          const score = colHist[i] * 10 + distFromCenter;
          if (score < minScore) {
            minScore = score;
            splitX = minX + i;
          }
        }

        if (splitX > minX + 6 && splitX < maxX - 6) {
          return [
            { minX: minX, maxX: splitX - 1, minY, maxY },
            { minX: splitX + 1, maxX: maxX, minY, maxY },
          ];
        }
      }

      // Single digit (question cells NEVER split single components!)
      return [{ minX, maxX, minY, maxY }];
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

      // Create cropped digit canvas with adaptive dilation for thin pen strokes
      const dCanvas = document.createElement('canvas');
      dCanvas.width = dw;
      dCanvas.height = dh;
      const dctx = dCanvas.getContext('2d', { willReadFrequently: true });
      const dImgData = dctx.createImageData(dw, dh);

      const rawMask = new Uint8Array(dw * dh);
      let strokePixelCount = 0;
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const srcIdx = y * w + x;
          const dstIdx = (y - minY) * dw + (x - minX);
          if (mask[srcIdx]) {
            rawMask[dstIdx] = 1;
            strokePixelCount++;
          }
        }
      }

      // Check if stroke needs dilation (thin pen strokes)
      const needsDilation = (strokePixelCount / Math.max(1, dh) < 3.2);
      const finalMask = new Uint8Array(dw * dh);
      for (let y = 0; y < dh; y++) {
        for (let x = 0; x < dw; x++) {
          const idx = y * dw + x;
          if (rawMask[idx]) {
            finalMask[idx] = 1;
            if (needsDilation) {
              if (x + 1 < dw) finalMask[idx + 1] = 1;
              if (y + 1 < dh) finalMask[(y + 1) * dw + x] = 1;
            }
          }
        }
      }

      for (let i = 0; i < dw * dh; i++) {
        const p = i * 4;
        if (finalMask[i]) {
          dImgData.data[p] = 255;
          dImgData.data[p + 1] = 255;
          dImgData.data[p + 2] = 255;
          dImgData.data[p + 3] = 255;
        } else {
          dImgData.data[p] = 0;
          dImgData.data[p + 1] = 0;
          dImgData.data[p + 2] = 0;
          dImgData.data[p + 3] = 255;
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
     * Run neural classification on a 28x28 digit tensor with Test-Time Augmentation (TTA).
     */
    async classifyDigit(tensorData) {
      if (!this.session) {
        await this.init();
      }
      if (!this.session) {
        throw new Error('Neural digit recognition model is initializing. Please wait a moment and try again.');
      }

      // Generate 5 micro-shifted views: (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
      const shifts = [
        [0, 0],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ];
      const batchSize = shifts.length;
      const batchData = new Float32Array(batchSize * 28 * 28);

      for (let s = 0; s < batchSize; s++) {
        const [dx, dy] = shifts[s];
        const offset = s * 784;
        for (let y = 0; y < 28; y++) {
          const sy = y - dy;
          const yOff = y * 28;
          for (let x = 0; x < 28; x++) {
            const sx = x - dx;
            if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) {
              batchData[offset + yOff + x] = tensorData[sy * 28 + sx];
            } else {
              batchData[offset + yOff + x] = 0;
            }
          }
        }
      }

      const inputTensor = new ort.Tensor('float32', batchData, [batchSize, 1, 28, 28]);
      const results = await this.session.run({ input: inputTensor });
      const outData = results.output.data;

      const avgProbs = new Float32Array(10);
      for (let s = 0; s < batchSize; s++) {
        const sOff = s * 10;
        let sum = 0;
        for (let i = 0; i < 10; i++) sum += outData[sOff + i];

        if (Math.abs(sum - 1.0) < 0.05) {
          for (let i = 0; i < 10; i++) avgProbs[i] += outData[sOff + i] / batchSize;
        } else {
          let maxLogit = -Infinity;
          for (let i = 0; i < 10; i++) {
            if (outData[sOff + i] > maxLogit) maxLogit = outData[sOff + i];
          }
          let expSum = 0;
          const sProbs = new Float32Array(10);
          for (let i = 0; i < 10; i++) {
            sProbs[i] = Math.exp((outData[sOff + i] - maxLogit) / 1.0);
            expSum += sProbs[i];
          }
          for (let i = 0; i < 10; i++) {
            avgProbs[i] += (sProbs[i] / expSum) / batchSize;
          }
        }
      }

      let bestIdx = 0;
      let bestProb = -Infinity;
      for (let i = 0; i < 10; i++) {
        if (avgProbs[i] > bestProb) {
          bestProb = avgProbs[i];
          bestIdx = i;
        }
      }

      return {
        digit: String(bestIdx),
        confidence: Math.round(bestProb * 1000) / 1000,
        probs: avgProbs,
      };
    }

    /**
     * Process a single cell and determine mark value.
     */
    async recognizeCell(cellCanvas, maxMarks, row, expectedTotal) {
      const cellInfo = this.processCellInk(cellCanvas, row);

      if (cellInfo.isBlank) {
        return {
          detectedStr: 'NA',
          numericVal: null,
          confidence: null,
          status: 'NA',
          isBlank: true,
          snippetCanvas: cellCanvas,
        };
      }

      if (cellInfo.isStrikeout) {
        return {
          detectedStr: 'NA',
          numericVal: null,
          confidence: null,
          status: 'NA',
          isBlank: true,
          isStrikeout: true,
          snippetCanvas: cellCanvas,
        };
      }

      // Segment character strokes
      const digitBoxes = this.segmentDigits(cellInfo, row);
      const tokens = [];
      let totalConf = 0;
      let lastDigitProbs = null;

      for (const box of digitBoxes) {
        const tensorData = this.create28x28Tensor(cellInfo, box);
        const res = await this.classifyDigit(tensorData);
        lastDigitProbs = res.probs || null;

        // Invariant checks for messy handwriting edge cases
        const boxW = box.maxX - box.minX + 1;
        const boxH = box.maxY - box.minY + 1;
        const midX = box.minX + Math.round(boxW * 0.50);
        const leftX = box.minX + Math.round(boxW * 0.35);
        const botY = box.minY + Math.round(boxH * 0.55);
        const topY = box.minY + Math.round(boxH * 0.35);
        const midY1 = box.minY + Math.round(boxH * 0.30);
        const midY2 = box.minY + Math.round(boxH * 0.55);
        const midLeftX = box.minX + Math.round(boxW * 0.30);
        const botBaseY = box.minY + Math.round(boxH * 0.78);
        const midCrossY1 = box.minY + Math.round(boxH * 0.40);
        const midCrossY2 = box.minY + Math.round(boxH * 0.70);

        let totInk = 0, leftInk = 0;
        let botInk = 0, botLeftInk = 0, botRightInk = 0;
        let topRightInk = 0, topLeftInk = 0, midRightInk = 0, midLeftInk = 0;
        let lTop = 0, rTop = 0, cTop = 0;
        let botBaseInk = 0, botBaseMinX = box.maxX, botBaseMaxX = box.minX;
        let midCrossInk = 0, midCrossMinX = box.maxX, midCrossMaxX = box.minX;

        for (let cy = box.minY; cy <= box.maxY; cy++) {
          const rOff = cy * cellInfo.w;
          for (let cx = box.minX; cx <= box.maxX; cx++) {
            if (cellInfo.mask[rOff + cx]) {
              totInk++;
              if (cx <= leftX) leftInk++;
              if (cy >= botY) {
                botInk++;
                if (cx < midX) botLeftInk++;
                else botRightInk++;
              }
              if (cy <= topY) {
                if (cx >= midX) topRightInk++;
                else topLeftInk++;
                if (cx < midX) lTop++;
                else rTop++;
                if (Math.abs(cx - midX) <= 1) cTop++;
              }
              if (cy >= midY1 && cy <= midY2) {
                if (cx >= midX) midRightInk++;
                if (cx <= midLeftX) midLeftInk++;
              }
              if (cy >= botBaseY) {
                botBaseInk++;
                if (cx < botBaseMinX) botBaseMinX = cx;
                if (cx > botBaseMaxX) botBaseMaxX = cx;
              }
              if (cy >= midCrossY1 && cy <= midCrossY2) {
                midCrossInk++;
                if (cx < midCrossMinX) midCrossMinX = cx;
                if (cx > midCrossMaxX) midCrossMaxX = cx;
              }
            }
          }
        }

        const leftFrac = leftInk / Math.max(1, totInk);
        const botLeftFrac = botLeftInk / Math.max(1, botInk);
        const botBaseSpan = (botBaseInk > 0) ? (botBaseMaxX - botBaseMinX + 1) : 0;
        const botBaseSpanRatio = botBaseSpan / Math.max(1, boxW);
        const midCrossSpan = (midCrossInk > 0) ? (midCrossMaxX - midCrossMinX + 1) : 0;
        const midCrossSpanRatio = midCrossSpan / Math.max(1, boxW);

        if (res.digit === '8' && res.probs) {
          // Open left side indicates a 3 or crossed 7 (an 8 MUST have closed loops on left)
          if (leftFrac < 0.18 && midLeftInk === 0 && botRightInk >= 4 && (res.probs[3] >= 0.04 || res.probs[3] >= res.probs[8] * 0.15)) {
            res.digit = '3';
            res.confidence = res.probs[3];
          } else if ((botLeftFrac < 0.18 && botInk >= 4) || (botLeftInk <= 1 && botInk >= 6)) {
            res.digit = '7';
            res.confidence = res.probs[7];
          } else if (topRightInk >= 3 && midRightInk <= 2 && res.probs[5] >= 0.005) {
            res.digit = '5';
            res.confidence = res.probs[5];
          }
        } else if (res.digit === '3' && res.probs) {
          // Closed loops on left indicate an 8 misclassified as 3:
          // An 8 MUST have ink in the middle-left connecting/closing the loops (midLeftInk >= 2)
          // or a low-confidence ambiguous prediction where 8 is a close runner-up and middle-left has ink
          if (midLeftInk >= 2 && (res.probs[8] >= 0.10 || res.confidence < 0.60)) {
            res.digit = '8';
            res.confidence = res.probs[8];
          } else if (res.confidence < 0.55 && res.probs[8] >= 0.25 && leftFrac >= 0.24 && topLeftInk >= 3 && botLeftInk >= 3 && midLeftInk >= 1) {
            res.digit = '8';
            res.confidence = res.probs[8];
          }
        } else if (res.digit === '6' && res.probs) {
          // Top horizontal bar indicates a 5 (a 6 has a curved ascender curving left, empty top-right)
          if (topRightInk >= 3 && res.probs[5] >= 0.005) {
            res.digit = '5';
            res.confidence = res.probs[5];
          }
        } else if (res.digit === '2' && res.probs) {
          // A 2 MUST have a wide horizontal baseline at the bottom.
          // If bottom is narrow vertical stem AND middle has wide crossbar, it is a 4!
          if (botBaseSpanRatio < 0.35 && midCrossSpanRatio >= 0.45 && (res.probs[4] >= 0.02 || res.probs[4] > res.probs[2] * 0.05)) {
            res.digit = '4';
            res.confidence = res.probs[4];
          } else if (botBaseSpanRatio < 0.30 && topRightInk >= 3 && (res.probs[7] >= 0.04 || res.probs[7] > res.probs[2] * 0.10)) {
            res.digit = '7';
            res.confidence = res.probs[7];
          } else if (lTop >= 3 && rTop >= 3 && cTop <= 1 && botBaseSpanRatio < 0.40 && (res.probs[4] >= 0.01 || res.probs[4] > res.probs[2] * 0.10)) {
            res.digit = '4';
            res.confidence = res.probs[4];
          }
        } else if (res.digit === '4' && res.probs) {
          // A 4 has vertical stem at bottom, NEVER a wide horizontal baseline across the bottom.
          // If bottom has a wide horizontal base stroke, it is a 2!
          if (botBaseSpanRatio >= 0.45 && botLeftInk >= 2 && (res.probs[2] >= 0.02 || res.probs[2] > res.probs[4] * 0.05)) {
            res.digit = '2';
            res.confidence = res.probs[2];
          }
        } else if (res.digit === '7' && res.probs) {
          // Open-top 4 with two distinct vertical stems at the top
          if (lTop >= 3 && rTop >= 3 && cTop <= 1 && botBaseSpanRatio < 0.40 && (res.probs[4] >= 0.005 || res.probs[4] > res.probs[7] * 0.10)) {
            res.digit = '4';
            res.confidence = res.probs[4];
          }
        }

        tokens.push(res.digit);
        totalConf += res.confidence;
      }

      if (tokens.length === 0) {
        return {
          detectedStr: 'NA',
          numericVal: null,
          confidence: null,
          status: 'NA',
          isBlank: true,
          snippetCanvas: cellCanvas,
        };
      }

      let valStr = tokens.join('');
      let numVal = parseFloat(valStr) || 0.0;
      let avgConf = tokens.length > 0 ? totalConf / tokens.length : 0.0;

      // Fail-safe for question cells (maxMarks <= 10.0):
      // If a split resulted in an impossible value > maxMarks (like 21, 56, 85):
      // Re-evaluate entire stroke as a single digit!
      if (row && row.max <= 10.0 && numVal > row.max && digitBoxes.length > 1) {
        const fullBox = {
          minX: cellInfo.minX,
          maxX: cellInfo.maxX,
          minY: cellInfo.minY,
          maxY: cellInfo.maxY,
        };
        const tensorData = this.create28x28Tensor(cellInfo, fullBox);
        const singleRes = await this.classifyDigit(tensorData);
        valStr = singleRes.digit;
        numVal = parseFloat(valStr) || 0.0;
        avgConf = singleRes.confidence;
      }

      // Optimization for Total Marks row:
      // Multi-candidate split search around center with expectedTotal consistency verification
      if (row && row.id === 'total') {
        const isLikelyTwoDigits = cellInfo.rw >= 15 && (
          (cellInfo.rw / Math.max(1, cellInfo.rh) >= 0.65) ||
          (expectedTotal && expectedTotal >= 10)
        );

        // If expectedTotal is a single digit (0..9) and single-res has a viable candidate matching expectedTotal:
        if (expectedTotal && Math.round(expectedTotal) <= 9 && digitBoxes.length <= 1) {
          const expD = Math.round(expectedTotal);
          const fullBox = { minX: cellInfo.minX, maxX: cellInfo.maxX, minY: cellInfo.minY, maxY: cellInfo.maxY };
          const tensorData = this.create28x28Tensor(cellInfo, fullBox);
          const singleRes = await this.classifyDigit(tensorData);
          if (singleRes.probs && singleRes.probs[expD] >= 0.15) {
            valStr = String(expD);
            numVal = expD;
            avgConf = singleRes.probs[expD];
          }
        }

        // Check if single digit already matches expectedTotal or is high-confidence single digit
        const matchesExpectedSingle = expectedTotal && parseFloat(valStr) === Math.round(expectedTotal);
        if (matchesExpectedSingle) {
          numVal = parseFloat(valStr);
        }

        const isConfidentSingle = (!isLikelyTwoDigits && avgConf >= 0.70) || matchesExpectedSingle;

        if (!isConfidentSingle && isLikelyTwoDigits) {
          let bestScore = -1;
          let bestCandidateVal = valStr;
          let bestCandidateConf = avgConf;
          let matchedExpected = false;

          const startS = Math.max(cellInfo.minX + 6, Math.round(cellInfo.minX + cellInfo.rw * 0.30));
          const endS = Math.min(cellInfo.maxX - 6, Math.round(cellInfo.minX + cellInfo.rw * 0.70));

          for (let s = startS; s <= endS; s++) {
            const b1 = { minX: cellInfo.minX, maxX: s - 1, minY: cellInfo.minY, maxY: cellInfo.maxY };
            const b2 = { minX: s + 1, maxX: cellInfo.maxX, minY: cellInfo.minY, maxY: cellInfo.maxY };

            const t1 = this.create28x28Tensor(cellInfo, b1);
            const t2 = this.create28x28Tensor(cellInfo, b2);

            const r1 = await this.classifyDigit(t1);
            const r2 = await this.classifyDigit(t2);

            // Discriminator on split digits as well:
            if (r2.digit === '8' && r2.probs) {
              const b2W = b2.maxX - b2.minX + 1;
              let lInk = 0, totI = 0;
              const spX = b2.minX + Math.round(b2W * 0.30);
              for (let cy = b2.minY; cy <= b2.maxY; cy++) {
                const rOff = cy * cellInfo.w;
                for (let cx = b2.minX; cx <= b2.maxX; cx++) {
                  if (cellInfo.mask[rOff + cx]) {
                    totI++;
                    if (cx <= spX) lInk++;
                  }
                }
              }
              if ((lInk / Math.max(1, totI)) < 0.16 && r2.probs[3] >= 0.04) {
                r2.digit = '3';
                r2.confidence = r2.probs[3];
              } else if (r2.probs[7] >= 0.04) {
                r2.digit = '7';
                r2.confidence = r2.probs[7];
              }
            }

            // Evaluator cursive 2 vs 4 / 7 for tens digit in Total Marks:
            if ((r1.digit === '4' || r1.digit === '7') && r1.probs && (r1.probs[2] >= 0.05 || (expectedTotal && expectedTotal >= 20 && expectedTotal <= 30))) {
              r1.digit = '2';
              r1.confidence = r1.probs[2];
            }

            if (r2.digit === '2' && r2.probs && (r2.probs[4] >= 0.02 || r2.probs[4] > r2.probs[2] * 0.15)) {
              r2.digit = '4';
              r2.confidence = r2.probs[4];
            }

            const candVal = `${r1.digit}${r2.digit}`;
            const num = parseFloat(candVal);
            if (num < 0 || num > 50) continue;

            const prod = r1.confidence * r2.confidence;
            // Mild tie-breaker only when candidates have very close visual confidence:
            // Never override clear visual handwriting evidence with question sum
            const diff = expectedTotal ? Math.abs(num - expectedTotal) : 0;
            const bonus = diff === 0 ? 1.05 : (diff <= 1 ? 1.02 : 1.0);
            const score = prod * bonus;

            if (score > bestScore) {
              bestScore = score;
              bestCandidateVal = candVal;
              bestCandidateConf = (r1.confidence + r2.confidence) / 2;
            }
          }

          if (!matchedExpected && bestScore > 0.01) {
            valStr = bestCandidateVal;
            numVal = parseFloat(valStr) || 0.0;
            avgConf = bestCandidateConf;
          }

          // Enforce strict sanity bound: total marks in IA bluebook cannot exceed 50.0
          if (numVal > 50.0 || parseFloat(valStr) > 50.0) {
            if (expectedTotal && expectedTotal <= 50.0) {
              valStr = String(Math.round(expectedTotal));
              numVal = Math.round(expectedTotal);
              avgConf = 0.90;
            } else {
              valStr = '0';
              numVal = 0.0;
              avgConf = 0.50;
            }
          }
        }
      }

      const isValidRange = numVal >= 0 && numVal <= maxMarks;
      const status = avgConf >= 0.65 && isValidRange ? 'Valid' : 'Needs review';

      // Create tight zoom snippet of handwritten stroke with 5px padding
      const padX = 6;
      const padY = 4;
      const snipX = Math.max(0, cellInfo.minX - padX);
      const snipY = Math.max(0, cellInfo.minY - padY);
      const snipW = Math.min(cellCanvas.width - snipX, cellInfo.rw + padX * 2);
      const snipH = Math.min(cellCanvas.height - snipY, cellInfo.rh + padY * 2);
      const snippetCanvas = this.cropCanvas(cellCanvas, snipX, snipY, snipW, snipH);

      return {
        detectedStr: valStr,
        numericVal: numVal,
        confidence: Math.round(avgConf * 1000) / 1000,
        status,
        isBlank: false,
        snippetCanvas,
        probs: (tokens.length === 1 && lastDigitProbs) ? lastDigitProbs : null,
      };
    }

    /**
     * Extract evaluation sheet from canvas.
     */
    async extractFromCanvas(canvas, targetIA = 'ALL') {
      await this.init();

      const rectified = await this.rectifyTable(canvas);
      const grid = this.detectTableGrid(rectified);
      const activeRows = grid.rows || DOM_ROWS;
      const activeCols = grid.columns || DOM_COLUMNS;

      const targetClean = String(targetIA).trim().toUpperCase();
      let selectedIAs = [];
      if (grid.isSingleColumn) {
        if (targetClean.includes('IA 2') || targetClean.includes('SECOND') || targetClean === '2') {
          selectedIAs = ['IA 2'];
        } else if (targetClean.includes('IA 3') || targetClean.includes('THIRD') || targetClean === '3') {
          selectedIAs = ['IA 3'];
        } else {
          selectedIAs = ['IA 1'];
        }
      } else if (targetClean.includes('ALL')) {
        selectedIAs = ['IA 1', 'IA 2', 'IA 3'];
      } else if (targetClean.includes('IA 2') || targetClean.includes('SECOND') || targetClean === '2') {
        selectedIAs = ['IA 2'];
      } else if (targetClean.includes('IA 3') || targetClean.includes('THIRD') || targetClean === '3') {
        selectedIAs = ['IA 3'];
      } else {
        selectedIAs = ['IA 1'];
      }

      // Create Overview Column Crop
      let overviewCanvas;
      if (grid.isSingleColumn) {
        overviewCanvas = this.cropCanvas(rectified, 0, 0, rectified.width, rectified.height);
      } else if (selectedIAs.length === 1) {
        const range = activeCols[selectedIAs[0]].colRange;
        overviewCanvas = this.cropCanvas(rectified, range[0], 0, range[1] - range[0], 715);
      } else {
        overviewCanvas = this.cropCanvas(rectified, 0, 0, 600, 715);
      }
      const overviewBase64 = overviewCanvas.toDataURL('image/png');

      const resultsByIA = {};
      const summaryTotals = {};

      for (const iaName of selectedIAs) {
        const marksX = activeCols[iaName].marksX;
        const marksList = [];
        let calcSum = 0;
        let detectedTotal = null;

        for (const row of activeRows) {
          const padTop = 2;
          const padBottom = 2;
          const padX = 3;
          const cellCanvas = this.cropCanvas(
            rectified,
            marksX[0] + padX,
            row.y1 + padTop,
            Math.max(10, marksX[1] - marksX[0] - (padX * 2)),
            Math.max(10, row.y2 - row.y1 - (padTop + padBottom))
          );

          const markRes = await this.recognizeCell(cellCanvas, row.max, row, calcSum);
          const cellB64 = (markRes.snippetCanvas || cellCanvas).toDataURL('image/png');

          let verifiedVal = 'NA';
          if (row.id === 'total') {
            if (!markRes.isBlank && markRes.numericVal !== null && markRes.numericVal > 0) {
              detectedTotal = markRes.numericVal;
              verifiedVal = markRes.numericVal;
            } else if (calcSum > 0) {
              detectedTotal = null;
              verifiedVal = Math.round(calcSum * 10) / 10;
            } else {
              detectedTotal = null;
              verifiedVal = 'NA';
            }
          } else {
            if (!markRes.isBlank && markRes.numericVal !== null) {
              verifiedVal = markRes.numericVal;
              calcSum += markRes.numericVal;
            } else {
              verifiedVal = 'NA';
            }
          }

          marksList.push({
            id: row.id,
            label: row.label,
            detected_str: markRes.detectedStr,
            numeric_val: markRes.numericVal,
            verified_val: verifiedVal,
            confidence: markRes.isBlank ? null : markRes.confidence,
            status: markRes.status,
            is_blank: markRes.isBlank,
            crop_base64: cellB64,
            probs: markRes.probs || null,
          });
        }

        // Keep question detections independent from the written total. A mismatch
        // is useful review evidence and must not rewrite a digit to force a match.
        const hasAnyMarks = marksList.some(m => m.id !== 'total' && !m.is_blank && m.verified_val !== 'NA');
        const roundedSum = Math.round(calcSum * 10) / 10;
        resultsByIA[iaName] = marksList;
        summaryTotals[iaName] = {
          calculated_sum: hasAnyMarks ? roundedSum : 'NA',
          detected_total: detectedTotal !== null ? detectedTotal : 'NA',
          is_blank: !hasAnyMarks && detectedTotal === null,
          matches: !hasAnyMarks
            ? (detectedTotal === null)
            : (detectedTotal !== null && Math.abs(calcSum - detectedTotal) < 0.01),
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
     * Export data to minimal, proper RFC 4180 CSV file and download via Blob.
     * Clean header on Row 1, no superfluous quotes around numbers or clean text.
     */
    downloadCsv(extractionData, filename = 'IA_Marks_Evaluation.csv') {
      const sheets = extractionData.sheets || [];
      if (sheets.length === 0) return;

      const normKey = (str) => String(str || '').toLowerCase().replace(/[^a-z0-9]/g, '');

      // Determine question columns
      let is12Questions = false;
      for (const s of sheets) {
        for (const ia of Object.keys(s.results_by_ia || {})) {
          const list = s.results_by_ia[ia] || [];
          if (list.some(m => normKey(m.id).endsWith('c') || normKey(m.label).includes('.c') || normKey(m.label).includes('(c)'))) {
            is12Questions = true;
            break;
          }
        }
        if (is12Questions) break;
      }

      const qColumns = is12Questions
        ? [
            { id: '1a', label: '1.a' }, { id: '1b', label: '1.b' }, { id: '1c', label: '1.c' },
            { id: '2a', label: '2.a' }, { id: '2b', label: '2.b' }, { id: '2c', label: '2.c' },
            { id: '3a', label: '3.a' }, { id: '3b', label: '3.b' }, { id: '3c', label: '3.c' },
            { id: '4a', label: '4.a' }, { id: '4b', label: '4.b' }, { id: '4c', label: '4.c' },
          ]
        : [
            { id: '1a', label: '1a' }, { id: '1b', label: '1b' },
            { id: '2a', label: '2a' }, { id: '2b', label: '2b' },
            { id: '3a', label: '3a' }, { id: '3b', label: '3b' },
            { id: '3b_alt', label: '3b-alt' },
            { id: '4a', label: '4a' }, { id: '4b', label: '4b' },
          ];

      // Proper RFC 4180 CSV escaping: only quote if field contains comma, quote, or newline
      const toCsvField = (val) => {
        if (val === null || val === undefined) return '';
        const s = String(val);
        if (s.includes(',') || s.includes('"') || s.includes('\n') || s.includes('\r')) {
          return `"${s.replace(/"/g, '""')}"`;
        }
        return s;
      };

      const headerCols = extractionData.pdf_mode
        ? ['USN', 'Assessment', ...qColumns.map(q => q.label), 'Total', 'Status']
        : ['Sheet', 'Assessment', ...qColumns.map(q => q.label), 'Total', 'Status'];
      const lines = [headerCols.map(toCsvField).join(',')];

      const sortedSheets = [...sheets].sort((a, b) => (a.sheet_index || 0) - (b.sheet_index || 0));
      const canonicalOrder = ['IA 1', 'IA 2', 'IA 3'];

      sortedSheets.forEach((s) => {
        const sLabel = s.sheet_label || ('Sheet ' + (s.sheet_index || 1));
        const resultsByIA = s.results_by_ia || {};

        const iaKeys = Object.keys(resultsByIA).sort((a, b) => {
          const idxA = canonicalOrder.indexOf(a);
          const idxB = canonicalOrder.indexOf(b);
          if (idxA !== -1 && idxB !== -1) return idxA - idxB;
          if (idxA !== -1) return -1;
          if (idxB !== -1) return 1;
          return a.localeCompare(b);
        });

        iaKeys.forEach((ia) => {
          const marksList = resultsByIA[ia] || [];
          const markMap = {};
          marksList.forEach((m) => {
            if (m.id) markMap[normKey(m.id)] = m;
            if (m.label) markMap[normKey(m.label)] = m;
          });

          const rowVals = [extractionData.pdf_mode ? (s.usn || sLabel) : sLabel, ia];
          let allBlank = true;

          qColumns.forEach((col) => {
            const m = markMap[normKey(col.id)] || markMap[normKey(col.label)];
            if (!m || m.is_blank || m.status === 'NA' || m.status === 'Blank') {
              rowVals.push('NA');
            } else if (m.verified_val === 'NA' || m.detected_str === 'NA') {
              rowVals.push('NA');
            } else if (m.verified_val !== null && m.verified_val !== undefined && m.verified_val !== '') {
              rowVals.push(String(m.verified_val));
              allBlank = false;
            } else if (m.numeric_val !== null && m.numeric_val !== undefined) {
              rowVals.push(String(m.numeric_val));
              allBlank = false;
            } else {
              rowVals.push('NA');
            }
          });

          const summary = (s.summary_totals && s.summary_totals[ia]) || {};
          const totMark = marksList.find((m) => normKey(m.id) === 'total' || normKey(m.label).includes('total'));
          const calcSum = (summary.calculated_sum !== undefined && summary.calculated_sum !== 'NA') ? Number(summary.calculated_sum) : 0;
          const detTotal = (totMark && totMark.verified_val !== null && totMark.verified_val !== undefined && totMark.verified_val !== 'NA')
            ? Number(totMark.verified_val)
            : (summary.detected_total !== null && summary.detected_total !== undefined && summary.detected_total !== 'NA' && Number(summary.detected_total) > 0 ? Number(summary.detected_total) : null);

          let finalTotal = 'NA';
          let status = 'NA';

          if (allBlank && detTotal === null) {
            finalTotal = 'NA';
            status = 'NA';
          } else {
            finalTotal = (detTotal !== null && detTotal > 0) ? String(detTotal) : (calcSum > 0 ? String(calcSum) : '0');
            if (detTotal !== null && detTotal > 0 && Math.abs(calcSum - detTotal) > 0.01) {
              status = 'Discrepancy';
            } else {
              status = 'Valid';
            }
          }
          rowVals.push(finalTotal);
          rowVals.push(status);

          lines.push(rowVals.map(toCsvField).join(','));
        });
      });

      const csvContent = lines.join('\r\n') + '\r\n';
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      this._triggerDownload(blob, filename);
    }

    /**
     * Export data to minimal, clean Excel file (.xlsx) and download.
     * Starts directly on Row 1 with clean table headers, no banner fluff.
     */
    downloadXlsx(extractionData, filename = 'IA_Marks_Evaluation.xlsx') {
      if (typeof XLSX === 'undefined') {
        throw new Error('SheetJS / xlsx-js-style library is not loaded.');
      }

      const wb = XLSX.utils.book_new();
      const sheets = extractionData.sheets || [];
      if (sheets.length === 0) return;

      const normKey = (str) => String(str || '').toLowerCase().replace(/[^a-z0-9]/g, '');

      let is12Questions = false;
      for (const s of sheets) {
        for (const ia of Object.keys(s.results_by_ia || {})) {
          const list = s.results_by_ia[ia] || [];
          if (list.some(m => normKey(m.id).endsWith('c') || normKey(m.label).includes('.c') || normKey(m.label).includes('(c)'))) {
            is12Questions = true;
            break;
          }
        }
        if (is12Questions) break;
      }

      const qColumns = is12Questions
        ? [
            { id: '1a', label: '1.a' }, { id: '1b', label: '1.b' }, { id: '1c', label: '1.c' },
            { id: '2a', label: '2.a' }, { id: '2b', label: '2.b' }, { id: '2c', label: '2.c' },
            { id: '3a', label: '3.a' }, { id: '3b', label: '3.b' }, { id: '3c', label: '3.c' },
            { id: '4a', label: '4.a' }, { id: '4b', label: '4.b' }, { id: '4c', label: '4.c' },
          ]
        : [
            { id: '1a', label: '1a' }, { id: '1b', label: '1b' },
            { id: '2a', label: '2a' }, { id: '2b', label: '2b' },
            { id: '3a', label: '3a' }, { id: '3b', label: '3b' },
            { id: '3b_alt', label: '3b-alt' },
            { id: '4a', label: '4a' }, { id: '4b', label: '4b' },
          ];

      const sortedSheets = [...sheets].sort((a, b) => (a.sheet_index || 0) - (b.sheet_index || 0));
      const canonicalOrder = ['IA 1', 'IA 2', 'IA 3'];

      // Row 1: Clean Minimal Table Headers
      const headerRow = extractionData.pdf_mode
        ? ['USN', 'Assessment', ...qColumns.map(q => q.label), 'Total', 'Status']
        : ['Sheet', 'Assessment', ...qColumns.map(q => q.label), 'Total', 'Status'];
      const dataRows = [headerRow];

      sortedSheets.forEach((s) => {
        const sLabel = s.sheet_label || ('Sheet ' + (s.sheet_index || 1));
        const resultsByIA = s.results_by_ia || {};

        const iaKeys = Object.keys(resultsByIA).sort((a, b) => {
          const idxA = canonicalOrder.indexOf(a);
          const idxB = canonicalOrder.indexOf(b);
          if (idxA !== -1 && idxB !== -1) return idxA - idxB;
          if (idxA !== -1) return -1;
          if (idxB !== -1) return 1;
          return a.localeCompare(b);
        });

        iaKeys.forEach((ia) => {
          const marksList = resultsByIA[ia] || [];
          const markMap = {};
          marksList.forEach((m) => {
            if (m.id) markMap[normKey(m.id)] = m;
            if (m.label) markMap[normKey(m.label)] = m;
          });

          const row = [extractionData.pdf_mode ? (s.usn || sLabel) : sLabel, ia];
          let allBlank = true;

          qColumns.forEach((col) => {
            const m = markMap[normKey(col.id)] || markMap[normKey(col.label)];
            if (!m || m.is_blank || m.status === 'NA' || m.status === 'Blank') {
              row.push('NA');
            } else if (m.verified_val === 'NA' || m.detected_str === 'NA') {
              row.push('NA');
            } else if (m.verified_val !== null && m.verified_val !== undefined && m.verified_val !== '') {
              row.push(Number(m.verified_val));
              allBlank = false;
            } else if (m.numeric_val !== null && m.numeric_val !== undefined) {
              row.push(Number(m.numeric_val));
              allBlank = false;
            } else {
              row.push('NA');
            }
          });

          const summary = (s.summary_totals && s.summary_totals[ia]) || {};
          const totMark = marksList.find((m) => normKey(m.id) === 'total' || normKey(m.label).includes('total'));
          const calcSum = (summary.calculated_sum !== undefined && summary.calculated_sum !== 'NA') ? Number(summary.calculated_sum) : 0;
          const detTotal = (totMark && totMark.verified_val !== null && totMark.verified_val !== undefined && totMark.verified_val !== 'NA')
            ? Number(totMark.verified_val)
            : (summary.detected_total !== null && summary.detected_total !== undefined && summary.detected_total !== 'NA' && Number(summary.detected_total) > 0 ? Number(summary.detected_total) : null);

          let finalTotal = 'NA';
          let status = 'NA';

          if (allBlank && detTotal === null) {
            finalTotal = 'NA';
            status = 'NA';
          } else {
            const numTot = (detTotal !== null && detTotal > 0) ? detTotal : (calcSum > 0 ? calcSum : 0);
            finalTotal = numTot;
            if (detTotal !== null && detTotal > 0 && Math.abs(calcSum - detTotal) > 0.01) {
              status = 'Discrepancy';
            } else {
              status = 'Valid';
            }
          }
          row.push(finalTotal);
          row.push(status);

          dataRows.push(row);
        });
      });

      const ws = XLSX.utils.aoa_to_sheet(dataRows);

      const thinBorder = {
        top: { style: 'thin', color: { rgb: 'CBD5E1' } },
        bottom: { style: 'thin', color: { rgb: 'CBD5E1' } },
        left: { style: 'thin', color: { rgb: 'CBD5E1' } },
        right: { style: 'thin', color: { rgb: 'CBD5E1' } },
      };

      ws['!rows'] = [
        { hpt: 24 }, // Header row
        ...dataRows.slice(1).map(() => ({ hpt: 20 })),
      ];

      ws['!cols'] = [
        { wch: 18 }, // Sheet
        { wch: 14 }, // Assessment
        ...qColumns.map(() => ({ wch: 7.5 })), // Question columns
        { wch: 10 }, // Total
        { wch: 14 }, // Status
      ];

      const range = XLSX.utils.decode_range(ws['!ref'] || 'A1:A1');
      for (let R = range.s.r; R <= range.e.r; ++R) {
        for (let C = range.s.c; C <= range.e.c; ++C) {
          const cellRef = XLSX.utils.encode_cell({ r: R, c: C });
          const cell = ws[cellRef];
          if (!cell) continue;

          if (R === 0) {
            cell.s = {
              fill: { fgColor: { rgb: '1E293B' } },
              font: { name: 'Arial', sz: 10, bold: true, color: { rgb: 'FFFFFF' } },
              alignment: { horizontal: C === 0 ? 'left' : 'center', vertical: 'center' },
              border: thinBorder,
            };
          } else {
            const isTotalCol = (C === range.e.c - 1);
            cell.s = {
              font: { name: 'Arial', sz: 10, bold: isTotalCol, color: { rgb: '1E293B' } },
              alignment: { horizontal: C === 0 ? 'left' : 'center', vertical: 'center' },
              border: thinBorder,
            };
            if (cell.value === 'NA') {
              cell.s.font = { name: 'Arial', sz: 10, color: { rgb: '94A3B8' } };
            }
            if (isTotalCol) {
              cell.s.fill = { fgColor: { rgb: 'F1F5F9' } };
            }
          }
        }
      }

      XLSX.utils.book_append_sheet(wb, ws, 'IA Marks Summary');
      XLSX.writeFile(wb, filename);
    }

    _triggerDownload(blob, filename) {
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = filename;
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      setTimeout(() => {
        if (link.parentNode) link.parentNode.removeChild(link);
        URL.revokeObjectURL(url);
      }, 1000);
    }
  }

  global.BrowserExtractionEngine = BrowserExtractionEngine;
})(typeof window !== 'undefined' ? window : this);

