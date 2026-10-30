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
     * Extract red-pen strokes and check strikeout status.
     * Uses adaptive background thresholding to eliminate paper tint and shadow false-positives.
     */
    processCellInk(cellCanvas) {
      const w = cellCanvas.width;
      const h = cellCanvas.height;
      const ctx = cellCanvas.getContext('2d', { willReadFrequently: true });
      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data;

      // 1. Compute ambient paper background red difference median
      // Sample inner region (shave 4px to avoid border lines)
      const diffValues = [];
      for (let y = 4; y < h - 4; y += 2) {
        const row = y * w;
        for (let x = 4; x < w - 4; x += 2) {
          const idx = (row + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          diffValues.push(r - Math.max(g, b));
        }
      }
      diffValues.sort((a, b) => a - b);
      const bgDiff = diffValues.length > 0 ? diffValues[Math.floor(diffValues.length * 0.5)] : 0;
      const redThreshold = Math.max(10, bgDiff + 7);

      // Stroke mask (1 for stroke, 0 for background)
      const mask = new Uint8Array(w * h);
      let redCount = 0;
      let darkCount = 0;

      let minX = w,
        maxX = -1,
        minY = h,
        maxY = -1;

      // Scan inner region (shave 4px border to ignore table lines)
      for (let y = 4; y < h - 4; y++) {
        const row = y * w;
        for (let x = 4; x < w - 4; x++) {
          const idx = (row + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          // Red ink isolation with background adaptation
          const diff = r - Math.max(g, b);
          if (diff >= redThreshold && r > 75) {
            mask[row + x] = 1;
            redCount++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          } else {
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            if (lum < 75 && r < 90 && g < 90 && b < 90) darkCount++;
          }
        }
      }

      // If red ink is insufficient, check for dark pen ink (black/blue ink)
      if (redCount < 20 && darkCount >= 35) {
        for (let y = 5; y < h - 5; y++) {
          const row = y * w;
          for (let x = 5; x < w - 5; x++) {
            const idx = (row + x) * 4;
            const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            if (lum < 75 && data[idx] < 90 && data[idx + 1] < 90 && data[idx + 2] < 90) {
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

      if (totalStrokes < 20 || minX > maxX || minY > maxY) {
        return { isBlank: true, isStrikeout: false, mask, minX, maxX, minY, maxY, w, h };
      }

      const rw = maxX - minX + 1;
      const rh = maxY - minY + 1;
      const ratio = rw / Math.max(1, rh);

      if (rw < 6 || rh < 10) {
        return { isBlank: true, isStrikeout: false, mask, minX, maxX, minY, maxY, w, h };
      }

      // Check max vertical stroke thickness across columns
      let maxColSpan = 0;
      for (let x = minX; x <= maxX; x++) {
        let span = 0;
        for (let y = minY; y <= maxY; y++) {
          if (mask[y * w + x]) span++;
        }
        if (span > maxColSpan) maxColSpan = span;
      }

      // Strikeout detection rule:
      // Real strikeouts are thin (rh <= 12, maxColSpan <= 7), wide (rw >= 26), and flat (ratio >= 2.6).
      // If rh >= 15 or maxColSpan >= 10, it is a vertical handwritten stroke, NEVER a strikeout!
      const isStrikeout = (rh <= 12 && rw >= 26 && ratio >= 2.6 && maxColSpan <= 7);

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

      // Check if output is already softmax probabilities
      let sum = 0;
      for (let i = 0; i < 10; i++) sum += output[i];

      let bestIdx = 0;
      let bestProb = -Infinity;

      if (Math.abs(sum - 1.0) < 0.05) {
        // Output is already softmax probabilities
        for (let i = 0; i < 10; i++) {
          if (output[i] > bestProb) {
            bestProb = output[i];
            bestIdx = i;
          }
        }
      } else {
        // Logits: compute stable softmax
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
        for (let i = 0; i < 10; i++) {
          const p = exps[i] / expSum;
          if (p > bestProb) {
            bestProb = p;
            bestIdx = i;
          }
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
    async recognizeCell(cellCanvas, maxMarks, row, expectedTotal) {
      const cellInfo = this.processCellInk(cellCanvas);

      if (cellInfo.isBlank) {
        return {
          detectedStr: '-',
          numericVal: 0.0,
          confidence: 1.0,
          status: 'Blank',
          isBlank: true,
          snippetCanvas: cellCanvas,
        };
      }

      if (cellInfo.isStrikeout) {
        return {
          detectedStr: '-',
          numericVal: 0.0,
          confidence: 0.95,
          status: 'Valid',
          isBlank: false,
          snippetCanvas: cellCanvas,
        };
      }

      // Segment character strokes
      const digitBoxes = this.segmentDigits(cellInfo, row);
      const tokens = [];
      let totalConf = 0;

      for (const box of digitBoxes) {
        const tensorData = this.create28x28Tensor(cellInfo, box);
        const res = await this.classifyDigit(tensorData);
        tokens.push(res.digit);
        totalConf += res.confidence;
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
      if (row && row.id === 'total' && cellInfo.rw >= 24) {
        let bestProd = -1;
        let bestCandidateVal = valStr;
        let bestCandidateConf = avgConf;
        let matchedExpected = false;

        const startS = Math.max(cellInfo.minX + 8, Math.round(cellInfo.minX + cellInfo.rw * 0.38));
        const endS = Math.min(cellInfo.maxX - 8, Math.round(cellInfo.minX + cellInfo.rw * 0.65));

        for (let s = startS; s <= endS; s++) {
          const b1 = { minX: cellInfo.minX, maxX: s - 1, minY: cellInfo.minY, maxY: cellInfo.maxY };
          const b2 = { minX: s + 1, maxX: cellInfo.maxX, minY: cellInfo.minY, maxY: cellInfo.maxY };

          const t1 = this.create28x28Tensor(cellInfo, b1);
          const t2 = this.create28x28Tensor(cellInfo, b2);

          const r1 = await this.classifyDigit(t1);
          const r2 = await this.classifyDigit(t2);

          const candVal = `${r1.digit}${r2.digit}`;
          const prod = r1.confidence * r2.confidence;

          // If this split matches expected total calculated from column questions
          if (expectedTotal && parseFloat(candVal) === Math.round(expectedTotal)) {
            valStr = candVal;
            numVal = parseFloat(valStr) || 0.0;
            avgConf = Math.max(0.95, (r1.confidence + r2.confidence) / 2);
            matchedExpected = true;
            break;
          }

          if (prod > bestProd) {
            bestProd = prod;
            bestCandidateVal = candVal;
            bestCandidateConf = (r1.confidence + r2.confidence) / 2;
          }
        }

        if (!matchedExpected && bestProd > 0.4) {
          valStr = bestCandidateVal;
          numVal = parseFloat(valStr) || 0.0;
          avgConf = bestCandidateConf;
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
      };
    }

    /**
     * Extract evaluation sheet from canvas.
     */
    async extractFromCanvas(canvas, targetIA = 'ALL') {
      await this.init();

      const rectified = await this.rectifyTable(canvas);

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

          const markRes = await this.recognizeCell(cellCanvas, row.max, row, calcSum);
          const cellB64 = (markRes.snippetCanvas || cellCanvas).toDataURL('image/png');

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
