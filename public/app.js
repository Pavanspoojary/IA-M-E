/**
 * IA Bluebook Marks Extraction System - Minimal Frontend Controller
 * Strict Zero Emojis Policy Enforced.
 */

(function () {
  'use strict';

  // State Management
  let selectedFile = null;
  let useDemo = false;
  let selectedIA = 'IA 1';
  let currentExtractionData = null;
  let currentSheetIndex = 0;
  let pdfMode = false;

  // Initialize In-Browser Extraction Engine
  const browserEngine = typeof BrowserExtractionEngine !== 'undefined' ? new BrowserExtractionEngine() : null;
  if (browserEngine) browserEngine.init();

  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const cameraInput = document.getElementById('cameraInput');
  const fileInput = document.getElementById('fileInput');
  const btnCamera = document.getElementById('btnCamera');
  const btnBrowse = document.getElementById('btnBrowse');
  const pdfModeSettings = document.getElementById('pdfModeSettings');
  const startingUsnInput = document.getElementById('startingUsn');
  const skipUsnsInput = document.getElementById('skipUsns');

  const fileBanner = document.getElementById('fileBanner');
  const fileBannerText = document.getElementById('fileBannerText');
  const btnChangeFile = document.getElementById('btnChangeFile');

  const iaSelectCard = document.getElementById('iaSelectCard');
  const iaButtons = document.querySelectorAll('.ia-btn');
  const btnFetchMarks = document.getElementById('btnFetchMarks');
  const progressBox = document.getElementById('progressBox');

  const resultsCard = document.getElementById('resultsCard');
  const sheetNavBar = document.getElementById('sheetNavBar');
  const btnPrevSheet = document.getElementById('btnPrevSheet');
  const btnNextSheet = document.getElementById('btnNextSheet');
  const sheetSelectDropdown = document.getElementById('sheetSelectDropdown');
  const sheetCountIndicator = document.getElementById('sheetCountIndicator');

  const cropColumnTitle = document.getElementById('cropColumnTitle');
  const columnCropImg = document.getElementById('columnCropImg');
  const evalTableBody = document.getElementById('evalTableBody');
  const displayCalculatedTotal = document.getElementById('displayCalculatedTotal');
  const displayDetectedTotal = document.getElementById('displayDetectedTotal');
  const matchBadge = document.getElementById('matchBadge');

  const btnDownloadXlsx = document.getElementById('btnDownloadXlsx');
  const btnDownloadCsv = document.getElementById('btnDownloadCsv');
  const btnNewExtraction = document.getElementById('btnNewExtraction');

  const snippetModal = document.getElementById('snippetModal');
  const modalTitle = document.getElementById('modalTitle');
  const modalImg = document.getElementById('modalImg');
  const modalDetails = document.getElementById('modalDetails');
  const btnModalClose = document.getElementById('btnModalClose');

  if (btnModalClose) {
    btnModalClose.addEventListener('click', () => {
      if (snippetModal) snippetModal.style.display = 'none';
    });
  }
  if (snippetModal) {
    snippetModal.addEventListener('click', (e) => {
      if (e.target === snippetModal) snippetModal.style.display = 'none';
    });
  }

  // Handle Document Selected
  function setDocument(file, isDemo = false) {
    selectedFile = file;
    useDemo = isDemo;
    pdfMode = Boolean(file && (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')));
    pdfModeSettings.style.display = pdfMode ? 'block' : 'none';

    if (isDemo) {
      fileBannerText.textContent = 'Loaded Demo Filled Evaluation Sheet (marks_eval_sheet_filled.jpg)';
    } else if (file) {
      const sizeKb = (file.size / 1024).toFixed(1);
      fileBannerText.textContent = `Loaded Document: ${file.name} (${sizeKb} KB)`;
    }

    fileBanner.style.display = 'flex';
    iaSelectCard.style.display = 'block';
    iaSelectCard.scrollIntoView({ behavior: 'smooth' });
  }

  // Input Buttons
  btnCamera.addEventListener('click', (e) => {
    e.stopPropagation();
    cameraInput.click();
  });

  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  cameraInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      setDocument(e.target.files[0], false);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      setDocument(e.target.files[0], false);
    }
  });

  btnChangeFile.addEventListener('click', () => {
    selectedFile = null;
    useDemo = false;
    pdfMode = false;
    pdfModeSettings.style.display = 'none';
    fileBanner.style.display = 'none';
    iaSelectCard.style.display = 'none';
    resultsCard.style.display = 'none';
    cameraInput.value = '';
    fileInput.value = '';
  });

  // Drag & Drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setDocument(e.dataTransfer.files[0], false);
    }
  });

  // Segmented IA Buttons
  iaButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      iaButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      selectedIA = btn.dataset.ia;
    });
  });

  // Fetch Marks and Crop Column
  btnFetchMarks.addEventListener('click', async () => {
    const isPdf = selectedFile && (selectedFile.type === 'application/pdf' || selectedFile.name.toLowerCase().endsWith('.pdf'));
    if (pdfMode && !isPdf) {
      alert('PDF Mode requires a PDF document. Please upload a PDF or turn PDF Mode off.');
      return;
    }
    let startingUsn = '';
    let skippedUsns = new Set();
    if (pdfMode) {
      const usnPrefix = '4MW';
      const startingSuffix = startingUsnInput.value.trim().toUpperCase().replace(/^4MW/, '');
      if (!/^.*\d+$/.test(startingSuffix)) {
        alert('Enter the part after 4MW, ending in digits, for example 24AD001.');
        startingUsnInput.focus();
        return;
      }
      startingUsn = `${usnPrefix}${startingSuffix}`;
      skippedUsns = new Set(skipUsnsInput.value.split(',')
        .map((suffix) => suffix.trim().toUpperCase().replace(/^4MW/, ''))
        .filter(Boolean)
        .map((suffix) => `${usnPrefix}${suffix}`));
    }
    const formData = new FormData();
    formData.append('target_ia', selectedIA);

    if (useDemo) {
      formData.append('use_demo', 'true');
    } else if (selectedFile) {
      formData.append('file', selectedFile);
      formData.append('use_demo', 'false');
    } else {
      alert('Please upload a file or choose the demo sheet first.');
      return;
    }

    btnFetchMarks.disabled = true;
    progressBox.style.display = 'block';

    try {
      let data = null;

      // In-Browser Neural Extraction (No server required)
      if (selectedFile && browserEngine) {
        progressBox.textContent = 'Processing document in local browser via ONNX WASM...';
        try {
          data = await browserEngine.processDocument(selectedFile, selectedIA);
        } catch (clientErr) {
          console.warn('Browser extraction failed, attempting server fallback:', clientErr);
        }
      }

      // Fallback to server endpoint if running with local Python server
      if (!data) {
        progressBox.textContent = 'Cropping column and recognizing handwritten marks...';
        const resp = await fetch('/api/extract-sheet', {
          method: 'POST',
          body: formData,
        });

        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || 'Extraction failed');
        }

        data = await resp.json();
      }

      if (pdfMode) {
        const match = startingUsn.match(/^(.*?)(\d+)$/);
        const prefix = match[1];
        const width = match[2].length;
        let sequence = Number(match[2]);
        if (!data.sheets || !data.sheets.length) {
          data.sheets = [{
            sheet_index: 1,
            page_number: 1,
            target_ia: data.target_ia || selectedIA,
            selected_ias: data.selected_ias || [data.target_ia || selectedIA],
            results_by_ia: data.results_by_ia || {},
            summary_totals: data.summary_totals || {},
            column_crop_base64: data.column_crop_base64,
          }];
        }
        data.sheets.sort((a, b) => (a.sheet_index || 0) - (b.sheet_index || 0));
        data.sheets.forEach((sheet, index) => {
          let usn = `${prefix}${String(sequence).padStart(width, '0')}`;
          while (skippedUsns.has(usn.toUpperCase())) {
            sequence += 1;
            usn = `${prefix}${String(sequence).padStart(width, '0')}`;
          }
          sheet.usn = usn;
          sheet.sheet_label = `${usn} (Page ${sheet.page_number || index + 1})`;
          sequence += 1;
        });
        data.pdf_mode = true;
      }

      currentExtractionData = data;
      renderResults(data);
    } catch (err) {
      alert(`Error during extraction: ${err.message}`);
    } finally {
      btnFetchMarks.disabled = false;
      progressBox.style.display = 'none';
    }
  });

  // Render Status Badge
  function getBadgeHTML(status) {
    const s = String(status || '').toLowerCase();
    if (s === 'na' || s.includes('blank')) {
      return '<span class="badge badge-blank">NA</span>';
    }
    if (s.includes('valid')) {
      return '<span class="badge badge-valid">Valid</span>';
    }
    if (s.includes('calc')) {
      return '<span class="badge badge-valid" style="background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;">Calculated</span>';
    }
    if (s.includes('review') || s.includes('discrepancy')) {
      return '<span class="badge badge-review">Needs Review</span>';
    }
    if (s.includes('correct')) {
      return '<span class="badge badge-corrected">Corrected</span>';
    }
    return `<span class="badge badge-review">${status}</span>`;
  }

  // Recalculate Totals from Live Table Inputs for Active Sheet
  function recalculateTotals() {
    if (!currentExtractionData || !currentExtractionData.sheets || !currentExtractionData.sheets[currentSheetIndex]) return;

    const sheet = currentExtractionData.sheets[currentSheetIndex];
    const iaKeys = Object.keys(sheet.results_by_ia);
    const isMultiIA = iaKeys.length > 1;
    const scoreBar = document.querySelector('.score-summary-bar');

    if (!isMultiIA) {
      const iaName = iaKeys[0] || 'IA 1';
      const marksList = sheet.results_by_ia[iaName] || [];
      let calcSum = 0;
      let hasMarks = false;
      let detTotal = null;

      marksList.forEach((m) => {
        if (m.id === 'total') {
          if (m.verified_val !== 'NA' && m.verified_val !== null && !isNaN(m.verified_val)) {
            detTotal = Number(m.verified_val);
          } else if (m.numeric_val !== null && !isNaN(m.numeric_val)) {
            detTotal = Number(m.numeric_val);
          }
        } else if (m.verified_val !== 'NA' && m.verified_val !== null && !isNaN(m.verified_val)) {
          calcSum += Number(m.verified_val);
          hasMarks = true;
        }
      });

      if (scoreBar) {
        let badgeText = 'Verified Match';
        let badgeStyle = '';

        if (!hasMarks && detTotal === null) {
          badgeText = 'NA (Blank)';
          badgeStyle = 'background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1;';
        } else if (detTotal !== null && Math.abs(calcSum - detTotal) < 0.01) {
          badgeText = 'Verified Match';
          badgeStyle = 'background:var(--success-bg); color:var(--success-text); border:1px solid var(--success-border);';
        } else if (detTotal !== null) {
          badgeText = 'Discrepancy Detected';
          badgeStyle = 'background:var(--danger-bg); color:var(--danger-text); border:1px solid var(--danger-border);';
        } else {
          badgeText = 'Evaluation Ready';
        }

        scoreBar.innerHTML = `
          <div class="score-item">
            <span class="score-lbl">Calculated Score</span>
            <span class="score-val" id="displayCalculatedTotal">${hasMarks ? calcSum.toFixed(1) : 'NA'}</span>
          </div>
          <div class="score-item">
            <span class="score-lbl">Sheet Detected Total</span>
            <span class="score-val" id="displayDetectedTotal">${detTotal !== null ? detTotal.toFixed(1) : 'NA'}</span>
          </div>
          <div>
            <span id="matchBadge" class="score-badge" style="${badgeStyle}">${badgeText}</span>
          </div>
        `;
      }
      return;
    }

    // Multi-IA: Render individual breakdown cards per IA!
    if (scoreBar) {
      let cardsHtml = '';
      iaKeys.forEach((iaName) => {
        const marksList = sheet.results_by_ia[iaName] || [];
        let calcSum = 0;
        let hasMarks = false;
        let detTotal = null;

        marksList.forEach((m) => {
          if (m.id === 'total') {
            if (m.verified_val !== 'NA' && m.verified_val !== null && !isNaN(m.verified_val)) {
              detTotal = Number(m.verified_val);
            } else if (m.numeric_val !== null && !isNaN(m.numeric_val)) {
              detTotal = Number(m.numeric_val);
            }
          } else if (m.verified_val !== 'NA' && m.verified_val !== null && !isNaN(m.verified_val)) {
            calcSum += Number(m.verified_val);
            hasMarks = true;
          }
        });

        let badgeText = 'Verified Match';
        let badgeStyle = 'background:var(--success-bg); color:var(--success-text); border:1px solid var(--success-border);';

        if (!hasMarks && detTotal === null) {
          badgeText = 'NA (Blank)';
          badgeStyle = 'background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1;';
        } else if (detTotal !== null && Math.abs(calcSum - detTotal) < 0.01) {
          badgeText = 'Verified Match';
          badgeStyle = 'background:var(--success-bg); color:var(--success-text); border:1px solid var(--success-border);';
        } else if (detTotal !== null) {
          badgeText = 'Discrepancy';
          badgeStyle = 'background:var(--danger-bg); color:var(--danger-text); border:1px solid var(--danger-border);';
        } else if (hasMarks) {
          badgeText = 'Calculated';
          badgeStyle = 'background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;';
        }

        const calcDisp = hasMarks ? calcSum.toFixed(1) : 'NA';
        const detDisp = detTotal !== null ? detTotal.toFixed(1) : 'NA';

        cardsHtml += `
          <div style="flex: 1; min-width: 160px; background: #ffffff; border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--primary); margin-bottom: 6px;">${iaName}</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; font-size: 0.85rem; margin-bottom: 6px;">
              <span style="color: var(--text-muted);">Calc: <strong style="color: var(--text-main); font-size: 1.1rem;">${calcDisp}</strong></span>
              <span style="color: var(--text-muted);">Sheet: <strong style="color: var(--text-main); font-size: 1.1rem;">${detDisp}</strong></span>
            </div>
            <div style="text-align: right;">
              <span class="score-badge" style="font-size: 0.72rem; padding: 2px 7px; ${badgeStyle}">${badgeText}</span>
            </div>
          </div>
        `;
      });
      scoreBar.innerHTML = cardsHtml;
    }
  }

  // Render Single Active Sheet in UI
  function renderActiveSheet() {
    if (!currentExtractionData || !currentExtractionData.sheets || !currentExtractionData.sheets[currentSheetIndex]) return;

    const sheet = currentExtractionData.sheets[currentSheetIndex];
    const totalSheets = currentExtractionData.total_sheets || currentExtractionData.sheets.length;

    // Update Dropdown, Counter, and Button States
    sheetSelectDropdown.value = String(currentSheetIndex);
    sheetCountIndicator.textContent = `Sheet ${currentSheetIndex + 1} of ${totalSheets}`;
    btnPrevSheet.disabled = (currentSheetIndex === 0);
    btnNextSheet.disabled = (currentSheetIndex >= totalSheets - 1);

    // 1. Cropped Column Preview
    if (sheet.column_crop_base64) {
      columnCropImg.src = sheet.column_crop_base64;
      const targetLabel = sheet.target_ia === 'ALL' ? 'Evaluation Grid' : sheet.target_ia;
      cropColumnTitle.textContent = totalSheets > 1
        ? `Cropped Column (${targetLabel}) - Sheet ${currentSheetIndex + 1}`
        : `Cropped Column (${targetLabel})`;
      if (sheet.usn) cropColumnTitle.textContent = `Cropped Column (${targetLabel}) - ${sheet.usn}`;
    }

    // 2. Table Rows
    evalTableBody.innerHTML = '';

    Object.keys(sheet.results_by_ia).forEach((iaName) => {
      const marks = sheet.results_by_ia[iaName];

      if (sheet.selected_ias.length > 1) {
        const trHeader = document.createElement('tr');
        trHeader.className = 'ia-section-header';
        const isIaBlank = marks.every(m => m.is_blank || m.status === 'NA' || m.status === 'Blank' || m.verified_val === 'NA');
        const statusNote = isIaBlank ? ' - Not Conducted / Blank' : '';
        const iaTitleMap = {
          'IA 1': 'First I.A. Test (IA 1)',
          'IA 2': 'Second I.A. Test (IA 2)',
          'IA 3': 'Third I.A. Test (IA 3)',
        };
        const titleText = (iaTitleMap[iaName] || iaName) + statusNote;
        trHeader.innerHTML = `<th colspan="5" style="background:#e2e8f0; color:#1e293b; padding:8px 12px; font-weight:700; text-align:left; font-size:0.9rem; border-top:2px solid #cbd5e1;">${titleText}</th>`;
        evalTableBody.appendChild(trHeader);
      }

      marks.forEach((m) => {
        const tr = document.createElement('tr');
        if (m.id === 'total') {
          tr.className = 'total-row';
        }

        // Question label cell
        const tdQ = document.createElement('td');
        const iaPrefix = sheet.selected_ias.length > 1 ? `[${iaName}] ` : '';
        tdQ.innerHTML = `<strong>${iaPrefix}${m.label}</strong>`;
        tr.appendChild(tdQ);

        // Snippet cell
        const tdCrop = document.createElement('td');
        if (m.crop_base64) {
          const imgCrop = document.createElement('img');
          imgCrop.src = m.crop_base64;
          imgCrop.alt = m.label;
          imgCrop.className = 'cell-crop';
          imgCrop.title = 'Click to inspect handwriting';
          imgCrop.style.cursor = 'zoom-in';
          imgCrop.addEventListener('click', () => {
            if (modalTitle) modalTitle.textContent = `${iaPrefix}${m.label} Handwriting Inspection`;
            if (modalImg) modalImg.src = m.crop_base64;
            const confPercent = m.confidence ? Math.round(m.confidence * 100) : 0;
            if (modalDetails) {
              modalDetails.textContent = `Detected Value: ${m.detected_str || '-'} | Confidence: ${confPercent}% | Status: ${m.status}`;
            }
            if (snippetModal) snippetModal.style.display = 'flex';
          });
          tdCrop.appendChild(imgCrop);
        } else {
          tdCrop.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">No crop</span>';
        }
        tr.appendChild(tdCrop);

        // Detected Value & Confidence
        const tdDet = document.createElement('td');
        const isNaCell = m.is_blank || m.status === 'NA' || m.status === 'Blank' || m.detected_str === 'NA' || m.detected_str === '-';
        const detStr = isNaCell ? 'NA' : (m.detected_str ? m.detected_str : 'NA');
        const confStr = m.confidence !== null && m.confidence !== undefined && !isNaCell
          ? `<span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 4px;">(${Math.round(m.confidence * 100)}%)</span>`
          : '';
        tdDet.innerHTML = `<strong>${detStr}</strong>${confStr}`;
        tr.appendChild(tdDet);

        // Status Badge
        const tdStatus = document.createElement('td');
        tdStatus.id = `status-${currentSheetIndex}-${iaName}-${m.id}`;
        tdStatus.innerHTML = getBadgeHTML(isNaCell ? 'NA' : m.status);
        tr.appendChild(tdStatus);

        // Verified Mark (Editable Input)
        const tdInput = document.createElement('td');
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'mark-input';
        const displayVal = (m.verified_val !== null && m.verified_val !== undefined && m.verified_val !== 'NA')
          ? m.verified_val
          : 'NA';
        input.value = displayVal;

        input.addEventListener('focus', () => {
          input.select();
        });

        input.addEventListener('input', (e) => {
          const valStr = e.target.value.trim().toUpperCase();
          if (valStr === 'NA' || valStr === '-' || valStr === '') {
            m.verified_val = 'NA';
            m.status = 'NA';
            m.is_blank = true;
          } else {
            const parsed = parseFloat(valStr);
            m.verified_val = !isNaN(parsed) ? parsed : 0.0;
            m.status = 'Corrected manually';
            m.is_blank = false;
          }

          const stCell = document.getElementById(`status-${currentSheetIndex}-${iaName}-${m.id}`);
          if (stCell) {
            stCell.innerHTML = getBadgeHTML(m.status);
          }
          recalculateTotals();
        });

        input.addEventListener('blur', (e) => {
          if (e.target.value.trim() === '') {
            e.target.value = 'NA';
            m.verified_val = 'NA';
            m.status = 'NA';
            m.is_blank = true;
            const stCell = document.getElementById(`status-${currentSheetIndex}-${iaName}-${m.id}`);
            if (stCell) {
              stCell.innerHTML = getBadgeHTML('NA');
            }
            recalculateTotals();
          }
        });

        // Fast keyboard navigation: Enter / ArrowDown moves down, ArrowUp moves up
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === 'ArrowDown') {
            e.preventDefault();
            const allInputs = Array.from(document.querySelectorAll('.mark-input'));
            const currIdx = allInputs.indexOf(input);
            if (currIdx >= 0 && currIdx < allInputs.length - 1) {
              allInputs[currIdx + 1].focus();
              allInputs[currIdx + 1].select();
            }
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            const allInputs = Array.from(document.querySelectorAll('.mark-input'));
            const currIdx = allInputs.indexOf(input);
            if (currIdx > 0) {
              allInputs[currIdx - 1].focus();
              allInputs[currIdx - 1].select();
            }
          }
        });

        tdInput.appendChild(input);
        tr.appendChild(tdInput);

        evalTableBody.appendChild(tr);
      });
    });

    recalculateTotals();
  }

  // Render Extraction Results (Multi-sheet or Single)
  function renderResults(data) {
    if (!data) return;

    // Normalize data.sheets
    if (!data.sheets || !data.sheets.length) {
      data.sheets = [{
        sheet_index: 1,
        page_number: 1,
        sheet_label: 'Sheet 1 (Page 1)',
        column_crop_base64: data.column_crop_base64,
        target_ia: data.target_ia,
        selected_ias: data.selected_ias || [data.target_ia],
        results_by_ia: data.results_by_ia,
        summary_totals: data.summary_totals,
      }];
      data.total_sheets = 1;
    }

    currentExtractionData = data;
    currentSheetIndex = 0;

    const totalSheets = data.total_sheets || data.sheets.length;

    // Populate Sheet Selector Dropdown
    sheetSelectDropdown.innerHTML = '';
    data.sheets.forEach((sheet, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = sheet.sheet_label || `Sheet ${idx + 1}`;
      sheetSelectDropdown.appendChild(opt);
    });

    if (totalSheets > 1) {
      sheetNavBar.style.display = 'flex';
      if (selectedFile) {
        const sizeKb = (selectedFile.size / 1024).toFixed(1);
        fileBannerText.textContent = `Loaded Document: ${selectedFile.name} (${sizeKb} KB, ${totalSheets} sheets detected)`;
      }
    } else {
      sheetNavBar.style.display = 'none';
    }

    renderActiveSheet();

    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
  }

  // Sheet Navigation Events
  btnPrevSheet.addEventListener('click', () => {
    if (currentSheetIndex > 0) {
      currentSheetIndex -= 1;
      renderActiveSheet();
    }
  });

  btnNextSheet.addEventListener('click', () => {
    const totalSheets = currentExtractionData ? (currentExtractionData.total_sheets || currentExtractionData.sheets.length) : 1;
    if (currentSheetIndex < totalSheets - 1) {
      currentSheetIndex += 1;
      renderActiveSheet();
    }
  });

  sheetSelectDropdown.addEventListener('change', (e) => {
    const newIdx = parseInt(e.target.value, 10);
    if (!isNaN(newIdx) && newIdx >= 0) {
      currentSheetIndex = newIdx;
      renderActiveSheet();
    }
  });

  // Download Trigger Handler
  async function downloadExport(format) {
    if (!currentExtractionData) {
      alert('No extracted data to download.');
      return;
    }

    let safeIa = 'IA_1';
    if (selectedIA.includes('All')) {
      safeIa = 'All_IA';
    } else if (selectedIA.includes('IA 2')) {
      safeIa = 'IA_2';
    } else if (selectedIA.includes('IA 3')) {
      safeIa = 'IA_3';
    }

    const ext = format === 'csv' ? 'csv' : 'xlsx';
    const filename = `IA_Marks_Evaluation_${safeIa}.${ext}`;

    // 1. In-browser direct export (Zero server calls)
    if (browserEngine) {
      try {
        if (format === 'csv') {
          browserEngine.downloadCsv(currentExtractionData, filename);
          return;
        } else if (typeof XLSX !== 'undefined') {
          browserEngine.downloadXlsx(currentExtractionData, filename);
          return;
        }
      } catch (clientExportErr) {
        console.warn('Browser export failed, attempting server fallback:', clientExportErr);
      }
    }

    // 2. Server fallback export
    try {
      // Prepare lightweight data payload omitting heavy base64 image crops (drops payload from 18MB to 4KB)
      const cleanSheets = (currentExtractionData.sheets || []).map((s) => {
        const cleanResults = {};
        if (s.results_by_ia) {
          Object.keys(s.results_by_ia).forEach((ia) => {
            cleanResults[ia] = (s.results_by_ia[ia] || []).map((m) => ({
              id: m.id,
              label: m.label,
              detected_str: m.detected_str,
              numeric_val: m.numeric_val,
              verified_val: m.verified_val,
              status: m.status,
              is_blank: m.is_blank,
            }));
          });
        }
        return {
          sheet_index: s.sheet_index,
          page_number: s.page_number,
          sheet_label: s.sheet_label,
          usn: s.usn,
          target_ia: s.target_ia,
          selected_ias: s.selected_ias,
          results_by_ia: cleanResults,
          summary_totals: s.summary_totals,
        };
      });

      const exportPayload = {
        total_sheets: currentExtractionData.total_sheets || cleanSheets.length,
        pdf_mode: Boolean(currentExtractionData.pdf_mode),
        sheets: cleanSheets,
        results_by_ia: cleanSheets.length > 0 ? cleanSheets[0].results_by_ia : {},
      };

      const resp = await fetch('/api/export-sheet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format: format,
          data: exportPayload,
        }),
      });

      if (!resp.ok) {
        throw new Error('Failed to generate export file');
      }

      let blob;
      if (format === 'csv') {
        const text = await resp.text();
        const cleanText = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
        blob = new Blob([cleanText], { type: 'text/csv;charset=utf-8;' });
      } else {
        blob = await resp.blob();
      }

      let safeIa = 'IA_1';
      if (selectedIA.includes('All')) {
        safeIa = 'All_IA';
      } else if (selectedIA.includes('IA 2')) {
        safeIa = 'IA_2';
      } else if (selectedIA.includes('IA 3')) {
        safeIa = 'IA_3';
      }

      const ext = format === 'csv' ? 'csv' : 'xlsx';
      const filename = `IA_Marks_Evaluation_${safeIa}.${ext}`;

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
    } catch (err) {
      alert(`Export error: ${err.message}`);
    }
  }

  btnDownloadXlsx.addEventListener('click', () => downloadExport('xlsx'));
  btnDownloadCsv.addEventListener('click', () => downloadExport('csv'));

  btnNewExtraction.addEventListener('click', () => {
    resultsCard.style.display = 'none';
    iaSelectCard.style.display = 'none';
    fileBanner.style.display = 'none';
    sheetNavBar.style.display = 'none';
    selectedFile = null;
    useDemo = false;
    currentExtractionData = null;
    currentSheetIndex = 0;
    cameraInput.value = '';
    fileInput.value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();
