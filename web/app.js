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

  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const cameraInput = document.getElementById('cameraInput');
  const fileInput = document.getElementById('fileInput');
  const btnCamera = document.getElementById('btnCamera');
  const btnBrowse = document.getElementById('btnBrowse');

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

  // Handle Document Selected
  function setDocument(file, isDemo = false) {
    selectedFile = file;
    useDemo = isDemo;

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
      const resp = await fetch('/api/extract-sheet', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Extraction failed');
      }

      const data = await resp.json();
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
    const s = String(status).toLowerCase();
    if (s.includes('blank')) {
      return '<span class="badge badge-blank">Blank</span>';
    }
    if (s.includes('valid')) {
      return '<span class="badge badge-valid">Valid</span>';
    }
    if (s.includes('review')) {
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

    let totalSum = 0;
    let totalDetectedSum = 0;
    let anyDetectedTotal = false;

    iaKeys.forEach((iaName) => {
      const marksList = sheet.results_by_ia[iaName];
      marksList.forEach((m) => {
        if (m.id === 'total') {
          if (m.numeric_val !== null && !isNaN(m.numeric_val)) {
            totalDetectedSum += Number(m.numeric_val);
            anyDetectedTotal = true;
          }
        } else if (m.verified_val !== null && !isNaN(m.verified_val)) {
          totalSum += Number(m.verified_val);
        }
      });
    });

    displayCalculatedTotal.textContent = isMultiIA
      ? `${totalSum.toFixed(1)} (All IA)`
      : totalSum.toFixed(1);

    if (anyDetectedTotal) {
      displayDetectedTotal.textContent = isMultiIA
        ? `${totalDetectedSum.toFixed(1)} (All IA)`
        : totalDetectedSum.toFixed(1);

      if (Math.abs(totalSum - totalDetectedSum) < 0.01) {
        matchBadge.textContent = 'Verified Match';
        matchBadge.className = 'score-badge';
        matchBadge.style.background = '';
        matchBadge.style.color = '';
        matchBadge.style.borderColor = '';
      } else {
        matchBadge.textContent = 'Discrepancy Detected';
        matchBadge.className = 'score-badge';
        matchBadge.style.background = 'var(--danger-bg)';
        matchBadge.style.color = 'var(--danger-text)';
        matchBadge.style.borderColor = 'var(--danger-border)';
      }
    } else {
      displayDetectedTotal.textContent = '-';
      matchBadge.textContent = 'Evaluation Ready';
      matchBadge.className = 'score-badge';
      matchBadge.style.background = '';
      matchBadge.style.color = '';
      matchBadge.style.borderColor = '';
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
    }

    // 2. Table Rows
    evalTableBody.innerHTML = '';

    Object.keys(sheet.results_by_ia).forEach((iaName) => {
      const marks = sheet.results_by_ia[iaName];

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
          tdCrop.innerHTML = `<img src="${m.crop_base64}" alt="${m.label}" class="cell-crop">`;
        } else {
          tdCrop.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">No crop</span>';
        }
        tr.appendChild(tdCrop);

        // Detected Value & Confidence
        const tdDet = document.createElement('td');
        const detStr = m.detected_str ? m.detected_str : '-';
        const confStr = m.confidence !== null && m.confidence !== undefined && !m.is_blank
          ? `<span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 4px;">(${Math.round(m.confidence * 100)}%)</span>`
          : '';
        tdDet.innerHTML = `<strong>${detStr}</strong>${confStr}`;
        tr.appendChild(tdDet);

        // Status Badge
        const tdStatus = document.createElement('td');
        tdStatus.id = `status-${currentSheetIndex}-${iaName}-${m.id}`;
        tdStatus.innerHTML = getBadgeHTML(m.status);
        tr.appendChild(tdStatus);

        // Verified Mark (Editable Input)
        const tdInput = document.createElement('td');
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'mark-input';
        input.value = m.verified_val !== null ? m.verified_val : '';

        input.addEventListener('input', (e) => {
          const valStr = e.target.value.trim();
          const parsed = parseFloat(valStr);
          m.verified_val = !isNaN(parsed) ? parsed : 0.0;
          m.status = 'Corrected manually';

          const stCell = document.getElementById(`status-${currentSheetIndex}-${iaName}-${m.id}`);
          if (stCell) {
            stCell.innerHTML = getBadgeHTML('Corrected manually');
          }
          recalculateTotals();
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
          target_ia: s.target_ia,
          selected_ias: s.selected_ias,
          results_by_ia: cleanResults,
          summary_totals: s.summary_totals,
        };
      });

      const exportPayload = {
        total_sheets: currentExtractionData.total_sheets || cleanSheets.length,
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
        const hasBom = text.charCodeAt(0) === 0xfeff;
        blob = new Blob([hasBom ? text : '\ufeff' + text], { type: 'text/csv;charset=utf-8;' });
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
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
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
