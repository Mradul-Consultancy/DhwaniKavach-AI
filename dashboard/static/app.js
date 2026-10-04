/**
 * Acoustic Machine Fault Detection and Monitoring System — Dashboard v3 JS
 * Single-page flow
 */

(function () {
    "use strict";

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // Header
    const statusChip = $("#modelStatus");
    const statusText = $("#statusText");
    const deviceText = $("#deviceText");
    const paramText = $("#paramText");

    // Mode toggle
    const modeBtns = $$(".mode-btn");
    const modeSingle = $("#modeSingle");
    const modeBatch = $("#modeBatch");
    const modeMic = $("#modeMic");

    // Upload
    const dropzone = $("#dropzone");
    const fileInput = $("#fileInput");
    const fileInfo = $("#fileInfo");
    const fileName = $("#fileName");
    const fileSize = $("#fileSize");
    const clearFile = $("#clearFile");
    const analyzeBtn = $("#analyzeBtn");
    const btnText = $("#btnText");
    const btnLoader = $("#btnLoader");

    // Batch Upload
    const dropzoneBatch = $("#dropzoneBatch");
    const fileInputBatch = $("#fileInputBatch");
    const batchInfo = $("#batchInfo");
    const batchCount = $("#batchCount");
    const clearBatch = $("#clearBatch");
    const batchBtn = $("#batchBtn");
    const batchBtnText = $("#batchBtnText");
    const batchBtnLoader = $("#batchBtnLoader");

    // Mic
    const micRing = $("#micRing");
    const micStatus = $("#micStatus");
    const micTimer = $("#micTimer");
    const micStartBtn = $("#micStartBtn");
    const micStopBtn = $("#micStopBtn");

    // Sections
    const loadingSection = $("#loadingSection");
    const resultsSection = $("#resultsSection");
    const batchResultsSection = $("#batchResultsSection");
    const errorSection = $("#errorSection");

    // Single Results
    const verdictBanner = $("#verdictBanner");
    const verdictIcon = $("#verdictIcon");
    const verdictTitle = $("#verdictTitle");
    const verdictDesc = $("#verdictDesc");
    const verdictFile = $("#verdictFile");
    const gaugeFill = $("#gaugeFill");
    const gaugeNumber = $("#gaugeNumber");
    const lossValue = $("#lossValue");
    const lossBar = $("#lossBar");
    const lossThreshold = $("#lossThreshold");
    const thresholdLabel = $("#thresholdLabel");
    const infoFile = $("#infoFile");
    const infoThreshold = $("#infoThreshold");
    const infoDevice = $("#infoDevice");

    const threshSlider = $("#threshSlider");
    const threshCurrent = $("#threshCurrent");
    const trThresh = $("#trThresh");
    const trVerdict = $("#trVerdict");

    // Images
    const imgWaveform = $("#waveformImg");
    const imgOriginal = $("#originalSpecImg");
    const imgRecon = $("#reconSpecImg");
    const imgError = $("#errorMapImg");

    // Batch Results
    const bsTotal = $("#bsTotal");
    const bsNormal = $("#bsNormal");
    const bsAnomaly = $("#bsAnomaly");
    const bsErrors = $("#bsErrors");
    const batchTableBody = $("#batchTableBody");

    // Evaluation
    const evalBtn = $("#evalBtn");
    const evalBtnText = $("#evalBtnText");
    const evalBtnLoader = $("#evalBtnLoader");
    const evalLoading = $("#evalLoading");
    const evalResults = $("#evalResults");
    const evalError = $("#evalError");

    // Architecture
    const imgArch = $("#imgArch");
    const layerTableBody = $("#layerTableBody");
    const configGrid = $("#configGrid");

    let selectedFile = null;
    let batchFiles = [];
    let currentLoss = 0;
    let currentThreshold = 0.035;

    let mediaRecorder = null;
    let audioChunks = [];
    let micInterval = null;
    let micStartTime = 0;
    let audioContext = null;

    init();

    function init() {
        checkHealth();
        // loadArchitecture();

        // Mode Toggles
        modeBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                modeBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                
                modeSingle.classList.add("hidden");
                modeBatch.classList.add("hidden");
                modeMic.classList.add("hidden");

                const mode = btn.dataset.mode;
                if (mode === "single") modeSingle.classList.remove("hidden");
                if (mode === "batch") modeBatch.classList.remove("hidden");
                if (mode === "mic") modeMic.classList.remove("hidden");
                
                hideAnalysisResults();
            });
        });

        // Single Upload Events
        dropzone.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (e) => { if(e.target.files[0]) handleFile(e.target.files[0]); });
        dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
        dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
        dropzone.addEventListener("drop", (e) => { e.preventDefault(); dropzone.classList.remove("drag-over"); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
        clearFile.addEventListener("click", (e) => { e.stopPropagation(); clearSingleFile(); });
        analyzeBtn.addEventListener("click", runSingleAnalysis);
        $("#newAnalysisBtn").addEventListener("click", resetSingleAnalysis);
        $("#errorRetryBtn").addEventListener("click", hideAnalysisResults);

        // Batch Upload Events
        dropzoneBatch.addEventListener("click", () => fileInputBatch.click());
        fileInputBatch.addEventListener("change", (e) => { if(e.target.files.length) handleBatch(e.target.files); });
        dropzoneBatch.addEventListener("dragover", (e) => { e.preventDefault(); dropzoneBatch.classList.add("drag-over"); });
        dropzoneBatch.addEventListener("dragleave", () => dropzoneBatch.classList.remove("drag-over"));
        dropzoneBatch.addEventListener("drop", (e) => { e.preventDefault(); dropzoneBatch.classList.remove("drag-over"); if(e.dataTransfer.files.length) handleBatch(e.dataTransfer.files); });
        clearBatch.addEventListener("click", (e) => { e.stopPropagation(); clearBatchFiles(); });
        batchBtn.addEventListener("click", runBatchAnalysis);
        $("#newBatchBtn").addEventListener("click", resetBatchAnalysis);

        // Mic Events
        micStartBtn.addEventListener("click", startRecording);
        micStopBtn.addEventListener("click", stopRecording);

        // Eval Events
        // evalBtn.addEventListener("click", runEvaluation);

        // Threshold Slider Event
        threshSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            threshCurrent.textContent = val.toFixed(3);
            updateThresholdVerdict(val);
        });

        // Edge ONNX Export Button
        const btnExportOnnx = $("#btnExportOnnx");
        if (btnExportOnnx) {
            btnExportOnnx.addEventListener("click", async () => {
                const onnxText = $("#onnxStatusText");
                onnxText.textContent = "Exporting…";
                try {
                    const res = await fetch("/api/export-onnx", { method: "POST" });
                    const d = await res.json();
                    if (d.success) {
                        onnxText.textContent = `ONNX (${d.size_kb} KB)`;
                        alert(`ONNX Model Exported!\nPath: ${d.onnx_path}\nReady for Edge Gateways (Raspberry Pi, NVIDIA Jetson).`);
                    } else {
                        throw new Error(d.error);
                    }
                } catch (err) {
                    onnxText.textContent = "Export Failed";
                    alert("ONNX Export: " + err.message);
                }
            });
        }

        // Calibrate Baseline Button
        const btnCalibrate = $("#btnCalibrateBaseline");
        if (btnCalibrate) {
            btnCalibrate.addEventListener("click", async () => {
                if (!currentLoss) return;
                const confirmCalib = confirm(`Calibrate dynamic threshold using current loss (${currentLoss.toFixed(5)}) as normal baseline reference?`);
                if (!confirmCalib) return;

                try {
                    const res = await fetch("/api/calibrate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            losses: [currentLoss * 0.9, currentLoss, currentLoss * 1.1],
                            method: "sigma",
                            k: 3.0
                        })
                    });
                    const d = await res.json();
                    if (d.success) {
                        currentThreshold = d.new_threshold;
                        threshSlider.value = currentThreshold;
                        threshCurrent.textContent = currentThreshold.toFixed(3);
                        updateThresholdVerdict(currentThreshold);
                        alert(`Baseline Calibrated!\nNew Dynamic Threshold: ${d.new_threshold.toFixed(5)}`);
                    }
                } catch (err) {
                    alert("Calibration error: " + err.message);
                }
            });
        }
    }

    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            const d = await res.json();
            if (d.model_loaded) {
                statusChip.classList.add("online");
                statusChip.classList.remove("offline");
                statusText.textContent = "Model Ready";
            } else {
                statusChip.classList.add("offline");
                statusText.textContent = "No Model";
            }
            deviceText.textContent = d.device === "cuda" ? "GPU" : "CPU";
            paramText.textContent = d.model_params ? d.model_params.toLocaleString() + " params" : "—";
            currentThreshold = d.threshold || 0.035;
            threshSlider.value = currentThreshold;
            threshCurrent.textContent = currentThreshold.toFixed(3);
        } catch (e) {
            statusChip.classList.add("offline");
            statusText.textContent = "Offline";
        }
    }

    async function loadArchitecture() {
        try {
            const res = await fetch("/api/model-info");
            if (!res.ok) return;
            const data = await res.json();

            setImg(imgArch, data.architecture_image);

            layerTableBody.innerHTML = "";
            data.layers.forEach((l) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${l.name}</td><td>${l.type}</td><td>${l.details}</td>`;
                layerTableBody.appendChild(tr);
            });

            configGrid.innerHTML = "";
            Object.entries(data.config).forEach(([k, v]) => {
                const d = document.createElement("div"); d.className = "cg-item";
                d.innerHTML = `<div class="cg-key">${k.replace(/_/g, " ")}</div><div class="cg-val">${v}</div>`;
                configGrid.appendChild(d);
            });
        } catch (e) {}
    }

    function handleFile(file) {
        if (!isAudioFile(file)) { showError("Unsupported format. Use WAV, MP3, FLAC, OGG, or M4A."); return; }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size);
        fileInfo.classList.remove("hidden");
        analyzeBtn.disabled = false;
    }

    function clearSingleFile() {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        analyzeBtn.disabled = true;
    }

    async function runSingleAnalysis() {
        if (!selectedFile) return;
        setLoading(analyzeBtn, btnText, btnLoader, true);
        hideAnalysisResults();
        loadingSection.classList.remove("hidden");

        try {
            const fd = new FormData();
            fd.append("audio", selectedFile);
            const res = await fetch("/api/predict", { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error);
            loadingSection.classList.add("hidden");
            renderSingleResults(data);
        } catch (e) {
            loadingSection.classList.add("hidden");
            showError(e.message);
        } finally {
            setLoading(analyzeBtn, btnText, btnLoader, false);
        }
    }

    function handleBatch(files) {
        batchFiles = Array.from(files).filter(isAudioFile);
        if (batchFiles.length === 0) { showError("No valid audio files found."); return; }
        batchCount.textContent = batchFiles.length + " files selected";
        batchInfo.classList.remove("hidden");
        batchBtn.disabled = false;
    }

    function clearBatchFiles() {
        batchFiles = [];
        fileInputBatch.value = "";
        batchInfo.classList.add("hidden");
        batchBtn.disabled = true;
    }

    async function runBatchAnalysis() {
        if (!batchFiles.length) return;
        setLoading(batchBtn, batchBtnText, batchBtnLoader, true);
        hideAnalysisResults();
        loadingSection.classList.remove("hidden");

        try {
            const fd = new FormData();
            batchFiles.forEach(f => fd.append("audio", f));
            const res = await fetch("/api/predict-batch", { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error);
            loadingSection.classList.add("hidden");
            renderBatchResults(data);
        } catch (e) {
            loadingSection.classList.add("hidden");
            showError(e.message);
        } finally {
            setLoading(batchBtn, batchBtnText, batchBtnLoader, false);
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            audioChunks = [];

            processor.onaudioprocess = (e) => {
                audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            micStartBtn._stream = stream;
            micStartBtn._source = source;
            micStartBtn._processor = processor;

            micRing.classList.add("recording");
            micStartBtn.classList.add("hidden");
            micStopBtn.classList.remove("hidden");
            micStatus.textContent = "Recording…";
            micStartTime = Date.now();
            micTimer.textContent = "00:00";

            micInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - micStartTime) / 1000);
                const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
                const s = String(elapsed % 60).padStart(2, "0");
                micTimer.textContent = m + ":" + s;
                if (elapsed >= 5) stopRecording();
            }, 200);

        } catch (e) {
            showError("Microphone access denied. Please allow microphone access.");
        }
    }

    async function stopRecording() {
        clearInterval(micInterval);

        if (micStartBtn._processor) { micStartBtn._processor.disconnect(); micStartBtn._source.disconnect(); }
        if (micStartBtn._stream) { micStartBtn._stream.getTracks().forEach(t => t.stop()); }

        micRing.classList.remove("recording");
        micStartBtn.classList.remove("hidden");
        micStopBtn.classList.add("hidden");
        micStatus.textContent = "Processing…";

        const sr = audioContext ? audioContext.sampleRate : 16000;
        const wavBlob = pcmToWav(audioChunks, sr);

        if (audioContext) { audioContext.close(); audioContext = null; }

        hideAnalysisResults();
        loadingSection.classList.remove("hidden");

        try {
            const fd = new FormData();
            fd.append("audio", wavBlob, "microphone_recording.wav");
            const res = await fetch("/api/predict", { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error);
            loadingSection.classList.add("hidden");
            renderSingleResults(data);
        } catch (e) {
            loadingSection.classList.add("hidden");
            showError(e.message);
        }

        micStatus.textContent = "Click to start recording";
        micTimer.textContent = "00:00";
    }

    function renderSingleResults(data) {
        resultsSection.classList.remove("hidden");
        const isAnomaly = data.is_anomaly;
        currentLoss = data.loss;
        currentThreshold = data.threshold;

        verdictBanner.classList.remove("normal", "anomaly");
        verdictBanner.classList.add(isAnomaly ? "anomaly" : "normal");
        verdictIcon.textContent = data.severity === "CRITICAL" ? "🚨" : (data.severity === "WARNING" ? "⚠️" : "✅");
        verdictTitle.textContent = data.verdict;
        verdictDesc.textContent = data.status_summary || (isAnomaly ? "High reconstruction error — deviates from normal patterns" : "Low reconstruction error — matches expected patterns");
        verdictFile.textContent = data.filename;

        // Populate Industrial Diagnostic Card
        const diagStatusSummary = $("#diagStatusSummary");
        const diagActionText = $("#diagActionText");
        const severityBadge = $("#severityBadge");
        const actionBox = $("#actionBox");

        if (diagStatusSummary && data.status_summary) {
            diagStatusSummary.textContent = data.status_summary;
        }
        if (diagActionText && data.recommended_action) {
            diagActionText.textContent = data.recommended_action;
        }
        if (severityBadge && data.severity) {
            severityBadge.textContent = data.severity;
            severityBadge.style.color = data.severity_color || "#00d4aa";
            severityBadge.style.borderColor = data.severity_color || "#00d4aa";
            severityBadge.style.background = (data.severity_color || "#00d4aa") + "22";
        }
        if (actionBox && data.severity_color) {
            actionBox.style.borderLeftColor = data.severity_color;
        }

        // Frequency band errors
        if (data.band_analysis) {
            const bLow = $("#bandLowErr");
            const bMid = $("#bandMidErr");
            const bHigh = $("#bandHighErr");
            if (bLow && data.band_analysis.low) bLow.textContent = data.band_analysis.low.error.toFixed(5);
            if (bMid && data.band_analysis.mid) bMid.textContent = data.band_analysis.mid.error.toFixed(5);
            if (bHigh && data.band_analysis.high) bHigh.textContent = data.band_analysis.high.error.toFixed(5);
        }

        const acc = Math.max(0, Math.min(100, data.accuracy));
        const circ = 2 * Math.PI * 85;
        requestAnimationFrame(() => {
            gaugeFill.style.strokeDashoffset = circ * (1 - acc / 100);
            gaugeFill.style.stroke = isAnomaly ? "#ef4444" : "";
        });
        animateNum(gaugeNumber, 0, acc, 1200);

        lossValue.textContent = data.loss.toFixed(6);
        lossValue.style.color = isAnomaly ? "#ef4444" : "var(--accent-teal)";

        const pct = Math.min(100, (data.loss / 0.1) * 100);
        const tpct = Math.min(100, (data.threshold / 0.1) * 100);
        requestAnimationFrame(() => {
            lossBar.style.width = pct + "%";
            lossBar.classList.toggle("anomaly", isAnomaly);
            lossThreshold.style.left = tpct + "%";
            thresholdLabel.textContent = "Threshold: " + data.threshold;
        });

        infoFile.textContent = data.filename;
        infoThreshold.textContent = data.threshold;
        infoDevice.textContent = data.device === "cuda" ? "GPU (CUDA)" : "CPU";

        threshSlider.value = data.threshold;
        threshCurrent.textContent = data.threshold.toFixed(3);
        updateThresholdVerdict(data.threshold);

        if (data.images) {
            setImg(imgWaveform, data.images.waveform);
            setImg(imgOriginal, data.images.original_spectrogram);
            setImg(imgRecon, data.images.reconstructed_spectrogram);
            setImg(imgError, data.images.error_map);
        }

        setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    }

    function updateThresholdVerdict(threshold) {
        trThresh.textContent = threshold.toFixed(3);
        const isAnomaly = currentLoss > threshold;
        trVerdict.textContent = isAnomaly ? "ANOMALY" : "NORMAL";
        trVerdict.className = "tr-verdict " + (isAnomaly ? "tr-anomaly" : "tr-normal");
    }

    function renderBatchResults(data) {
        batchResultsSection.classList.remove("hidden");
        bsTotal.textContent = data.total;
        bsNormal.textContent = data.normal;
        bsAnomaly.textContent = data.anomalies;
        bsErrors.textContent = data.errors;

        batchTableBody.innerHTML = "";
        data.results.forEach((r, i) => {
            const tr = document.createElement("tr");
            if (r.error) {
                tr.innerHTML = `<td>${i + 1}</td><td>${r.filename}</td><td colspan="3" style="color:var(--accent-amber);font-style:italic">${r.error}</td>`;
            } else {
                const color = r.is_anomaly ? "var(--accent-red)" : "var(--accent-green)";
                tr.innerHTML = `<td>${i + 1}</td><td>${r.filename}</td><td>${r.loss.toFixed(6)}</td><td>${r.accuracy.toFixed(1)}%</td><td style="color:${color};font-weight:600">${r.verdict}</td>`;
            }
            batchTableBody.appendChild(tr);
        });

        setTimeout(() => batchResultsSection.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    }

    async function runEvaluation() {
        setLoading(evalBtn, evalBtnText, evalBtnLoader, true);
        evalLoading.classList.remove("hidden");
        evalResults.classList.add("hidden");
        evalError.classList.add("hidden");

        const normalClasses = $("#evalNormal").value.split(",").map((s) => s.trim()).filter(Boolean);
        const anomalyClasses = $("#evalAnomaly").value.split(",").map((s) => s.trim()).filter(Boolean);

        try {
            const res = await fetch("/api/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ normal_classes: normalClasses, anomaly_classes: anomalyClasses, threshold: currentThreshold }),
            });
            const data = await res.json();
            if (!res.ok) throw { message: data.error, hint: data.hint };

            evalLoading.classList.add("hidden");
            
            $("#esAUC").textContent = data.auc;
            $("#esF1").textContent = data.metrics.f1;
            $("#esPrecision").textContent = data.metrics.precision;
            $("#esRecall").textContent = data.metrics.recall;
            $("#esAccuracy").textContent = data.metrics.accuracy;
            $("#esOptimal").textContent = data.optimal_threshold;

            if (data.images) {
                setImg($("#imgROC"), data.images.roc_curve);
                setImg($("#imgCM"), data.images.confusion_matrix);
                setImg($("#imgDist"), data.images.loss_distribution);
                setImg($("#imgThresh"), data.images.threshold_analysis);
            }

            evalResults.classList.remove("hidden");
            setTimeout(() => evalResults.scrollIntoView({ behavior: "smooth", block: "start" }), 100);

        } catch (e) {
            evalLoading.classList.add("hidden");
            evalError.classList.remove("hidden");
            $("#evalErrorMsg").textContent = e.message || "Evaluation failed.";
            $("#evalHint").textContent = e.hint || "";
        } finally {
            setLoading(evalBtn, evalBtnText, evalBtnLoader, false);
        }
    }

    function hideAnalysisResults() {
        resultsSection.classList.add("hidden");
        batchResultsSection.classList.add("hidden");
        errorSection.classList.add("hidden");
        loadingSection.classList.add("hidden");
    }

    function resetSingleAnalysis() {
        hideAnalysisResults();
        clearSingleFile();
        resetGauge();
        window.scrollTo({ top: $("#uploadSection").offsetTop - 100, behavior: "smooth" });
    }

    function resetBatchAnalysis() {
        hideAnalysisResults();
        clearBatchFiles();
        window.scrollTo({ top: $("#uploadSection").offsetTop - 100, behavior: "smooth" });
    }

    function showError(msg) {
        hideAnalysisResults();
        errorSection.classList.remove("hidden");
        $("#errorMessage").textContent = msg;
    }

    function resetGauge() {
        gaugeFill.style.strokeDashoffset = "534";
        gaugeNumber.textContent = "0";
        lossBar.style.width = "0%";
    }

    function setLoading(btn, textEl, loaderEl, loading) {
        if (loading) { textEl.classList.add("hidden"); loaderEl.classList.remove("hidden"); btn.disabled = true; } 
        else { textEl.classList.remove("hidden"); loaderEl.classList.add("hidden"); btn.disabled = false; }
    }

    function setImg(img, b64) { if (b64) img.src = "data:image/png;base64," + b64; }
    function isAudioFile(file) { const ext = file.name.split(".").pop().toLowerCase(); return ["wav", "mp3", "flac", "ogg", "m4a"].includes(ext); }
    function formatSize(bytes) { if (bytes < 1024) return bytes + " B"; if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB"; return (bytes / 1048576).toFixed(1) + " MB"; }
    
    function animateNum(el, from, to, dur) {
        const start = performance.now();
        const diff = to - from;
        function update(t) {
            const p = Math.min((t - start) / dur, 1);
            const e = 1 - Math.pow(1 - p, 3);
            el.textContent = (from + diff * e).toFixed(1);
            if (p < 1) requestAnimationFrame(update);
            else el.textContent = to.toFixed(1);
        }
        requestAnimationFrame(update);
    }

    // PCM to WAV encoding
    function pcmToWav(chunks, sampleRate) {
        let totalLength = 0; chunks.forEach(c => totalLength += c.length);
        const merged = new Float32Array(totalLength);
        let offset = 0; chunks.forEach(c => { merged.set(c, offset); offset += c.length; });

        const buffer = new ArrayBuffer(44 + merged.length * 2);
        const view = new DataView(buffer);
        const writeStr = (pos, str) => { for(let i=0; i<str.length; i++) view.setUint8(pos+i, str.charCodeAt(i)); };

        writeStr(0, "RIFF"); view.setUint32(4, 36 + merged.length * 2, true);
        writeStr(8, "WAVE"); writeStr(12, "fmt ");
        view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true); view.setUint16(34, 16, true);
        writeStr(36, "data"); view.setUint32(40, merged.length * 2, true);

        for (let i = 0; i < merged.length; i++) {
            const s = Math.max(-1, Math.min(1, merged[i]));
            view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return new Blob([buffer], { type: "audio/wav" });
    }

})();
