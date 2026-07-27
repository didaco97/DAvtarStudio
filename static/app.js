document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('generate-form');
    const faceInput = document.getElementById('face-file');
    const audioInput = document.getElementById('audio-file');
    const tabVideo = document.getElementById('tab-video');
    const tabImage = document.getElementById('tab-image');
    const faceDropMessage = document.querySelector('#face-drop-area .file-message');
    const useEsrganCheckbox = document.getElementById('use-esrgan');
    const faceDetectionBatchSize = document.getElementById('face-det-batch-size');
    const generationBatchSize = document.getElementById('wav2lip-batch-size');
    const resizeFactor = document.getElementById('resize-factor');
    const btn = document.getElementById('generate-btn');
    const btnText = btn.querySelector('span');
    const btnSpinner = document.getElementById('btn-spinner');
    const endTaskButton = document.getElementById('end-task-btn');
    const statusContainer = document.getElementById('status-container');
    const statusText = document.getElementById('status-text');
    const generationClock = document.getElementById('generation-clock');
    const generationClockLabel = document.getElementById('generation-clock-label');
    const generationClockNote = document.getElementById('generation-clock-note');
    const generationClockValue = document.getElementById('generation-clock-value');
    const errorContainer = document.getElementById('error-container');
    const resultContainer = document.getElementById('result-container');
    const resultVideo = document.getElementById('result-video');
    const placeholderText = resultContainer.querySelector('.placeholder-text');
    const downloadActions = document.getElementById('download-actions');
    const downloadLink = document.getElementById('download-link');
    const terminalOutput = document.getElementById('terminal-output');
    const terminalEmpty = document.getElementById('terminal-empty');
    const terminalState = document.getElementById('terminal-state');
    const terminalStateLabel = document.getElementById('terminal-state-label');
    const terminalFollowButton = document.getElementById('terminal-follow');
    const terminalPauseButton = document.getElementById('terminal-pause');
    const terminalClearButton = document.getElementById('terminal-clear');
    const terminalJob = document.getElementById('terminal-job');

    let activeRun = 0;
    let activeController = null;
    let activeJobId = null;
    let logSource = null;
    let logsPaused = false;
    let followTail = true;
    let pausedLogs = [];
    let timerInterval = null;
    let timerStartedAt = null;
    let timerRunId = 0;

    const acceptedVideoExtensions = ['.mp4', '.m4v', '.mov', '.avi', '.webm', '.mkv'];
    const acceptedImageExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
    const acceptedAudioExtensions = ['.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac'];

    let faceMode = 'video'; // 'video' | 'image'

    function setFaceMode(mode) {
        faceMode = mode;
        const isImage = mode === 'image';
        tabVideo.classList.toggle('active', !isImage);
        tabVideo.setAttribute('aria-selected', String(!isImage));
        tabImage.classList.toggle('active', isImage);
        tabImage.setAttribute('aria-selected', String(isImage));
        faceInput.accept = isImage ? 'image/jpeg,image/png,image/webp' : 'video/mp4,video/x-m4v,video/*';
        faceInput.value = '';
        faceDropMessage.textContent = isImage ? 'Drag & drop Image here or click to browse' : 'Drag & drop Video here or click to browse';
        faceDropMessage.style.color = '';
        document.getElementById('face-drop-area').style.borderColor = '';
    }

    tabVideo.addEventListener('click', () => setFaceMode('video'));
    tabImage.addEventListener('click', () => setFaceMode('image'));


    function isAcceptedFile(file, kind) {
        if (!file || file.size <= 0) return false;
        if (kind === 'face-video') {
            const lowerName = file.name.toLowerCase();
            return file.type.startsWith('video/') || acceptedVideoExtensions.some((ext) => lowerName.endsWith(ext));
        }
        if (kind === 'face-image') {
            const lowerName = file.name.toLowerCase();
            return file.type.startsWith('image/') || acceptedImageExtensions.some((ext) => lowerName.endsWith(ext));
        }
        // audio
        const lowerName = file.name.toLowerCase();
        return file.type.startsWith('audio/') || acceptedAudioExtensions.some((ext) => lowerName.endsWith(ext));
    }

    function setupDragAndDrop(dropAreaId, inputElement, kind) {
        const dropArea = document.getElementById(dropAreaId);
        const message = dropArea.querySelector('.file-message');

        inputElement.addEventListener('change', () => {
            const file = inputElement.files[0];
            if (!file) return;
            if (!isAcceptedFile(file, kind === 'face' ? `face-${faceMode}` : kind)) {
                inputElement.value = '';
                const label = kind === 'face' ? faceMode : kind;
                showError(`Please select a non-empty ${label} file.`);
                return;
            }
            message.textContent = `Selected: ${file.name}`;
            message.style.color = '#fff';
            dropArea.style.borderColor = '#6366f1';
            errorContainer.classList.add('hidden');
        });

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
            dropArea.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
            });
        });
        ['dragenter', 'dragover'].forEach((eventName) => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'));
        });
        ['dragleave', 'drop'].forEach((eventName) => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'));
        });
        dropArea.addEventListener('drop', (event) => {
            const file = event.dataTransfer.files[0];
            if (!isAcceptedFile(file, kind === 'face' ? `face-${faceMode}` : kind)) {
                const label = kind === 'face' ? faceMode : kind;
                showError(`Please drop a non-empty ${label} file.`);
                return;
            }
            const transfer = new DataTransfer();
            transfer.items.add(file);
            inputElement.files = transfer.files;
            inputElement.dispatchEvent(new Event('change'));
        });
    }

    function updateModeLabels() {
        btnText.textContent = useEsrganCheckbox.checked ? 'Generate HD Video' : 'Generate Draft Video';
    }

    function formatElapsed(milliseconds) {
        const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        return [hours, minutes, seconds].map((part) => String(part).padStart(2, '0')).join(':');
    }

    function renderGenerationTimer() {
        if (timerStartedAt === null) return;
        generationClockValue.textContent = formatElapsed(performance.now() - timerStartedAt);
    }

    function startGenerationTimer(runId) {
        if (timerInterval !== null) clearInterval(timerInterval);
        timerRunId = runId;
        timerStartedAt = performance.now();
        generationClock.classList.remove('hidden');
        generationClock.dataset.state = 'running';
        generationClockLabel.textContent = 'Elapsed time';
        generationClockNote.textContent = 'Upload + model pipeline';
        generationClockValue.textContent = '00:00:00';
        timerInterval = window.setInterval(renderGenerationTimer, 250);
    }

    function stopGenerationTimer(state, runId) {
        if (timerRunId !== runId || timerStartedAt === null) return;
        const elapsed = performance.now() - timerStartedAt;
        if (timerInterval !== null) clearInterval(timerInterval);
        timerInterval = null;
        timerStartedAt = null;
        const formatted = formatElapsed(elapsed);
        generationClockValue.textContent = formatted;
        generationClock.dataset.state = state;
        generationClockLabel.textContent = state === 'completed' ? 'Completed in' : state === 'cancelled' ? 'Cancelled after' : 'Stopped after';
        generationClockNote.textContent = state === 'completed'
            ? 'Backend generation finished'
            : state === 'cancelled' ? 'Ended from the dashboard' : 'Generation did not complete';
        logBrowserEvent(
            `Generation ${state === 'completed' ? 'completed' : state === 'cancelled' ? 'cancelled' : 'stopped'} after ${formatted}`,
            state === 'completed' ? 'success' : 'warning',
        );
    }

    function setTerminalState(label, state) {
        terminalStateLabel.textContent = label;
        terminalState.dataset.state = state;
    }

    function formatLogTime(timestamp) {
        const date = new Date(timestamp);
        if (Number.isNaN(date.getTime())) return '--:--:--.---';
        const time = date.toLocaleTimeString([], { hour12: false });
        return `${time}.${String(date.getMilliseconds()).padStart(3, '0')}`;
    }

    function appendLog(entry) {
        terminalEmpty?.remove();
        const line = document.createElement('div');
        const level = typeof entry.level === 'string' ? entry.level : 'info';
        line.className = 'terminal-line';
        line.dataset.level = level;

        const time = document.createElement('span');
        time.className = 'terminal-time';
        time.textContent = formatLogTime(entry.timestamp);

        const levelLabel = document.createElement('span');
        levelLabel.className = 'terminal-level';
        levelLabel.textContent = level;

        const source = document.createElement('span');
        source.className = 'terminal-source';
        const legacyEngineLabel = ['Wav', '2', 'Lip'].join('');
        const legacyEnginePattern = new RegExp(legacyEngineLabel, 'gi');
        const displaySource = (entry.source || 'server').replace(legacyEnginePattern, 'DAvtar');
        source.textContent = displaySource;
        source.title = displaySource;

        const message = document.createElement('span');
        message.className = 'terminal-message';
        if (entry.job_id) {
            const jobTag = document.createElement('span');
            jobTag.className = 'terminal-job-tag';
            jobTag.textContent = `[${entry.job_id.slice(0, 8)}]`;
            jobTag.title = entry.job_id;
            message.appendChild(jobTag);
        }
        const displayMessage = String(entry.message || '').replace(legacyEnginePattern, 'DAvtar');
        message.append(document.createTextNode(displayMessage));

        line.append(time, levelLabel, source, message);
        terminalOutput.appendChild(line);
        while (terminalOutput.children.length > 500) {
            terminalOutput.firstElementChild.remove();
        }
        if (followTail) terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    function receiveLog(entry) {
        if (logsPaused) {
            pausedLogs.push(entry);
            if (pausedLogs.length > 500) pausedLogs.shift();
            terminalPauseButton.textContent = `Resume (${pausedLogs.length})`;
            return;
        }
        appendLog(entry);
    }

    function logBrowserEvent(message, level = 'info') {
        receiveLog({
            timestamp: new Date().toISOString(),
            level,
            source: 'browser',
            job_id: activeJobId,
            message,
        });
    }

    function connectLogStream() {
        logSource?.close();
        setTerminalState('Connecting', 'connecting');
        logSource = new EventSource('/api/logs/stream');
        logSource.onopen = () => setTerminalState('Live', 'live');
        logSource.onerror = () => setTerminalState('Reconnecting', 'offline');
        logSource.addEventListener('log', (event) => {
            try {
                receiveLog(JSON.parse(event.data));
            } catch (_) {
                receiveLog({
                    timestamp: new Date().toISOString(),
                    level: 'warning',
                    source: 'client',
                    message: 'Received a malformed log event',
                });
            }
        });
    }

    function clearResult() {
        resultVideo.pause();
        resultVideo.removeAttribute('src');
        resultVideo.load();
        resultVideo.classList.add('hidden');
        resultContainer.classList.add('empty');
        placeholderText.textContent = 'Your generated video will appear here';
        placeholderText.style.display = 'block';
        downloadLink.removeAttribute('href');
        downloadActions.classList.add('hidden');
    }

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.classList.remove('hidden');
        statusContainer.classList.add('hidden');
    }

    function resetButton() {
        btn.disabled = false;
        btnSpinner.classList.add('hidden');
        endTaskButton.classList.add('hidden');
        endTaskButton.disabled = false;
        endTaskButton.innerHTML = '<span aria-hidden="true">×</span> End task';
        updateModeLabels();
    }

    async function errorMessage(response, fallback) {
        try {
            const body = await response.json();
            return body.detail || body.error || fallback;
        } catch (_) {
            return fallback;
        }
    }

    function delay(ms, signal) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(resolve, ms);
            signal.addEventListener('abort', () => {
                clearTimeout(timeout);
                reject(new DOMException('Aborted', 'AbortError'));
            }, { once: true });
        });
    }

    function loadPreview(mediaUrl, signal) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => finish(new Error('Preview metadata timed out after 45 seconds.')), 45000);
            const onLoaded = () => finish();
            const onError = () => {
                const code = resultVideo.error?.code || 0;
                const detail = resultVideo.error?.message || 'unknown media error';
                finish(new Error(`Media error ${code}: ${detail}`));
            };
            const onAbort = () => finish(new DOMException('Aborted', 'AbortError'));
            const finish = (error) => {
                clearTimeout(timeout);
                resultVideo.removeEventListener('loadedmetadata', onLoaded);
                resultVideo.removeEventListener('error', onError);
                signal.removeEventListener('abort', onAbort);
                if (error) reject(error); else resolve();
            };
            resultVideo.addEventListener('loadedmetadata', onLoaded, { once: true });
            resultVideo.addEventListener('error', onError, { once: true });
            signal.addEventListener('abort', onAbort, { once: true });
            resultVideo.src = mediaUrl.toString();
            resultVideo.load();
        });
    }

    async function displayResult(resultUrl, signal) {
        if (typeof resultUrl !== 'string' || !resultUrl.startsWith('/')) {
            throw new Error('The server completed the job without a valid output URL.');
        }

        statusText.textContent = 'Finalizing and checking the generated video...';
        const downloadUrl = new URL(resultUrl, window.location.origin);
        downloadUrl.searchParams.set('v', Date.now().toString());
        const check = await fetch(downloadUrl, { method: 'HEAD', cache: 'no-store', signal });
        if (!check.ok) {
            throw new Error('The pipeline finished, but the generated video is unavailable.');
        }

        downloadLink.href = downloadUrl.toString();
        downloadActions.classList.remove('hidden');
        let lastPreviewError = null;
        for (let attempt = 1; attempt <= 2; attempt += 1) {
            const previewUrl = new URL(resultUrl, window.location.origin);
            previewUrl.searchParams.set('v', `${Date.now()}-${attempt}`);
            if (attempt > 1) {
                statusText.textContent = 'Retrying the browser preview...';
                await delay(750, signal);
            }
            resultVideo.pause();
            resultVideo.removeAttribute('src');
            resultVideo.load();
            try {
                await loadPreview(previewUrl, signal);
                resultContainer.classList.remove('empty');
                placeholderText.style.display = 'none';
                resultVideo.classList.remove('hidden');
                logBrowserEvent(`Preview loaded on attempt ${attempt}`, 'success');
                return;
            } catch (error) {
                if (error.name === 'AbortError') throw error;
                lastPreviewError = error;
                logBrowserEvent(`Preview attempt ${attempt} failed: ${error.message}`, 'warning');
            }
        }

        resultVideo.classList.add('hidden');
        resultContainer.classList.add('empty');
        placeholderText.textContent = 'Preview unavailable — use Download Result below';
        placeholderText.style.display = 'block';
        showError(`The browser preview could not open (${lastPreviewError?.message || 'unknown error'}). Your generated video is still available to download.`);
    }

    async function pollStatus(jobId, runId, signal, enhanced) {
        while (!signal.aborted && runId === activeRun) {
            const response = await fetch(`/api/status/${encodeURIComponent(jobId)}`, {
                cache: 'no-store',
                signal,
            });
            if (!response.ok) {
                throw new Error(await errorMessage(response, 'Failed to check job status.'));
            }

            const job = await response.json();
            if (job.status === 'completed') {
                stopGenerationTimer('completed', runId);
                terminalJob.textContent = `Completed job · ${jobId.slice(0, 8)}`;
                await displayResult(job.result_url, signal);
                return;
            }
            if (job.status === 'failed') {
                stopGenerationTimer('failed', runId);
                terminalJob.textContent = `Failed job · ${jobId.slice(0, 8)}`;
                throw new Error(job.error || 'The video pipeline failed.');
            }
            if (job.status === 'cancelled') {
                stopGenerationTimer('cancelled', runId);
                terminalJob.textContent = `Cancelled job · ${jobId.slice(0, 8)}`;
                throw new Error(job.error || 'Generation was ended.');
            }
            if (job.status !== 'processing') {
                throw new Error('The server returned an unknown job status.');
            }

            const mode = enhanced ? 'DAvtar and Real-ESRGAN' : 'DAvtar';
            statusText.textContent = `Running ${mode}... ${Number.isFinite(job.progress) ? `${job.progress}%` : ''}`;
            await delay(3000, signal);
        }
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const face = faceInput.files[0];
        const audio = audioInput.files[0];
        const faceKind = `face-${faceMode}`;
        if (!isAcceptedFile(face, faceKind) || !isAcceptedFile(audio, 'audio')) {
            showError(`Please select a non-empty ${faceMode} and audio file.`);
            return;
        }

        activeController?.abort();
        activeController = new AbortController();
        const signal = activeController.signal;
        const runId = ++activeRun;
        const enhanced = useEsrganCheckbox.checked;
        startGenerationTimer(runId);

        clearResult();
        errorContainer.classList.add('hidden');
        statusContainer.classList.remove('hidden');
        statusText.textContent = 'Uploading files...';
        btn.disabled = true;
        btnText.textContent = 'Processing...';
        btnSpinner.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('face', face);
            formData.append('audio', audio);
            formData.append('use_esrgan', enhanced.toString());
            formData.append('face_det_batch_size', faceDetectionBatchSize.value);
            formData.append('wav2lip_batch_size', generationBatchSize.value);
            formData.append('resize_factor', resizeFactor.value);
            const response = await fetch('/api/generate', { method: 'POST', body: formData, signal });
            if (!response.ok) {
                throw new Error(await errorMessage(response, 'Failed to start the video job.'));
            }
            const body = await response.json();
            if (typeof body.job_id !== 'string' || !body.job_id) {
                throw new Error('The server did not return a valid job ID.');
            }

            activeJobId = body.job_id;
            terminalJob.textContent = `Active job · ${activeJobId.slice(0, 8)}`;
            endTaskButton.classList.remove('hidden');

            await pollStatus(body.job_id, runId, signal, enhanced);
            if (runId === activeRun) statusContainer.classList.add('hidden');
        } catch (error) {
            if (error.name !== 'AbortError' && runId === activeRun) {
                stopGenerationTimer('failed', runId);
                clearResult();
                showError(error.message || 'The video job failed.');
            }
        } finally {
            if (runId === activeRun) resetButton();
        }
    });

    setupDragAndDrop('face-drop-area', faceInput, 'face');
    setupDragAndDrop('audio-drop-area', audioInput, 'audio');
    useEsrganCheckbox.addEventListener('change', updateModeLabels);
    endTaskButton.addEventListener('click', async () => {
        if (!activeJobId) return;

        const jobId = activeJobId;
        endTaskButton.disabled = true;
        endTaskButton.textContent = 'Ending task…';
        statusText.textContent = 'Stopping the active pipeline…';

        try {
            const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
            if (!response.ok) {
                throw new Error(await errorMessage(response, 'Could not end the generation task.'));
            }
            logBrowserEvent('End-task request sent to backend', 'warning');
        } catch (error) {
            endTaskButton.disabled = false;
            endTaskButton.innerHTML = '<span aria-hidden="true">×</span> End task';
            showError(error.message || 'Could not end the generation task.');
        }
    });
    terminalPauseButton.addEventListener('click', () => {
        logsPaused = !logsPaused;
        terminalPauseButton.setAttribute('aria-pressed', logsPaused.toString());
        if (logsPaused) {
            terminalPauseButton.textContent = 'Resume';
            return;
        }
        const buffered = pausedLogs;
        pausedLogs = [];
        terminalPauseButton.textContent = 'Pause';
        buffered.forEach(appendLog);
    });
    terminalFollowButton.addEventListener('click', () => {
        followTail = !followTail;
        terminalFollowButton.setAttribute('aria-pressed', followTail.toString());
        terminalFollowButton.textContent = followTail ? 'Follow tail' : 'Follow off';
        if (followTail) terminalOutput.scrollTop = terminalOutput.scrollHeight;
    });
    terminalClearButton.addEventListener('click', () => {
        terminalOutput.replaceChildren();
        pausedLogs = [];
        if (logsPaused) terminalPauseButton.textContent = 'Resume';
    });
    window.addEventListener('beforeunload', () => logSource?.close());
    connectLogStream();
    updateModeLabels();
});
