/**
 * Duplicate Image Detector - Web UI
 */

class DuplicateDetector {
    constructor() {
        this.currentJobId = null;
        this.pollInterval = null;
        this.results = null;
        
        this.init();
    }
    
    init() {
        // DOM elements
        this.elements = {
            directoryInput: document.getElementById('directory-input'),
            browseBtn: document.getElementById('browse-btn'),
            thresholdSlider: document.getElementById('threshold-slider'),
            thresholdValue: document.getElementById('threshold-value'),
            recursiveCheck: document.getElementById('recursive-check'),
            skipCacheCheck: document.getElementById('skip-cache-check'),
            scanBtn: document.getElementById('scan-btn'),
            
            progressPanel: document.getElementById('progress-panel'),
            progressStatus: document.getElementById('progress-status'),
            progressBar: document.getElementById('progress-bar'),
            progressCount: document.getElementById('progress-count'),
            progressTotal: document.getElementById('progress-total'),
            progressFile: document.getElementById('progress-file'),
            
            resultsPanel: document.getElementById('results-panel'),
            resultsSummary: document.getElementById('results-summary'),
            groupsContainer: document.getElementById('groups-container'),
            exportBtn: document.getElementById('export-btn'),
            
            performancePanel: document.getElementById('performance-panel'),
            perfTotalTime: document.getElementById('perf-total-time'),
            perfScan: document.getElementById('perf-scan'),
            perfCache: document.getElementById('perf-cache'),
            perfEmbedding: document.getElementById('perf-embedding'),
            perfDetection: document.getElementById('perf-detection'),
            perfCachedCount: document.getElementById('perf-cached-count'),
            perfEmbeddedCount: document.getElementById('perf-embedded-count'),
            perfSpeed: document.getElementById('perf-speed'),
            
            emptyState: document.getElementById('empty-state'),
            
            actionModal: document.getElementById('action-modal'),
            modalTitle: document.getElementById('modal-title'),
            modalBody: document.getElementById('modal-body'),
            modalCancel: document.getElementById('modal-cancel'),
            modalConfirm: document.getElementById('modal-confirm'),
            modalClose: document.getElementById('modal-close'),
            
            previewModal: document.getElementById('preview-modal'),
            previewTitle: document.getElementById('preview-title'),
            previewImage: document.getElementById('preview-image'),
            previewInfo: document.getElementById('preview-info'),
            previewClose: document.getElementById('preview-close'),
            
            modelStatus: document.getElementById('model-status')
        };
        
        this.bindEvents();
    }
    
    bindEvents() {
        // Browse button - folder picker
        this.elements.browseBtn.addEventListener('click', () => this.browseFolder());
        
        // Threshold slider
        this.elements.thresholdSlider.addEventListener('input', (e) => {
            this.elements.thresholdValue.textContent = `${Math.round(e.target.value * 100)}%`;
        });
        
        // Scan button
        this.elements.scanBtn.addEventListener('click', () => this.startScan());
        
        // Export button
        this.elements.exportBtn.addEventListener('click', () => this.exportResults());
        
        // Modal close buttons
        this.elements.modalCancel.addEventListener('click', () => this.closeModal('action'));
        this.elements.modalClose.addEventListener('click', () => this.closeModal('action'));
        this.elements.previewClose.addEventListener('click', () => this.closeModal('preview'));
        
        // Backdrop clicks
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal.id === 'action-modal') this.closeModal('action');
                if (modal.id === 'preview-modal') this.closeModal('preview');
            });
        });
        
        // Enter key on directory input
        this.elements.directoryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.startScan();
        });
    }
    
    async browseFolder() {
        // Disable button while dialog is open
        this.elements.browseBtn.disabled = true;
        this.elements.browseBtn.innerHTML = `
            <svg class="btn-icon spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-dasharray="30 70"/>
            </svg>
            Opening...
        `;
        
        try {
            // Use backend native folder picker
            const response = await fetch('/api/browse');
            const data = await response.json();
            
            if (data.path) {
                this.elements.directoryInput.value = data.path;
            }
        } catch (error) {
            console.error('Browse error:', error);
            this.showError('Failed to open folder picker');
        } finally {
            // Re-enable button
            this.elements.browseBtn.disabled = false;
            this.elements.browseBtn.innerHTML = `
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/>
                </svg>
                Browse
            `;
        }
    }
    
    async startScan() {
        const directory = this.elements.directoryInput.value.trim();
        
        if (!directory) {
            this.showError('Please enter a directory path');
            return;
        }
        
        const threshold = parseFloat(this.elements.thresholdSlider.value);
        const recursive = this.elements.recursiveCheck.checked;
        const skipCache = this.elements.skipCacheCheck.checked;
        const method = document.querySelector('input[name="method"]:checked').value;
        
        // Update UI
        this.elements.scanBtn.disabled = true;
        this.elements.scanBtn.innerHTML = `
            <svg class="btn-icon spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-dasharray="30 70"/>
            </svg>
            Scanning...
        `;
        
        this.elements.emptyState.style.display = 'none';
        this.elements.progressPanel.style.display = 'block';
        this.elements.resultsPanel.style.display = 'none';
        this.elements.performancePanel.style.display = 'none';
        
        this.setStatus('Loading model...');
        
        try {
            const response = await fetch('/api/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory, threshold, recursive, skip_cache: skipCache, method })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Scan failed');
            }
            
            const data = await response.json();
            this.currentJobId = data.job_id;
            
            // Start polling
            this.pollProgress();
            
        } catch (error) {
            this.showError(error.message);
            this.resetScanButton();
        }
    }
    
    async pollProgress() {
        if (!this.currentJobId) return;
        
        try {
            const response = await fetch(`/api/scan/${this.currentJobId}`);
            const data = await response.json();
            
            // Update progress UI
            this.elements.progressStatus.textContent = data.status;
            this.elements.progressCount.textContent = data.progress;
            this.elements.progressTotal.textContent = data.total;
            this.elements.progressFile.textContent = data.current_file || '—';
            
            if (data.total > 0) {
                const percent = (data.progress / data.total) * 100;
                this.elements.progressBar.style.width = `${percent}%`;
            }
            
            if (data.status === 'running') {
                this.setStatus('Processing...');
                this.pollInterval = setTimeout(() => this.pollProgress(), 500);
            } else if (data.status === 'completed') {
                this.setStatus('Ready');
                this.displayPerformance(data.performance);
                await this.loadResults();
                this.resetScanButton();
            } else if (data.status === 'failed') {
                this.showError(data.error || 'Scan failed');
                this.resetScanButton();
            }
            
        } catch (error) {
            console.error('Poll error:', error);
            this.pollInterval = setTimeout(() => this.pollProgress(), 1000);
        }
    }
    
    displayPerformance(perf) {
        if (!perf) return;
        
        this.elements.performancePanel.style.display = 'block';
        this.elements.perfTotalTime.textContent = `${perf.total_time}s`;
        this.elements.perfScan.textContent = `${perf.scan_files}s`;
        this.elements.perfCache.textContent = `${perf.load_cache}s`;
        this.elements.perfEmbedding.textContent = `${perf.embedding}s`;
        this.elements.perfDetection.textContent = `${perf.detection}s`;
        this.elements.perfCachedCount.textContent = perf.cached_count;
        this.elements.perfEmbeddedCount.textContent = perf.embedded_count;
        this.elements.perfSpeed.textContent = `${perf.images_per_second} img/s`;
        
        // Update labels based on method
        const isPhash = perf.method === 'phash';
        document.querySelector('#perf-embedding').closest('.perf-item').querySelector('.perf-label').textContent = 
            isPhash ? 'Hashing' : 'Embedding';
        document.querySelector('#perf-embedded-count').closest('.perf-item').querySelector('.perf-label').textContent = 
            isPhash ? 'Hashed' : 'Newly Embedded';
    }
    
    async loadResults() {
        if (!this.currentJobId) return;
        
        try {
            const response = await fetch(`/api/scan/${this.currentJobId}/results`);
            const data = await response.json();
            
            this.results = data.groups;
            this.renderResults();
            
        } catch (error) {
            this.showError('Failed to load results');
        }
    }
    
    renderResults() {
        if (!this.results || this.results.length === 0) {
            this.elements.resultsPanel.style.display = 'none';
            this.elements.emptyState.style.display = 'flex';
            this.elements.emptyState.querySelector('.empty-title').textContent = 'No duplicates found!';
            this.elements.emptyState.querySelector('.empty-text').textContent = 
                'Your images are all unique. Try lowering the similarity threshold if you want to find near-duplicates.';
            return;
        }
        
        const totalDuplicates = this.results.reduce((sum, g) => sum + (g.count - 1), 0);
        this.elements.resultsSummary.textContent = 
            `${this.results.length} group${this.results.length !== 1 ? 's' : ''}, ${totalDuplicates} duplicate${totalDuplicates !== 1 ? 's' : ''}`;
        
        this.elements.groupsContainer.innerHTML = this.results.map(group => this.renderGroup(group)).join('');
        this.elements.resultsPanel.style.display = 'block';
        
        // Bind group events
        this.bindGroupEvents();
    }
    
    renderGroup(group) {
        const images = [group.original, ...group.duplicates];
        const scores = group.similarity_scores;
        
        return `
            <div class="group-card" data-group-id="${group.id}">
                <div class="group-header">
                    <div class="group-title">
                        <span class="group-id">#${group.id + 1}</span>
                        <span class="group-count">${group.count} images</span>
                    </div>
                    <div class="group-actions">
                        <button class="group-btn move-btn" data-group-id="${group.id}">Move</button>
                        <button class="group-btn danger delete-btn" data-group-id="${group.id}">Delete</button>
                    </div>
                </div>
                <div class="group-content">
                    <div class="group-images">
                        ${images.map((path, i) => {
                            const filename = path.split(/[/\\]/).pop();
                            return `
                            <div class="image-item" data-path="${path}" title="${path}">
                                <img src="/api/thumbnail?path=${encodeURIComponent(path)}&size=300" alt="Image ${i + 1}" loading="lazy">
                                <span class="image-badge ${i === 0 ? 'original' : 'duplicate'}">
                                    ${i === 0 ? 'Original' : 'Duplicate'}
                                </span>
                                <span class="image-filename">${filename}</span>
                                ${i > 0 ? `<span class="image-score">${(scores[i] * 100).toFixed(0)}%</span>` : ''}
                            </div>
                        `}).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    bindGroupEvents() {
        // Image clicks for preview
        document.querySelectorAll('.image-item').forEach(item => {
            item.addEventListener('click', () => {
                const path = item.dataset.path;
                this.showPreview(path);
            });
        });
        
        // Move buttons
        document.querySelectorAll('.move-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const groupId = parseInt(btn.dataset.groupId);
                this.showMoveModal(groupId);
            });
        });
        
        // Delete buttons
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const groupId = parseInt(btn.dataset.groupId);
                this.showDeleteModal(groupId);
            });
        });
    }
    
    showPreview(path) {
        this.elements.previewTitle.textContent = path.split(/[/\\]/).pop();
        this.elements.previewImage.src = `/api/thumbnail?path=${encodeURIComponent(path)}&size=800`;
        this.elements.previewInfo.textContent = path;
        this.elements.previewModal.classList.add('active');
    }
    
    showMoveModal(groupId) {
        const group = this.results.find(g => g.id === groupId);
        const dupCount = group.duplicates.length;
        
        this.elements.modalTitle.textContent = 'Move Duplicates';
        this.elements.modalBody.innerHTML = `
            <p style="margin-bottom: 16px; color: var(--text-secondary);">
                Move <strong>${dupCount}</strong> duplicate${dupCount !== 1 ? 's' : ''} to a folder?
            </p>
            <div class="form-group" style="margin-bottom: 0;">
                <label class="form-label">Destination Folder</label>
                <input type="text" id="move-destination" class="form-input" placeholder="C:\\duplicates or /home/user/duplicates">
            </div>
        `;
        
        this.elements.modalConfirm.textContent = 'Move';
        this.elements.modalConfirm.className = 'btn btn-primary';
        this.elements.modalConfirm.onclick = () => this.executeMove(groupId);
        
        this.elements.actionModal.classList.add('active');
        
        setTimeout(() => document.getElementById('move-destination')?.focus(), 100);
    }
    
    showDeleteModal(groupId) {
        const group = this.results.find(g => g.id === groupId);
        const dupCount = group.duplicates.length;
        
        this.elements.modalTitle.textContent = 'Delete Duplicates';
        this.elements.modalBody.innerHTML = `
            <p style="color: var(--text-secondary);">
                Permanently delete <strong>${dupCount}</strong> duplicate${dupCount !== 1 ? 's' : ''}?
            </p>
            <p style="color: var(--danger); margin-top: 12px; font-size: 0.875rem;">
                ⚠️ This action cannot be undone. The original file will be kept.
            </p>
        `;
        
        this.elements.modalConfirm.textContent = 'Delete';
        this.elements.modalConfirm.className = 'btn btn-danger';
        this.elements.modalConfirm.onclick = () => this.executeDelete(groupId);
        
        this.elements.actionModal.classList.add('active');
    }
    
    async executeMove(groupId) {
        const destination = document.getElementById('move-destination')?.value.trim();
        
        if (!destination) {
            this.showError('Please enter a destination folder');
            return;
        }
        
        try {
            const response = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: this.currentJobId,
                    action: 'move',
                    group_ids: [groupId],
                    destination
                })
            });
            
            const data = await response.json();
            
            this.closeModal('action');
            
            if (data.moved > 0) {
                this.showSuccess(`Moved ${data.moved} file${data.moved !== 1 ? 's' : ''}`);
                // Remove group from results
                this.results = this.results.filter(g => g.id !== groupId);
                this.renderResults();
            }
            
        } catch (error) {
            this.showError('Move failed: ' + error.message);
        }
    }
    
    async executeDelete(groupId) {
        try {
            const response = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: this.currentJobId,
                    action: 'delete',
                    group_ids: [groupId]
                })
            });
            
            const data = await response.json();
            
            this.closeModal('action');
            
            if (data.deleted > 0) {
                this.showSuccess(`Deleted ${data.deleted} file${data.deleted !== 1 ? 's' : ''}`);
                // Remove group from results
                this.results = this.results.filter(g => g.id !== groupId);
                this.renderResults();
            }
            
        } catch (error) {
            this.showError('Delete failed: ' + error.message);
        }
    }
    
    async exportResults() {
        if (!this.currentJobId) return;
        
        try {
            const response = await fetch(`/api/report/${this.currentJobId}?format=json`);
            const blob = await response.blob();
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `duplicates_${this.currentJobId}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
        } catch (error) {
            this.showError('Export failed');
        }
    }
    
    closeModal(type) {
        if (type === 'action') {
            this.elements.actionModal.classList.remove('active');
        } else if (type === 'preview') {
            this.elements.previewModal.classList.remove('active');
        }
    }
    
    resetScanButton() {
        this.elements.scanBtn.disabled = false;
        this.elements.scanBtn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"/>
                <path d="M21 21l-4.35-4.35"/>
            </svg>
            Start Scan
        `;
    }
    
    setStatus(text) {
        const statusText = this.elements.modelStatus.querySelector('.status-text');
        if (statusText) statusText.textContent = text;
    }
    
    showError(message) {
        // Simple alert for now - could be replaced with a toast
        alert('Error: ' + message);
    }
    
    showSuccess(message) {
        // Simple alert for now - could be replaced with a toast
        alert(message);
    }
}

// Add spin animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .spin {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(style);

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DuplicateDetector();
});

