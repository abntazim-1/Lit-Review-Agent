// DOM Elements
const formPanel = document.getElementById('panel-form');
const progressPanel = document.getElementById('panel-progress');
const resultsPanel = document.getElementById('panel-results');

const reviewForm = document.getElementById('review-request-form');
const advancedToggle = document.getElementById('advanced-toggle');
const advancedChevron = document.getElementById('advanced-chevron');
const advancedOptions = document.getElementById('advanced-options');
const submitBtn = document.getElementById('btn-submit');
const newJobBtn = document.getElementById('btn-new-job');
const exportJsonBtn = document.getElementById('btn-export-json');

const progressTopicTitle = document.getElementById('progress-topic-title');
const progressBadgeIteration = document.getElementById('progress-badge-iteration');
const logStream = document.getElementById('log-stream');
const progressClustersList = document.getElementById('progress-clusters-list');

const resultsTopicTitle = document.getElementById('results-topic-title');
const synthesisBackground = document.getElementById('synthesis-background');
const synthesisMethodology = document.getElementById('synthesis-methodology');
const synthesisFindings = document.getElementById('synthesis-findings');
const synthesisOpenQuestions = document.getElementById('synthesis-open-questions');

const badgeContradictions = document.getElementById('badge-contradictions');
const badgeReferences = document.getElementById('badge-references');
const contradictionsGrid = document.getElementById('contradictions-grid');
const resultsClustersGrid = document.getElementById('results-clusters-grid');
const referencesList = document.getElementById('references-list');

const historyList = document.getElementById('history-list');
const topStatusText = document.getElementById('top-status-text');
const topStatusBar = document.getElementById('top-status-bar').querySelector('.pulse-indicator');

// Application State
let activeJobId = null;
let pollIntervalId = null;
let currentIteration = 1;
let lastKnownStatus = null;
let lastRenderedClusterCount = 0;
let lastRenderedPaperCount = 0;
let lastRenderedLogIndex = 0;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
  loadHistorySidebar();
  setupEventListeners();
});

// Event Listeners Setup
function setupEventListeners() {
  // Advanced Options Toggle
  advancedToggle.addEventListener('click', () => {
    advancedToggle.classList.toggle('open');
    advancedOptions.classList.toggle('hidden');
  });

  // Form Submit
  reviewForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(reviewForm);
    const requestData = {
      topic: formData.get('topic'),
      max_sub_questions: formData.get('max_sub_questions') ? parseInt(formData.get('max_sub_questions')) : undefined,
      max_papers_per_sub_question: formData.get('max_papers_per_sub_question') ? parseInt(formData.get('max_papers_per_sub_question')) : undefined,
    };

    await submitNewReview(requestData);
  });

  // New Review Button (Sidebar / Reset)
  newJobBtn.addEventListener('click', () => {
    stopPolling();
    activeJobId = null;
    reviewForm.reset();
    showPanel(formPanel);
    updateTopStatus('Ready for new task', 'status-idle');
  });

  // Tab Switching
  document.querySelectorAll('.tab-link').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.tab-link').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(p => p.classList.add('hidden'));

      button.classList.add('active');
      const targetId = button.getAttribute('data-tab');
      document.getElementById(targetId).classList.remove('hidden');
    });
  });

  // Export JSON
  exportJsonBtn.addEventListener('click', async () => {
    if (!activeJobId) return;
    try {
      const response = await fetch(`/reviews/${activeJobId}`);
      if (response.ok) {
        const data = await response.json();
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", `lit_review_${activeJobId}.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
      }
    } catch (err) {
      console.error("Export failed", err);
    }
  });
}

// Panel switcher
function showPanel(panel) {
  formPanel.classList.remove('active');
  progressPanel.classList.remove('active');
  resultsPanel.classList.remove('active');
  panel.classList.add('active');
}

// Status bar update
function updateTopStatus(text, className) {
  topStatusText.textContent = text;
  topStatusBar.className = 'pulse-indicator ' + className;
}

// Logger stream writer
function writeLog(message, type = 'info') {
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `[${time}] ${message}`;
  logStream.appendChild(entry);
  logStream.scrollTop = logStream.scrollHeight;
}

// Local Storage & History sidebar
function saveToHistory(jobId, topic) {
  const history = JSON.parse(localStorage.getItem('lit_review_history') || '[]');
  // Avoid duplicates
  if (!history.some(item => item.id === jobId)) {
    history.unshift({ id: jobId, topic: topic, date: new Date().toLocaleString() });
    localStorage.setItem('lit_review_history', JSON.stringify(history));
    loadHistorySidebar();
  }
}

function loadHistorySidebar() {
  const history = JSON.parse(localStorage.getItem('lit_review_history') || '[]');
  historyList.innerHTML = '';
  
  if (history.length === 0) {
    historyList.innerHTML = '<li class="empty-history">No recent reviews</li>';
    return;
  }

  history.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item.topic;
    li.title = item.topic;
    li.setAttribute('data-id', item.id);
    if (activeJobId === item.id) {
      li.classList.add('active');
    }
    li.addEventListener('click', () => loadHistoricalJob(item.id));
    historyList.appendChild(li);
  });
}

async function loadHistoricalJob(jobId) {
  stopPolling();
  activeJobId = jobId;
  loadHistorySidebar(); // Update active class
  updateTopStatus('Loading review results...', 'status-running');

  try {
    const response = await fetch(`/reviews/${jobId}`);
    if (response.ok) {
      const job = await response.json();
      if (job.status === 'complete') {
        renderResults(job);
        showPanel(resultsPanel);
        updateTopStatus('Completed', 'status-success-dot');
      } else if (job.status === 'failed') {
        alert(`This review failed: ${job.error}`);
        updateTopStatus('Failed', 'status-failed-dot');
      } else {
        // Resume polling if it is still running
        progressTopicTitle.textContent = job.request.topic;
        resetProgressSteps();
        logStream.innerHTML = '<div class="log-entry system">Resuming progress monitoring...</div>';
        progressClustersList.innerHTML = '';
        lastKnownStatus = null;
        lastRenderedClusterCount = 0;
        lastRenderedPaperCount = 0;
        lastRenderedLogIndex = 0;
        currentIteration = 1;
        showPanel(progressPanel);
        startPolling(jobId);
      }
    } else {
      alert("Failed to load historical review.");
      updateTopStatus('Error loading job', 'status-failed-dot');
    }
  } catch (err) {
    console.error("Failed loading job", err);
    updateTopStatus('Connection error', 'status-failed-dot');
  }
}

// Form Submission Action
async function submitNewReview(requestData) {
  submitBtn.disabled = true;
  submitBtn.querySelector('span').textContent = 'Orchestrating...';
  updateTopStatus('Submitting review job...', 'status-running');

  try {
    const response = await fetch('/reviews', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      throw new Error(`Server returned code ${response.status}`);
    }

    const jobHandle = await response.json();
    activeJobId = jobHandle.job_id;
    
    // Clear and transition to progress panel
    progressTopicTitle.textContent = requestData.topic;
    progressBadgeIteration.textContent = "Round 1";
    logStream.innerHTML = '<div class="log-entry system">Job submitted successfully. Pipeline initialized.</div>';
    progressClustersList.innerHTML = '<p class="empty-text">Determining themes...</p>';
    resetProgressSteps();
    lastRenderedLogIndex = 0;
    
    showPanel(progressPanel);
    saveToHistory(activeJobId, requestData.topic);
    
    // Start status polling
    startPolling(activeJobId);
  } catch (err) {
    console.error("Submission failed", err);
    alert(`Failed to start literature review: ${err.message}`);
    updateTopStatus('Submission failed', 'status-failed-dot');
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector('span').textContent = 'Initiate Deep Review';
  }
}

// Progress Stepper Updates
function resetProgressSteps() {
  document.querySelectorAll('.step').forEach(step => {
    step.className = 'step';
  });
}

function updateProgressSteps(status) {
  const steps = ['decomposing', 'researching', 'detecting_contradictions', 'synthesizing', 'evaluating'];
  const activeIndex = steps.indexOf(status);

  steps.forEach((stepName, index) => {
    const element = document.getElementById(`step-${stepName}`);
    if (!element) return;
    
    if (index < activeIndex) {
      element.className = 'step completed';
    } else if (index === activeIndex) {
      element.className = 'step active';
    } else {
      element.className = 'step';
    }
  });
}

// Polling Loop Manager
function startPolling(jobId) {
  stopPolling();
  updateTopStatus('Processing...', 'status-running');
  
  pollIntervalId = setInterval(async () => {
    try {
      const response = await fetch(`/reviews/${jobId}`);
      if (!response.ok) {
        throw new Error("HTTP error polling status");
      }
      const job = await response.json();
      
      handlePollUpdate(job);
    } catch (err) {
      console.error("Polling error", err);
      writeLog(`Connection error during status polling: ${err.message}`, 'error');
    }
  }, 3000);
}

function stopPolling() {
  if (pollIntervalId) {
    clearInterval(pollIntervalId);
    pollIntervalId = null;
  }
}

// Process Polling Response
function handlePollUpdate(job) {
  const { status, clusters, sub_questions, agent_results, contradictions, error, logs } = job;
  
  if (status !== lastKnownStatus) {
    updateTopStatus(status.charAt(0).toUpperCase() + status.slice(1) + '...', 'status-running');
    lastKnownStatus = status;
    updateProgressSteps(status);
  }

  // Render backend logs
  if (logs && logs.length > lastRenderedLogIndex) {
    for (let i = lastRenderedLogIndex; i < logs.length; i++) {
      const logLine = logs[i];
      const match = logLine.match(/^\[(INFO|WARNING|ERROR)\]\s*(.*)$/i);
      if (match) {
        const level = match[1].toLowerCase();
        const msg = match[2];
        writeLog(msg, level === 'warning' ? 'system' : level);
      } else {
        writeLog(logLine, 'info');
      }
    }
    lastRenderedLogIndex = logs.length;
  }

  // Monitor Iteration / Round (UI Badge update)
  const clusterCount = clusters.length;
  if (clusterCount > lastRenderedClusterCount) {
    if (lastRenderedClusterCount > 0) {
      currentIteration++;
      progressBadgeIteration.textContent = `Round ${currentIteration}`;
    }
    renderProgressClusters(clusters);
    lastRenderedClusterCount = clusterCount;
  }

  // Handle Pipeline Termination
  if (status === 'complete') {
    stopPolling();
    updateTopStatus('Completed', 'status-success-dot');
    setTimeout(() => {
      renderResults(job);
      showPanel(resultsPanel);
    }, 1500);
  } else if (status === 'failed') {
    stopPolling();
    updateTopStatus('Failed', 'status-failed-dot');
    alert(`Review generation failed: ${error}`);
  }
}

// Render dynamic elements in Progress Card
function renderProgressClusters(clusters) {
  progressClustersList.innerHTML = '';
  clusters.forEach(c => {
    const div = document.createElement('div');
    div.className = 'cluster-bubble';
    
    const qList = c.sub_questions.map(q => `<li>${q.text}</li>`).join('');
    div.innerHTML = `
      <h5>${c.theme}</h5>
      <ul style="margin-left:1.25rem; font-size:0.8rem; color:var(--text-secondary)">
        ${qList}
      </ul>
    `;
    progressClustersList.appendChild(div);
  });
}

// Render Review Results Screen
function renderResults(job) {
  const { topic, result, clusters, sub_questions, agent_results, contradictions } = job;
  
  resultsTopicTitle.textContent = topic;
  
  // Render Synthesis Narrative
  synthesisBackground.innerHTML = parseMarkdown(result.background);
  synthesisMethodology.innerHTML = parseMarkdown(result.methodology_comparison);
  synthesisFindings.innerHTML = parseMarkdown(result.key_findings);
  synthesisOpenQuestions.innerHTML = parseMarkdown(result.open_questions);

  // Render Contradictions
  badgeContradictions.textContent = contradictions.length;
  contradictionsGrid.innerHTML = '';
  
  if (contradictions.length === 0) {
    contradictionsGrid.innerHTML = `
      <div class="empty-state">
        <i data-lucide="check-circle-2" style="color:var(--status-success)"></i>
        <p>No academic contradictions detected. The reviewed papers report consistent results.</p>
      </div>
    `;
  } else {
    contradictions.forEach(c => {
      const card = document.createElement('div');
      card.className = 'contradiction-card';
      card.innerHTML = `
        <div class="contradiction-card-header">
          <i data-lucide="alert-triangle"></i>
          <span>Conflict on Theme: "${c.topic}"</span>
        </div>
        <div class="conflict-split">
          <div class="conflict-side">
            <h5>Paper ${c.paper_a_key}</h5>
            <p>"${c.paper_a_claim}"</p>
          </div>
          <div class="conflict-side">
            <h5>Paper ${c.paper_b_key}</h5>
            <p>"${c.paper_b_claim}"</p>
          </div>
        </div>
        <div class="conflict-explanation">
          <h5>Editor Explanation & Context</h5>
          <p>${c.explanation}</p>
        </div>
      `;
      contradictionsGrid.appendChild(card);
    });
  }

  // Render Topic Decomposition tree
  resultsClustersGrid.innerHTML = '';
  clusters.forEach(c => {
    const card = document.createElement('div');
    card.className = 'cluster-card';
    
    let questionsHtml = '';
    c.sub_questions.forEach(q => {
      questionsHtml += `
        <div class="cluster-question-item">
          <p>${q.text}</p>
          ${q.rationale ? `<span>Rationale: ${q.rationale}</span>` : ''}
        </div>
      `;
    });
    
    card.innerHTML = `
      <h3>${c.theme}</h3>
      <div class="cluster-questions-list">
        ${questionsHtml}
      </div>
    `;
    resultsClustersGrid.appendChild(card);
  });

  // Render References Bibliography
  badgeReferences.textContent = result.references.length;
  referencesList.innerHTML = '';
  
  result.references.forEach((ref) => {
    // Find all findings associated with this paper key across sub-questions
    const paperFindings = [];
    agent_results.forEach(res => {
      const finding = res.findings.find(f => f.paper.paper_key === ref.paper_key && !f.extraction_failed);
      if (finding) {
        paperFindings.push({
          sub_question: res.sub_question.text,
          finding: finding
        });
      }
    });

    const card = document.createElement('div');
    card.className = 'ref-card';
    
    let findingsDetailsHtml = '';
    if (paperFindings.length > 0) {
      paperFindings.forEach(pf => {
        let claimsHtml = '';
        pf.finding.claims.forEach(c => {
          const confidencePct = Math.round(c.confidence * 100);
          claimsHtml += `
            <li class="claim-item">
              <div class="claim-item-title">"${c.claim}"</div>
              <div class="claim-item-meta">
                <span>Evidence: ${c.evidence || 'unspecified'}</span>
                <div class="confidence-bar">
                  <span>Confidence ${confidencePct}%</span>
                  <div class="confidence-fill">
                    <div class="confidence-value" style="width: ${confidencePct}%"></div>
                  </div>
                </div>
              </div>
            </li>
          `;
        });

        findingsDetailsHtml += `
          <div class="ref-findings-box" style="margin-bottom: 1.5rem; border-bottom: 1px dashed var(--border-color); padding-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: var(--accent-primary); margin-bottom: 0.75rem;">
              SUB-QUESTION: "${pf.sub_question}"
            </div>
            <div class="ref-methodology">
              <h5>Methodology</h5>
              <p>${pf.finding.methodology_summary || 'No methodology summary extracted.'}</p>
            </div>
            ${claimsHtml ? `
            <div class="ref-claims" style="margin-top: 1rem;">
              <h5>Extracted Claims</h5>
              <ul class="claims-sublist">${claimsHtml}</ul>
            </div>
            ` : ''}
            ${pf.finding.limitations ? `
            <div class="ref-limitations" style="margin-top: 1rem;">
              <h5>Limitations & Caveats</h5>
              <p>${pf.finding.limitations}</p>
            </div>
            ` : ''}
          </div>
        `;
      });
    } else {
      findingsDetailsHtml = '<p class="empty-text">No structured findings extracted (abstract-only fallback or extraction issue).</p>';
    }

    const authorsStr = ref.authors && ref.authors.length > 0 ? ref.authors.join(', ') : 'Unknown Authors';
    const publishedYear = ref.published ? ref.published.slice(0, 4) : 'N/A';

    card.innerHTML = `
      <div class="ref-card-main">
        <div class="ref-details">
          <h3>${ref.title}</h3>
          <div class="ref-meta">
            <span><i data-lucide="users" style="width:12px;height:12px"></i> ${authorsStr}</span>
            <span><i data-lucide="calendar" style="width:12px;height:12px"></i> ${publishedYear}</span>
            <span><i data-lucide="link" style="width:12px;height:12px"></i> <a href="${ref.url}" target="_blank">View Paper (${ref.source})</a></span>
          </div>
        </div>
        <i data-lucide="chevron-down" class="ref-toggle-icon"></i>
      </div>
      <div class="ref-card-details">
        ${findingsDetailsHtml}
      </div>
    `;

    // Hook expansion click
    card.querySelector('.ref-card-main').addEventListener('click', () => {
      card.classList.toggle('open');
    });

    referencesList.appendChild(card);
  });

  // Re-run icons render
  lucide.createIcons();
}

// Simple Markdown Parser (paragraphs, bold, bullet lists)
function parseMarkdown(text) {
  if (!text) return '';
  let html = text.trim();
  
  // Escape HTML entities to prevent injection
  html = html
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold (**text** or __text__)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

  // Bullet Lists
  // Look for blocks of lines starting with - or *
  // Replace each list with <ul>...</ul> and elements with <li>...</li>
  const lines = html.split('\n');
  let inList = false;
  const processedLines = [];

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) {
        processedLines.push('<ul>');
        inList = true;
      }
      processedLines.push(`<li>${line.slice(2)}</li>`);
    } else {
      if (inList) {
        processedLines.push('</ul>');
        inList = false;
      }
      processedLines.push(line);
    }
  }
  if (inList) {
    processedLines.push('</ul>');
  }

  // Paragraphs (join non-list, non-empty lines, separated by double newlines)
  html = processedLines.join('\n');
  const paragraphs = html.split(/\n\s*\n/);
  
  return paragraphs.map(p => {
    const trimmed = p.trim();
    if (!trimmed) return '';
    if (trimmed.startsWith('<ul>') || trimmed.startsWith('</ul>') || trimmed.startsWith('<li>')) {
      return trimmed;
    }
    return `<p>${trimmed}</p>`;
  }).join('');
}
