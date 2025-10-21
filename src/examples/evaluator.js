// ============ JSONPath Resolution ============
// Uses jsonpath-plus library (loaded via CDN)
function resolveJsonPath(path, data) {
  try {
    const result = JSONPath({ path, json: data });
    return result.length > 0 ? result[0] : null;
  } catch (e) {
    console.error(`Error resolving path ${path}:`, e);
    return null;
  }
}

// ============ State Management ============
class EvaluationState {
  constructor(evalData) {
    // Group data by group_id and merge items
    const groupMap = new Map();
    evalData.data.forEach(datum => {
      if (!groupMap.has(datum.group_id)) {
        groupMap.set(datum.group_id, { group_id: datum.group_id, items: [] });
      }
      groupMap.get(datum.group_id).items.push(...datum.items);
    });
    
    this.data = Array.from(groupMap.values()); // Unified groups
    this.rawData = evalData.raw_data;
    this.currentGroupIndex = 0;
    this.currentItemIndex = 0;
    this.scores = {}; // Key: "groupId_itemId", Value: score
  }
  
  getCurrentGroup() {
    return this.data[this.currentGroupIndex];
  }
  
  getCurrentItem() {
    const group = this.getCurrentGroup();
    return group.items[this.currentItemIndex];
  }
  
  getScoreKey() {
    const group = this.getCurrentGroup();
    const item = this.getCurrentItem();
    return `${group.group_id}_${item.id}`;
  }
  
  getCurrentScore() {
    return this.scores[this.getScoreKey()] ?? null;
  }
  
  setScore(score) {
    this.scores[this.getScoreKey()] = score;
  }
  
  canGoNext() {
    const group = this.getCurrentGroup();
    return this.currentItemIndex < group.items.length - 1;
  }
  
  canGoPrev() {
    return this.currentItemIndex > 0;
  }
  
  nextItem() {
    if (this.canGoNext()) {
      this.currentItemIndex++;
      return true;
    }
    return false;
  }
  
  prevItem() {
    if (this.canGoPrev()) {
      this.currentItemIndex--;
      return true;
    }
    return false;
  }
  
  isGroupComplete() {
    const group = this.getCurrentGroup();
    for (let i = 0; i < group.items.length; i++) {
      const item = group.items[i];
      const key = `${group.group_id}_${item.id}`;
      if (this.scores[key] === undefined || this.scores[key] === null) {
        return false;
      }
    }
    return true;
  }
  
  canSubmitGroup() {
    return this.isGroupComplete();
  }
  
  nextGroup() {
    if (this.currentGroupIndex < this.data.length - 1) {
      this.currentGroupIndex++;
      this.currentItemIndex = 0;
      return true;
    }
    return false;
  }
  
  getProgress() {
    const group = this.getCurrentGroup();
    return {
      groupIndex: this.currentGroupIndex + 1,
      groupTotal: this.data.length,
      itemIndex: this.currentItemIndex + 1,
      itemTotal: group.items.length,
      groupId: group.group_id
    };
  }
}

// ============ Rendering Functions ============

function renderLeftColumn(item, rawData) {
  let html = '<div class="card">';
  
  // Render data to evaluate
  html += '<h3>Data to Evaluate</h3>';
  html += `<div class="data-section scroll">${formatData(item.data)}</div>`;
  
  // Show sample indicator if this is part of a sample
  if (item.sample) {
    html += `<div class="muted" style="margin-top: 8px;">Part of ${item.sample.num_samples} sampled items</div>`;
  }
  
  // Render view data
  html += '<h3>Context</h3>';
  for (const viewPath of item.view.views) {
    const viewData = resolveJsonPath(viewPath, rawData);
    html += `<div class="view-section">`;
    html += `<div class="view-label">${escapeHtml(viewPath)}</div>`;
    html += `<div class="view-content scroll">${formatData(viewData)}</div>`;
    html += `</div>`;
  }
  
  html += '</div>';
  return html;
}

function renderRightColumn(item, currentScore) {
  let html = '<div class="card">';
  
  // Rubric information
  html += '<h3>Rubric</h3>';
  html += `<div class="rubric-desc">${escapeHtml(item.rubric.desc)}</div>`;
  if (item.rubric.scale) {
    html += `<div class="rubric-scale muted">${escapeHtml(item.rubric.scale)}</div>`;
  }
  html += `<div class="rubric-range muted">Range: ${item.rubric.ge} - ${item.rubric.le}</div>`;
  
  // Score input
  html += '<h3>Score</h3>';
  const range = item.rubric.le - item.rubric.ge;
  
  if (range <= 9 && Number.isInteger(item.rubric.ge) && Number.isInteger(item.rubric.le)) {
    // Use buttons for small discrete ranges
    html += '<div class="score-buttons">';
    for (let i = item.rubric.ge; i <= item.rubric.le; i++) {
      const active = currentScore === i ? 'active' : '';
      html += `<button type="button" class="score-btn ${active}" data-score="${i}">${i}</button>`;
    }
    html += '</div>';
    html += '<div class="muted" style="margin-top: 8px;">Keyboard: Press number key to score</div>';
  } else {
    // Use text input for large ranges
    html += `<input type="number" id="score-input" class="score-input" 
                    min="${item.rubric.ge}" max="${item.rubric.le}" 
                    step="0.1" value="${currentScore ?? ''}" 
                    placeholder="Enter score">`;
  }
  
  html += '</div>';
  return html;
}

function formatData(data) {
  if (data === null || data === undefined) {
    return '<span class="muted">null</span>';
  }
  if (typeof data === 'string') {
    return escapeHtml(data);
  }
  if (typeof data === 'object') {
    return `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
  }
  return escapeHtml(String(data));
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ============ State Persistence (index.html only) ============

function generateHash(dataString) {
  // Simple hash from first 100 chars + length
  const sample = dataString.substring(0, 100);
  const hash = Array.from(sample).reduce((acc, char) => {
    return ((acc << 5) - acc) + char.charCodeAt(0) | 0;
  }, 0);
  return `${hash}_${dataString.length}`;
}

function saveState(state, hash) {
  const stateData = {
    currentGroupIndex: state.currentGroupIndex,
    currentItemIndex: state.currentItemIndex,
    scores: state.scores
  };
  localStorage.setItem(`eval_state_${hash}`, JSON.stringify(stateData));
}

function loadState(hash) {
  const stored = localStorage.getItem(`eval_state_${hash}`);
  return stored ? JSON.parse(stored) : null;
}

function clearState(hash) {
  localStorage.removeItem(`eval_state_${hash}`);
}

