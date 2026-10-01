/**
 * JamesBot Dashboard Client v4.0
 * Browser-based interface for FastAPI backend neural network
 * Handles chat, telemetry polling, and real-time visualization
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
  apiUrl: typeof window !== 'undefined' && window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '', // Use relative path for GitHub Pages
  telemetryInterval: 500, // Poll every 500ms
  maxHistoryLength: 50,
  canvasWidth: 600,
  canvasHeight: 120
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const state = {
  isConnected: false,
  lastTelemetry: null,
  errorHistory: [],
  confidenceHistory: [],
  conversationHistory: [],
  telemetryIntervalId: null,
  isProcessing: false
};

// ============================================================================
// DOM ELEMENT CACHING
// ============================================================================

const DOM = {
  // Chat elements
  chatForm: document.getElementById('chatForm'),
  userInput: document.getElementById('userInput'),
  sendBtn: document.getElementById('sendBtn'),
  chatWindow: document.getElementById('chatWindow'),

  // Status indicator
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),

  // Activation nodes
  nodes: {
    greetings: document.getElementById('node-greetings'),
    tech: document.getElementById('node-tech'),
    gaming: document.getElementById('node-gaming'),
    school: document.getElementById('node-school')
  },
  weights: {
    greetings: document.getElementById('weight-greetings'),
    tech: document.getElementById('weight-tech'),
    gaming: document.getElementById('weight-gaming'),
    school: document.getElementById('weight-school')
  },
  fills: {
    greetings: document.getElementById('greetings-fill'),
    tech: document.getElementById('tech-fill'),
    gaming: document.getElementById('gaming-fill'),
    school: document.getElementById('school-fill')
  },

  // Telemetry displays
  telemetry: {
    topic: document.getElementById('telemetry-topic'),
    turns: document.getElementById('telemetry-turns'),
    lr: document.getElementById('telemetry-lr'),
    error: document.getElementById('telemetry-error'),
    confidence: document.getElementById('telemetry-confidence'),
    gradient: document.getElementById('telemetry-gradient'),
    synapses: document.getElementById('telemetry-synapses'),
    neurons: document.getElementById('telemetry-neurons'),
    trendBar: document.getElementById('trendBar')
  },

  // Analytics card
  analyticsBox: document.getElementById('analyticsBox'),

  // Canvas elements
  synapticCanvas: document.getElementById('synapticCanvas'),
  learningCurve: document.getElementById('learningCurve')
};

// ============================================================================
// CONNECTION MANAGEMENT
// ============================================================================

/**
 * Check if backend is reachable
 */
async function checkConnection() {
  try {
    const response = await fetch(`${CONFIG.apiUrl}/`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (response.ok || response.status === 404) {
      // 404 is fine - means server is running but endpoint doesn't exist
      setConnectionStatus(true);
      return true;
    }
  } catch (error) {
    setConnectionStatus(false);
    console.warn('Backend connection check failed:', error.message);
    return false;
  }
}

/**
 * Update connection status UI
 */
function setConnectionStatus(connected) {
  state.isConnected = connected;

  if (connected) {
    DOM.statusDot.classList.add('connected');
    DOM.statusText.textContent = '🟢 Connected to Neural Engine';
    DOM.sendBtn.disabled = false;
  } else {
    DOM.statusDot.classList.remove('connected');
    DOM.statusText.textContent = '🔴 Backend Unreachable (uvicorn app:app --reload)';
    DOM.sendBtn.disabled = true;
  }
}

// ============================================================================
// CHAT FUNCTIONALITY
// ============================================================================

/**
 * Send message to FastAPI backend
 */
async function sendMessage(event) {
  if (event) event.preventDefault();

  const message = DOM.userInput.value.trim();
  if (!message) return;

  if (!state.isConnected) {
    addMessageToChat(
      '⚠️ Error: Backend not connected. Start the FastAPI server: uvicorn app:app --reload',
      'error'
    );
    return;
  }

  // Add user message to chat
  addMessageToChat(message, 'user');
  DOM.userInput.value = '';
  DOM.userInput.focus();

  // Disable input during processing
  state.isProcessing = true;
  DOM.sendBtn.disabled = true;

  try {
    // Create FormData for multipart/form-data submission (matching FastAPI endpoint)
    const formData = new FormData();
    formData.append('user_message', message);

    // Send to /chat endpoint
    const response = await fetch(`${CONFIG.apiUrl}/chat`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    // Add bot response to chat
    addMessageToChat(data.bot_reply, 'bot');

    // Update neural visualization
    if (data.network) {
      updateNeuralVisualization(data.network);
    }

    // Store in history
    state.conversationHistory.push({
      user: message,
      bot: data.bot_reply,
      timestamp: new Date(),
      network: data.network
    });

  } catch (error) {
    addMessageToChat(
      `❌ Error: ${error.message}. Make sure backend is running.`,
      'error'
    );
    console.error('Chat request failed:', error);
  } finally {
    state.isProcessing = false;
    DOM.sendBtn.disabled = false;
  }
}

/**
 * Add message to chat window
 */
function addMessageToChat(text, role = 'bot') {
  const messageDiv = document.createElement('div');
  messageDiv.className = `msg ${role}`;

  if (role === 'user') {
    messageDiv.innerHTML = `<b>You:</b> ${escapeHtml(text)}`;
  } else if (role === 'bot') {
    messageDiv.innerHTML = `<b>James:</b> ${escapeHtml(text)}`;
  } else if (role === 'error') {
    messageDiv.innerHTML = `<b>System Alert:</b> ${escapeHtml(text)}`;
  }

  DOM.chatWindow.appendChild(messageDiv);
  DOM.chatWindow.scrollTop = DOM.chatWindow.scrollHeight;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, (char) => map[char]);
}

// ============================================================================
// NEURAL VISUALIZATION
// ============================================================================

/**
 * Update activation nodes and weights display
 */
function updateNeuralVisualization(network) {
  const weights = network.weights || {};
  const activeTopic = network.active_node || 'idle';

  // Update each node's activation
  const topics = ['greetings', 'tech', 'gaming', 'school'];

  topics.forEach((topic) => {
    const weight = weights[topic] || 0;
    const percentage = Math.min(100, Math.max(0, weight * 100));

    // Update weight display
    if (DOM.weights[topic]) {
      DOM.weights[topic].textContent = weight.toFixed(2);
    }

    // Update activation bar
    if (DOM.fills[topic]) {
      DOM.fills[topic].style.width = percentage + '%';
    }

    // Highlight active node
    if (DOM.nodes[topic]) {
      if (topic === activeTopic) {
        DOM.nodes[topic].classList.add('active-winner');
      } else {
        DOM.nodes[topic].classList.remove('active-winner');
      }
    }
  });

  // Update diagnostics card
  const tiers = network.tiers_accessed || {};
  const diagnosticsHTML = `
    <p><b>Inference Method:</b> <code>${escapeHtml(network.seed_source || 'Neural Network Inference')}</code></p>
    <p><b>Layer Telemetry:</b> 
      Hidden Neurons: <code>${tiers.Layer_1_Neurons || 8}</code> | 
      Active Synapses: <code>${tiers.Active_Synapses || 0}</code> | 
      Confidence: <code>${(tiers.Output_Confidence || 0).toFixed(4)}</code>
    </p>
  `;
  DOM.analyticsBox.innerHTML = diagnosticsHTML;

  // Store telemetry for history
  if (!state.lastTelemetry) {
    state.lastTelemetry = {};
  }

  state.lastTelemetry.weights = weights;
  state.lastTelemetry.activeNode = activeTopic;
  state.lastTelemetry.confidence = tiers.Output_Confidence || 0;
  state.lastTelemetry.activeSynapses = tiers.Active_Synapses || 0;

  // Track confidence for graph
  state.confidenceHistory.push(state.lastTelemetry.confidence);
  if (state.confidenceHistory.length > CONFIG.maxHistoryLength) {
    state.confidenceHistory.shift();
  }

  // Redraw learning curve
  drawLearningCurve();
}

// ============================================================================
// TELEMETRY POLLING
// ============================================================================

/**
 * Poll backend for telemetry data
 */
async function pollTelemetry() {
  try {
    const response = await fetch(`${CONFIG.apiUrl}/telemetry`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      // Endpoint may not exist in older backend versions
      return;
    }

    const telemetry = await response.json();
    updateTelemetryDisplay(telemetry);

  } catch (error) {
    // Silently fail if telemetry endpoint doesn't exist
    // This is optional and won't break chat functionality
    if (error.message.includes('404')) {
      return;
    }
    console.debug('Telemetry polling skipped:', error.message);
  }
}

/**
 * Update telemetry displays
 */
function updateTelemetryDisplay(telemetry) {
  if (!telemetry) return;

  // Update text displays
  if (DOM.telemetry.topic && telemetry.active_topic) {
    DOM.telemetry.topic.textContent = telemetry.active_topic.toUpperCase();
  }

  if (DOM.telemetry.turns && telemetry.turns_learned !== undefined) {
    DOM.telemetry.turns.textContent = telemetry.turns_learned;
  }

  if (DOM.telemetry.lr && telemetry.learning_rate !== undefined) {
    DOM.telemetry.lr.textContent = parseFloat(telemetry.learning_rate).toFixed(4);
  }

  if (DOM.telemetry.error && telemetry.error_rate !== undefined) {
    DOM.telemetry.error.textContent = parseFloat(telemetry.error_rate).toFixed(4);
    
    // Track error for history
    state.errorHistory.push(parseFloat(telemetry.error_rate));
    if (state.errorHistory.length > CONFIG.maxHistoryLength) {
      state.errorHistory.shift();
    }
  }

  if (DOM.telemetry.confidence && telemetry.confidence !== undefined) {
    DOM.telemetry.confidence.textContent = parseFloat(telemetry.confidence).toFixed(4);
  }

  if (DOM.telemetry.gradient && telemetry.gradient_magnitude !== undefined) {
    DOM.telemetry.gradient.textContent = parseFloat(telemetry.gradient_magnitude).toFixed(4);
  }

  if (DOM.telemetry.synapses && telemetry.active_synapses !== undefined) {
    DOM.telemetry.synapses.textContent = telemetry.active_synapses;
  }

  if (DOM.telemetry.neurons && telemetry.hidden_neurons !== undefined) {
    DOM.telemetry.neurons.textContent = telemetry.hidden_neurons;
  }

  // Update error trend visualization
  if (telemetry.error_trend !== undefined && DOM.telemetry.trendBar) {
    const trend = parseFloat(telemetry.error_trend);
    const trendPercent = Math.min(100, Math.max(-100, trend * 10)); // Scale for visibility
    
    if (trendPercent > 0) {
      DOM.telemetry.trendBar.style.width = '50%';
      DOM.telemetry.trendBar.style.backgroundColor = '#ef4444'; // Red = increasing error
    } else {
      DOM.telemetry.trendBar.style.width = (50 - (Math.abs(trendPercent) / 2)) + '%';
      DOM.telemetry.trendBar.style.backgroundColor = '#10b981'; // Green = decreasing error
    }
  }

  state.lastTelemetry = telemetry;
}

// ============================================================================
// CANVAS VISUALIZATION
// ============================================================================

/**
 * Draw learning curve (confidence over time)
 */
function drawLearningCurve() {
  const canvas = DOM.learningCurve;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const padding = 20;
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2;

  // Clear canvas
  ctx.fillStyle = '#010102';
  ctx.fillRect(0, 0, width, height);

  // Draw border
  ctx.strokeStyle = '#1e1e24';
  ctx.lineWidth = 1;
  ctx.strokeRect(padding, padding, graphWidth, graphHeight);

  if (state.confidenceHistory.length < 2) return;

  // Draw grid lines
  ctx.strokeStyle = '#1e1e24';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = padding + (graphHeight / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }

  // Draw confidence line
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2;
  ctx.beginPath();

  state.confidenceHistory.forEach((confidence, index) => {
    const x = padding + (graphWidth / (state.confidenceHistory.length - 1)) * index;
    const y = height - padding - (confidence * graphHeight);

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();

  // Draw axis labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';

  for (let i = 0; i <= 5; i++) {
    const value = (i / 5).toFixed(1);
    const y = height - padding - (graphHeight / 5) * i;
    ctx.fillText(value, padding - 5, y + 4);
  }

  ctx.textAlign = 'center';
  ctx.fillText('Confidence Score Over Time', width / 2, height - 5);
}

/**
 * Draw synaptic connection map (simplified)
 */
function drawSynapticMap() {
  const canvas = DOM.synapticCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.fillStyle = '#010102';
  ctx.fillRect(0, 0, width, height);

  // Draw layers
  const layerX = [50, 200, 350, 500];
  const layerNames = ['Input', 'Hidden', 'Hidden', 'Output'];
  const layerSizes = [4, 8, 8, 4];

  layerX.forEach((x, layerIdx) => {
    const size = layerSizes[layerIdx];
    const spacing = (height - 40) / size;

    // Draw neurons
    for (let i = 0; i < size; i++) {
      const y = 20 + i * spacing;
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Layer label (only on first layer)
      if (i === Math.floor(size / 2)) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(layerNames[layerIdx], x, height - 5);
      }
    }

    // Draw connections to next layer
    if (layerIdx < layerX.length - 1) {
      const nextX = layerX[layerIdx + 1];
      const nextSize = layerSizes[layerIdx + 1];
      const nextSpacing = (height - 40) / nextSize;

      ctx.strokeStyle = 'rgba(16, 185, 129, 0.2)';
      ctx.lineWidth = 0.5;

      for (let i = 0; i < size; i++) {
        const y = 20 + i * spacing;
        for (let j = 0; j < nextSize; j++) {
          const nextY = 20 + j * nextSpacing;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(nextX, nextY);
          ctx.stroke();
        }
      }
    }
  });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize dashboard
 */
async function initializeDashboard() {
  // Check backend connection
  const connected = await checkConnection();

  if (connected) {
    addMessageToChat(
      '✅ AI Backbone Online. Matrix layer operations running on Python backend.',
      'bot'
    );
    
    // Start telemetry polling
    if (state.telemetryIntervalId) {
      clearInterval(state.telemetryIntervalId);
    }
    state.telemetryIntervalId = setInterval(pollTelemetry, CONFIG.telemetryInterval);

    // Draw initial canvas
    drawLearningCurve();
    drawSynapticMap();
  } else {
    addMessageToChat(
      '⚠️ Backend not detected. Start the FastAPI server:\n' +
      'pip install numpy fastapi uvicorn\n' +
      'uvicorn app:app --reload',
      'error'
    );
  }
}

/**
 * Periodic connection check
 */
function startConnectionMonitoring() {
  setInterval(async () => {
    await checkConnection();
  }, 5000);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

if (DOM.chatForm) {
  DOM.chatForm.addEventListener('submit', sendMessage);
}

if (DOM.userInput) {
  DOM.userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// ============================================================================
// STARTUP
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  initializeDashboard();
  startConnectionMonitoring();
});