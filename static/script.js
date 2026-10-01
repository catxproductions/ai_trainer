/**
 * JamesBot AI Control Room Dashboard
 * Real-time neural network visualization and telemetry client
 * Communicates with FastAPI backend for live model introspection
 */

// ============================================================================
// CONFIGURATION & STATE
// ============================================================================

const CONFIG = {
    apiUrl: window.location.hostname === 'localhost' ? 'http://localhost:8000' : '',
    telemetryInterval: 500,
    maxHistoryLength: 50,
    canvasWidth: 550,
    canvasHeight: 100
};

const state = {
    isConnected: false,
    lastTelemetry: null,
    confidenceHistory: [],
    errorHistory: [],
    conversationHistory: [],
    telemetryIntervalId: null,
    isProcessing: false,
    messageCount: 0
};

// ============================================================================
// DOM ELEMENT CACHING
// ============================================================================

const DOM = {
    // Chat interface
    chatForm: document.getElementById('chatForm'),
    userInput: document.getElementById('userInput'),
    sendBtn: document.getElementById('sendBtn'),
    chatWindow: document.getElementById('chatWindow'),

    // Status indicator
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),

    // Activation nodes and visualization
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

    // Analytics and graphs
    analyticsBox: document.getElementById('analyticsBox'),
    synapticCanvas: document.getElementById('synapticCanvas'),
    learningCurve: document.getElementById('learningCurve')
};

// ============================================================================
// CONNECTION MANAGEMENT
// ============================================================================

async function checkConnection() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok || response.status === 404) {
            setConnectionStatus(true);
            return true;
        }
    } catch (error) {
        setConnectionStatus(false);
        return false;
    }
}

function setConnectionStatus(connected) {
    state.isConnected = connected;

    if (connected) {
        DOM.statusDot.classList.add('connected');
        DOM.statusText.textContent = '🟢 Connected to Neural Engine';
        DOM.sendBtn.disabled = false;
    } else {
        DOM.statusDot.classList.remove('connected');
        DOM.statusText.textContent = '🔴 Backend Unreachable';
        DOM.sendBtn.disabled = true;
    }
}

// ============================================================================
// CHAT FUNCTIONALITY
// ============================================================================

async function sendMessage(event) {
    if (event) event.preventDefault();

    const message = DOM.userInput.value.trim();
    if (!message) return;

    if (!state.isConnected) {
        addMessageToChat(
            '⚠️ Backend not connected. Start: uvicorn app:app --reload',
            'error'
        );
        return;
    }

    // Add user message
    addMessageToChat(`You: ${message}`, 'user');
    DOM.userInput.value = '';
    DOM.userInput.focus();

    state.isProcessing = true;
    DOM.sendBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('user_message', message);

        const response = await fetch(`${CONFIG.apiUrl}/chat`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Add bot response
        addMessageToChat(`James: ${data.bot_reply}`, 'bot');

        // Update neural visualization
        if (data.network) {
            updateNeuralVisualization(data.network);
        }

        // Store conversation history
        state.conversationHistory.push({
            user: message,
            bot: data.bot_reply,
            timestamp: new Date(),
            network: data.network
        });

        state.messageCount++;

    } catch (error) {
        addMessageToChat(`❌ Error: ${error.message}`, 'error');
    } finally {
        state.isProcessing = false;
        DOM.sendBtn.disabled = false;
    }
}

function addMessageToChat(text, role = 'bot') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `msg ${role}`;
    messageDiv.textContent = text;
    DOM.chatWindow.appendChild(messageDiv);
    DOM.chatWindow.scrollTop = DOM.chatWindow.scrollHeight;
}

// ============================================================================
// NEURAL VISUALIZATION
// ============================================================================

function updateNeuralVisualization(network) {
    const weights = network.weights || {};
    const activeTopic = network.active_node || 'idle';
    const topics = ['greetings', 'tech', 'gaming', 'school'];

    // Update each node
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

    // Update diagnostics
    const tiers = network.tiers_accessed || {};
    DOM.analyticsBox.innerHTML = `
        <p><b>Inference Method:</b> <code>${network.seed_source || 'Neural Network'}</code></p>
        <p><b>Architecture:</b> Input(${28}) → Hidden(128) → Output(4)</p>
        <p><b>Neurons Active:</b> <code>${tiers.Layer_1_Neurons || 128}</code> | 
           <b>Synapses:</b> <code>${tiers.Active_Synapses || 0}</code> | 
           <b>Confidence:</b> <code>${(tiers.Output_Confidence || 0).toFixed(4)}</code></p>
    `;

    // Store for history
    if (!state.lastTelemetry) {
        state.lastTelemetry = {};
    }

    state.lastTelemetry.weights = weights;
    state.lastTelemetry.confidence = tiers.Output_Confidence || 0;

    state.confidenceHistory.push(state.lastTelemetry.confidence);
    if (state.confidenceHistory.length > CONFIG.maxHistoryLength) {
        state.confidenceHistory.shift();
    }

    // Redraw graphs
    drawLearningCurve();
    drawSynapticMap();
}

// ============================================================================
// TELEMETRY POLLING
// ============================================================================

async function pollTelemetry() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/telemetry`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) return;

        const telemetry = await response.json();
        updateTelemetryDisplay(telemetry);

    } catch (error) {
        // Silently fail if telemetry endpoint unavailable
        return;
    }
}

function updateTelemetryDisplay(telemetry) {
    if (!telemetry) return;

    // Update text values
    if (DOM.telemetry.topic && telemetry.active_topic) {
        DOM.telemetry.topic.textContent = telemetry.active_topic.toUpperCase();
    }

    if (DOM.telemetry.turns !== undefined) {
        DOM.telemetry.turns.textContent = telemetry.turns_learned || 0;
    }

    if (DOM.telemetry.lr !== undefined) {
        DOM.telemetry.lr.textContent = (telemetry.learning_rate || 0.01).toFixed(4);
    }

    if (DOM.telemetry.error !== undefined) {
        DOM.telemetry.error.textContent = (telemetry.error_rate || 0).toFixed(4);
        state.errorHistory.push(telemetry.error_rate || 0);
        if (state.errorHistory.length > CONFIG.maxHistoryLength) {
            state.errorHistory.shift();
        }
    }

    if (DOM.telemetry.confidence !== undefined) {
        DOM.telemetry.confidence.textContent = (telemetry.confidence || 0).toFixed(4);
    }

    if (DOM.telemetry.gradient !== undefined) {
        DOM.telemetry.gradient.textContent = (telemetry.gradient_magnitude || 0).toFixed(4);
    }

    if (DOM.telemetry.synapses !== undefined) {
        DOM.telemetry.synapses.textContent = telemetry.active_synapses || 0;
    }

    if (DOM.telemetry.neurons !== undefined) {
        DOM.telemetry.neurons.textContent = telemetry.hidden_neurons || 128;
    }

    // Update error trend
    if (telemetry.error_trend !== undefined && DOM.telemetry.trendBar) {
        const trend = parseFloat(telemetry.error_trend) || 0;
        const trendPercent = Math.min(100, Math.max(0, 50 + trend * 50));

        if (trend > 0.01) {
            // Error increasing (red)
            DOM.telemetry.trendBar.style.width = trendPercent + '%';
            DOM.telemetry.trendBar.style.background = 'linear-gradient(90deg, #ef4444, #ff6b6b)';
        } else {
            // Error decreasing (green)
            DOM.telemetry.trendBar.style.width = trendPercent + '%';
            DOM.telemetry.trendBar.style.background = 'linear-gradient(90deg, #10b981, #00ff88)';
        }
    }

    state.lastTelemetry = telemetry;
}

// ============================================================================
// CANVAS VISUALIZATION
// ============================================================================

function drawLearningCurve() {
    const canvas = DOM.learningCurve;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 15;
    const graphWidth = width - padding * 2;
    const graphHeight = height - padding * 2;

    // Clear
    ctx.fillStyle = '#010102';
    ctx.fillRect(0, 0, width, height);

    // Border
    ctx.strokeStyle = '#1e1e24';
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, graphWidth, graphHeight);

    if (state.confidenceHistory.length < 2) return;

    // Grid lines
    ctx.strokeStyle = 'rgba(30, 30, 36, 0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = padding + (graphHeight / 4) * i;
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

    // Draw points
    ctx.fillStyle = '#00ff88';
    state.confidenceHistory.forEach((confidence, index) => {
        const x = padding + (graphWidth / (state.confidenceHistory.length - 1)) * index;
        const y = height - padding - (confidence * graphHeight);
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
    });

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 4; i++) {
        const value = (i / 4).toFixed(1);
        const y = height - padding - (graphHeight / 4) * i;
        ctx.fillText(value, padding - 5, y + 3);
    }
}

function drawSynapticMap() {
    const canvas = DOM.synapticCanvas;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = '#010102';
    ctx.fillRect(0, 0, width, height);

    // Layer positions and sizes
    const layers = [
        { x: 30, name: 'Input\n(28)', size: 4, color: '#60a5fa' },
        { x: 140, name: 'Hidden\n(128)', size: 8, color: '#10b981' },
        { x: 280, name: 'Hidden\n(128)', size: 8, color: '#10b981' },
        { x: 420, name: 'Output\n(4)', size: 4, color: '#f43f5e' }
    ];

    // Draw layers and connections
    layers.forEach((layer, layerIdx) => {
        const spacing = (height - 20) / Math.max(1, layer.size - 1);

        // Draw neurons
        for (let i = 0; i < layer.size; i++) {
            const y = 10 + (layer.size === 1 ? height / 2 - 10 : i * spacing);
            ctx.fillStyle = layer.color;
            ctx.beginPath();
            ctx.arc(layer.x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw connections to next layer
        if (layerIdx < layers.length - 1) {
            const nextLayer = layers[layerIdx + 1];
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.15)';
            ctx.lineWidth = 0.8;

            for (let i = 0; i < layer.size; i++) {
                const y1 = 10 + (layer.size === 1 ? height / 2 - 10 : i * spacing);
                for (let j = 0; j < nextLayer.size; j++) {
                    const y2 = 10 + (nextLayer.size === 1 ? height / 2 - 10 : j * ((height - 20) / (nextLayer.size - 1)));
                    ctx.beginPath();
                    ctx.moveTo(layer.x, y1);
                    ctx.lineTo(nextLayer.x, y2);
                    ctx.stroke();
                }
            }
        }

        // Draw layer labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        layer.name.split('\n').forEach((line, i) => {
            ctx.fillText(line, layer.x, height - 10 + i * 10);
        });
    });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

async function initializeDashboard() {
    const connected = await checkConnection();

    if (connected) {
        addMessageToChat('✅ AI Backbone Online. Neural matrix ready.', 'bot');
        
        // Start telemetry polling
        if (state.telemetryIntervalId) {
            clearInterval(state.telemetryIntervalId);
        }
        state.telemetryIntervalId = setInterval(pollTelemetry, CONFIG.telemetryInterval);

        // Initial canvas draws
        drawLearningCurve();
        drawSynapticMap();
    } else {
        addMessageToChat(
            '⚠️ Backend Offline. Run: uvicorn app:app --reload',
            'error'
        );
    }
}

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
