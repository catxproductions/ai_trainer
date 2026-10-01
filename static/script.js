document.getElementById('chatForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const inputArea = document.getElementById('userInput');
    const chatWindow = document.getElementById('chatWindow');
    const analyticsBox = document.getElementById('analyticsBox');
    const messageText = inputArea.value.trim();

    if (!messageText) return;

    chatWindow.innerHTML += `<div class="msg user"><b>You:</b> ${messageText}</div>`;
    inputArea.value = '';
    chatWindow.scrollTop = chatWindow.scrollHeight;

    const dataPayload = new FormData();
    dataPayload.append('user_message', messageText);

    try {
        const response = await fetch('/chat', { method: 'POST', body: dataPayload });
        const data = await response.json();

        chatWindow.innerHTML += `<div class="msg bot"><b>James:</b> ${data.bot_reply}</div>`;
        chatWindow.scrollTop = chatWindow.scrollHeight;

        const topics = ['greetings', 'tech', 'gaming', 'school'];
        topics.forEach(t => {
            const targetElement = document.getElementById(`node-${t}`);
            const currentWeight = data.network.weights[t] || 0;
            
            targetElement.querySelector('.weight').innerText = `Probability: ${currentWeight}`;
            
            if (t === data.network.active_node) {
                targetElement.className = "node active-winner";
            } else {
                targetElement.className = "node";
            }
        });

        const metrics = data.network.tiers_accessed;
        analyticsBox.innerHTML = `
            <p><b>Inference Method:</b> <code>${data.network.seed_source}</code></p>
            <p><b>Layer Telemetry:</b> Hidden Neurons: <code>${metrics.Layer_1_Neurons}</code> | Active Synapses: <code>${metrics.Active_Synapses}</code> | Confidence: <code>${metrics.Output_Confidence}</code></p>
        `;

    } catch (err) {
        chatWindow.innerHTML += `<div class="msg error"><b>System Alert:</b> Matrix transmission lost.</div>`;
    }
});
