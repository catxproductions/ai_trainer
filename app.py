import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from james import UltimateEngineJamesBot

app = FastAPI()

# Enable CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate James locally as an artificial neural network
james = UltimateEngineJamesBot()

@app.get("/", response_class=HTMLResponse)
async def home_root():
    """Serve the root index.html as the main entry point"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>404: index.html not found</h1>"

@app.post("/chat")
async def process_chat_action(user_message: str = Form(...)):
    """Process user message and generate AI response with neural telemetry"""
    reply_output = james.generate_reply(user_message)
    active_topic = james.current_mood_topic
    telemetry = james.get_telemetry()

    return JSONResponse(content={
        "bot_reply": reply_output,
        "network": {
            "active_node": active_topic,
            "weights": {k: round(v, 2) for k, v in james.context_scores.items()},
            "seed_source": james.last_seed_source,
            "tiers_accessed": james.last_tier_trace
        },
        "telemetry": {
            "turns_learned": telemetry["turns_learned"],
            "learning_rate": telemetry["learning_rate"],
            "error_rate": telemetry["error_rate"],
            "error_trend": telemetry["error_trend"],
            "confidence": telemetry["confidence"],
            "gradient_magnitude": telemetry["gradient_magnitude"],
            "active_synapses": telemetry["active_synapses"],
            "hidden_neurons": telemetry["hidden_neurons"]
        }
    })

@app.get("/telemetry")
async def get_telemetry():
    """Get complete neural telemetry for real-time dashboard updates"""
    telemetry = james.get_telemetry()

    return JSONResponse(content={
        "turns_learned": telemetry["turns_learned"],
        "active_topic": telemetry["active_topic"],
        "learning_rate": telemetry["learning_rate"],
        "error_rate": telemetry["error_rate"],
        "error_trend": telemetry["error_trend"],
        "confidence": telemetry["confidence"],
        "gradient_magnitude": telemetry["gradient_magnitude"],
        "active_synapses": telemetry["active_synapses"],
        "hidden_neurons": telemetry["hidden_neurons"],
        "cumulative_error": telemetry["cumulative_error"],
        "weights": telemetry["weights"]
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "online", "model": "JamesBot v5.0"})
