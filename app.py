import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from james import UltimateEngineJamesBot

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

james = UltimateEngineJamesBot()

@app.get("/", response_class=HTMLResponse)
async def home_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def process_chat_action(user_message: str = Form(...)):
    reply_output = james.generate_reply(user_message)
    active_topic = james.current_mood_topic

    return JSONResponse(content={
        "bot_reply": reply_output,
        "network": {
            "active_node": active_topic,
            "weights": {k: round(v, 2) for k, v in james.context_scores.items()},
            "seed_source": james.last_seed_source,
            "tiers_accessed": james.last_tier_trace
        }
    })

@app.get("/telemetry")
async def get_telemetry():
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
