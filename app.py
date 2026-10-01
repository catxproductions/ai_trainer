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

# Instantiate James locally as an artificial neural network
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
