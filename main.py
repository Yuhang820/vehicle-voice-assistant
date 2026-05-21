from fastapi import FastAPI
from pydantic import BaseModel
from predictor import BertIntentPredictor
from agent import agent_executor
from typing import List, Dict
app = FastAPI()
predictor = BertIntentPredictor()


class Message(BaseModel):
    role: str
    content: str | None = None


class Request(BaseModel):
    text: str
    history: List[Message] = []


@app.post("/predict")
def predict(req: Request):
    intent, confidence = predictor.predict(req.text)

    if confidence > 0.85:
        return {
            "text": req.text,
            "intent": intent,
            "confidence": confidence,
            "source": "model"
        }
    else:
        # 把历史对话拼进 messages
        messages = [{"role": m.role, "content": m.content} for m in req.history if m.content and isinstance(m.content, str)]
        messages.append({"role": "user", "content": req.text})

        result = agent_executor.invoke({"messages": messages})
        return {
            "text": req.text,
            "agent_response": result["messages"][-1].content,
            "source": "agent"
        }

@app.get("/")
def root():
    return {"status": "running"}