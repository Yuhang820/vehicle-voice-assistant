import gradio as gr
import requests


def predict(text, history):
    # 把历史对话格式化传给后端
    messages = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = content[0].get("text", "") if content else ""
        if content and isinstance(content, str):
            messages.append({"role": role, "content": content})

    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": text, "history": messages}
    )
    result = response.json()

    source = result.get("source", "agent")
    if source == "model":
        reply = f"意图：{result.get('intent', '')}（置信度：{result.get('confidence', '')}）"
    else:
        reply = result.get("agent_response", str(result))

    return reply


demo = gr.ChatInterface(
    fn=predict,
    title="欢迎使用宇航车载语音人工智障",
    description="输入自然语言指令，支持多轮对话",
    examples=["导航到大同南站", "我有点冷", "你叫什么名字", "播放王力宏"]
)

if __name__ == "__main__":
    demo.launch()