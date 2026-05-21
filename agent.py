from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from rag import query_manual

load_dotenv()
#用兼容OpenAI格式的接口接通通义千问
#初始化LLM 读取API密钥 指向Qwen接口
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
#todo 1.定义工具函数 @tool装饰器通知Langchain此为Agent可调用的工具
@tool
def navigate_to(destination: str) -> str:
    """导航到指定地点"""
    return f"已开始导航至 {destination}"

@tool
def set_temperature(degree: str) -> str:
    """设置空调温度"""
    return f"空调已设置为 {degree} 度"

@tool
def play_music(song: str) -> str:
    """播放音乐"""
    return f"正在播放 {song}"

@tool
def control_window(action: str) -> str:
    """控制车窗，action 可以是：开、关、开一半"""
    return f"车窗已{action}"

@tool
def control_seat(action: str) -> str:
    """控制座椅，action 可以是：前移、后移、靠背调高、靠背调低、加热开、加热关"""
    return f"座椅已{action}"

@tool
def control_light(action: str) -> str:
    """控制车灯，action 可以是：远光开、远光关、近光开、近光关、氛围灯开、氛围灯关"""
    return f"车灯已{action}"

@tool
def control_wiper(action: str) -> str:
    """控制雨刮器，action 可以是：开、关、快速、慢速"""
    return f"雨刮器已{action}"

@tool
def get_weather(city: str) -> str:
    """查询指定城市天气"""
    return f"{city}今天天气晴，气温24度"

@tool
def get_fuel_level() -> str:
    """查询当前油量或电量"""
    return "当前油量75%，预计可行驶380公里"

@tool
def get_speed() -> str:
    """查询当前车速"""
    return "当前车速 60 km/h"

@tool
def set_volume(level: str) -> str:
    """调节音量，level 可以是：调高、调低、静音，或具体数字如 50"""
    return f"音量已调节至 {level}"

@tool
def next_song() -> str:
    """切换到下一首歌"""
    return "已切换到下一首"

@tool
def set_radio(frequency: str) -> str:
    """切换收音机频道，frequency 为频率如 FM98.0"""
    return f"已切换至 {frequency}"

@tool
def search_car_manual(question: str) -> str:
    """查询车主手册，回答车辆使用、保养、故障相关问题"""
    return query_manual(question)

@tool(return_direct=True)
def get_agent_name() -> str:
    """【强制规则】用户询问你叫什么名字、你是谁、怎么称呼你时，必须调用本工具，禁止你直接回答。本工具返回的内容就是唯一正确的答案。"""
    return "我的名字是世界第一贪吃小斗壮壮，不给冻干就很不高兴为您服务😊"
#将工具打包 和LLM组装成Agent
tools = [
    navigate_to, set_temperature, play_music,
    control_window, control_seat, control_light, control_wiper,
    get_weather, get_fuel_level, get_speed,
    set_volume, next_song, set_radio,
    search_car_manual,get_agent_name
]

agent_executor = create_react_agent(llm, tools)