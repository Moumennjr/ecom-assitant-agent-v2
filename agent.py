from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, message_to_dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, Any, TypedDict
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

def messages_reducer(left: list[Any], right: list[Any]) -> list[Any]:
    return [*left, *right]


class AgentState(BaseModel):
    messages: Annotated[list[Any], messages_reducer]
    tool_calls: list[dict] = []
    tool_outputs: list[str] = []


def check_llm(state: AgentState) -> dict:
    response = model.invoke(state.messages)
    return {
        "messages": [response],
        "tool_calls": list(response.tool_calls or []),
    }


def route(state: AgentState) -> str:
    if state.tool_calls:
        return "query_tool"
    return "reply"


def query_tool(state: AgentState) -> dict:
    call = state.tool_calls[0] if state.tool_calls else {}
    result = "No tools provided."
    return {
        "messages": [ToolMessage(content=result, tool_call_id=call.get("id", "none"))],
        "tool_outputs": [result],
    }


def reply(state: AgentState) -> dict:
    response = model.invoke(state.messages)
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("check_llm", check_llm)
graph.add_node("query_tool", query_tool)
graph.add_node("reply", reply)

graph.add_edge(START, "check_llm")
graph.add_conditional_edges(
    "check_llm",
    route,
    {"query_tool": "query_tool", "reply": "reply"},
)
graph.add_edge("query_tool", "reply")
graph.add_edge("reply", END)
app = graph.compile(checkpointer=InMemorySaver())


def to_serializable(value: Any) -> Any:
    if isinstance(value, list):
        return [to_serializable(m) for m in value]
    if hasattr(value, "type"):
        return message_to_dict(value)
    return value


def log_state(state: Any) -> None:
    data = state if isinstance(state, dict) else state.model_dump()
    payload = {k: to_serializable(v) for k, v in data.items()}
    logger.info("Final AgentState:\n%s", json.dumps(payload, indent=2, default=str))


def save_graph_png(path: str = "graph.png") -> str:
    try:
        with open(path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        logger.info("Graph saved to %s", path)
        return path
    except Exception as e:
        logger.warning("PNG render failed (%s); falling back to mermaid source", e)
        with open("graph.mmd", "w") as f:
            f.write(app.get_graph().draw_mermaid())
        print(app.get_graph().draw_ascii())
        logger.info("Saved mermaid source to graph.mmd")
        return "graph.mmd"


if __name__ == "__main__":
    save_graph_png()
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        state = result.model_dump() if hasattr(result, "model_dump") else result
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else last.get("content", last)
        print("Assistant:", content)
        log_state(state)

    from IPython.display import Image, display
    display(Image("graph.png"))
    
    
    

