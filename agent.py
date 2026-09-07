from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, message_to_dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, Any, Literal, TypedDict
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("google.genai").setLevel(logging.WARNING)

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
    needs_tool: bool = False
    reply: str = ""


class CheckResult(BaseModel):
    needs_tool: bool = Field(default=False, description="Whether the query requires a tool")
    tool_name: str | None = Field(default=None, description="Name of the tool to use if needed")
    reply: str = Field(default="", description="Direct answer if no tool is needed")


structured_model = model.with_structured_output(CheckResult)


AVAILABLE_TOOLS = [
    "searchProducts",
    "recallPreviousProducts",
    "selectProduct",
    "getProductDetails",
    "suggestProducts",
    "calculateShipping",
    "getOrderStatus",
    "createOrder",
    "confirmOrder",
    "modifyOrder",
    "cancelOrder",
    "escalateConversation",
]


class ToolChoice(BaseModel):
    tool: Literal[
        "searchProducts",
        "recallPreviousProducts",
        "selectProduct",
        "getProductDetails",
        "suggestProducts",
        "calculateShipping",
        "getOrderStatus",
        "createOrder",
        "confirmOrder",
        "modifyOrder",
        "cancelOrder",
        "escalateConversation",
    ] = Field(description="The tool to use for the user's request")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


tool_picker = model.with_structured_output(ToolChoice)


def check_llm(state: AgentState) -> dict:
    result = structured_model.invoke(state.messages)
    return {
        "messages": [AIMessage(content=result.model_dump_json())],
        "tool_calls": [{"name": result.tool_name}] if (result.needs_tool and result.tool_name) else [],
        "needs_tool": result.needs_tool,
        "reply": result.reply,
    }


def route(state: AgentState) -> str:
    if state.needs_tool:
        return "query_tool"
    return "reply"


def query_tool(state: AgentState) -> dict:
    call = state.tool_calls[0] if state.tool_calls else {}
    if not call:
        return {"messages": [ToolMessage(content="No tools provided.", tool_call_id="none")], "tool_outputs": ["No tools provided."]}
    mcp_tool = state.messages[-1]
    user_text = mcp_tool.get("content") if isinstance(mcp_tool, dict) else getattr(mcp_tool, "content", None)
    choice = tool_picker.invoke([
        HumanMessage(content=json.dumps({"requested_tool": call.get("name"), "user_request": user_text}))
    ])
    result = f"Selected tool: {choice.tool}. Args: {choice.arguments}"
    return {
        "messages": [ToolMessage(content=result, tool_call_id=call.get("id", "none"))],
        "tool_outputs": [result],
    }


def last_user_text(state: AgentState) -> str:
    for m in reversed(state.messages):
        if isinstance(m, str):
            return m
        if isinstance(m, dict) and m.get("role") == "user":
            text = m.get("content", "")
            return text if isinstance(text, str) else str(text)
        content = getattr(m, "content", "")
        if getattr(m, "type", "") == "human" and content:
            return content if isinstance(content, str) else str(content)
    return ""


def reply(state: AgentState) -> dict:
    if not state.needs_tool:
        text = state.reply if state.reply.strip() else "I understood your request."
    else:
        tool_result = state.tool_outputs[-1] if state.tool_outputs else "No tools provided."
        user_text = last_user_text(state)
        response = model.invoke([
            SystemMessage(content="You are a helpful e-commerce assistant. Answer the user's request based only on the tool result provided."),
            HumanMessage(content=f"Tool result:\n{tool_result}\n\nUser request: {user_text}"),
        ])
        text = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=text)]}


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
        if isinstance(state["messages"][-1], str):
            reply_text = state["messages"][-1]
        else:
            last = state["messages"][-1]
            reply_text = last.content if hasattr(last, "content") else last.get("content", last)
        logger.info(
            "Turn done -> needs_tool=%s tools=%s reply=%r",
            state["needs_tool"],
            state["tool_calls"],
            reply_text,
        )
        print("Assistant:", reply_text)

    from IPython.display import Image, display
    display(Image("graph.png"))
    
    
    

