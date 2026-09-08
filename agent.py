import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from langchain_core.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("google.genai").setLevel(logging.WARNING)

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def messages_reducer(left: list[Any], right: list[Any]) -> list[Any]:
    return [*left, *right]


# ============================================================
# Domain / conversation models
# ============================================================

class FlowState(str, Enum):
    PRODUCT_DISCOVERY = "PRODUCT_DISCOVERY"
    ORDER = "ORDER"
    SHIPPING = "SHIPPING"
    ORDER_TRACKING = "ORDER_TRACKING"
    RETURN = "RETURN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class Filter(BaseModel):
    color: str | None = None
    size: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    brand: str | None = None


class Variant(BaseModel):
    variant_id: str
    title: str
    available: bool = True


class Product(BaseModel):
    product_id: str
    product_name: str
    price: float
    variants: list[Variant] = Field(default_factory=list)


class ProductDiscoveryInput(BaseModel):
    product_name: str | None = None
    filters: Filter = Field(default_factory=Filter)


class ProductDiscoveryContext(BaseModel):
    input: ProductDiscoveryInput = Field(default_factory=ProductDiscoveryInput)
    tool_results: list[Product] = Field(default_factory=list)
    retrieved_at: datetime | None = None


class OrderContext(BaseModel):
    order_id: str | None = None
    product_id: str | None = None
    variant_id: str | None = None
    quantity: int | None = None


class ShippingContext(BaseModel):
    wilaya: str | None = None
    commune: str | None = None
    address: str | None = None
    shipping_cost: float | None = None


class Flow(BaseModel):
    flow_id: str
    state: FlowState
    created_at: datetime
    updated_at: datetime
    product_discovery: ProductDiscoveryContext | None = None
    order: OrderContext | None = None
    shipping: ShippingContext | None = None


class GlobalInformation(BaseModel):
    customer_name: str | None = None
    wilaya: str | None = None
    commune: str | None = None


class ConversationMemory(BaseModel):
    global_information: GlobalInformation = Field(default_factory=GlobalInformation)
    flows: list[Flow] = Field(default_factory=list)
    active_flow_id: str | None = None


# ============================================================
# Tools (stubs)
# ============================================================

@tool
def searchProducts(query: str) -> str:
    """Search the product catalog."""
    return "searchProducts tool executed (stub)."


@tool
def recallPreviousProducts() -> str:
    """Recall the products the customer recently viewed or searched."""
    return "recallPreviousProducts tool executed (stub)."


@tool
def selectProduct(selector: str) -> str:
    """Select one product from the current results."""
    return "selectProduct tool executed (stub)."


@tool
def getProductDetails(product_id: str) -> str:
    """Get details for a specific product."""
    return "getProductDetails tool executed (stub)."


@tool
def suggestProducts(category: str | None = None) -> str:
    """Suggest products for the customer."""
    return "suggestProducts tool executed (stub)."


@tool
def calculateShipping(wilaya: str, commune: str) -> str:
    """Calculate shipping cost for an address."""
    return "calculateShipping tool executed (stub)."


@tool
def getOrderStatus(order_id: str) -> str:
    """Get the status of an order."""
    return "getOrderStatus tool executed (stub)."


@tool
def createOrder(product_id: str, quantity: int) -> str:
    """Create a new order for a product."""
    return "createOrder tool executed (stub)."


@tool
def confirmOrder(order_id: str) -> str:
    """Confirm a pending order."""
    return "confirmOrder tool executed (stub)."


@tool
def modifyOrder(order_id: str, quantity: int | None = None) -> str:
    """Modify an existing order."""
    return "modifyOrder tool executed (stub)."


@tool
def cancelOrder(order_id: str) -> str:
    """Cancel an order."""
    return "cancelOrder tool executed (stub)."


@tool
def escalateConversation(reason: str) -> str:
    """Escalate the conversation to a human agent."""
    return "escalateConversation tool executed (stub)."


TOOLS = [
    searchProducts,
    recallPreviousProducts,
    selectProduct,
    getProductDetails,
    suggestProducts,
    calculateShipping,
    getOrderStatus,
    createOrder,
    confirmOrder,
    modifyOrder,
    cancelOrder,
    escalateConversation,
]

tool_node = ToolNode(TOOLS)
tool_model = model.bind_tools(TOOLS)


# ============================================================
# Structured output models
# ============================================================

class AgentState(BaseModel):
    messages: Annotated[list[Any], messages_reducer]
    tool_calls: list[dict] = []
    tool_outputs: list[str] = []
    needs_tool: bool = False
    reply: str = ""
    conversation_memory: ConversationMemory = Field(default_factory=ConversationMemory)
    resolved_flow_id: str | None = None
    flow_action: str | None = None


class CheckResult(BaseModel):
    needs_tool: bool = Field(default=False, description="Whether the query requires a tool")
    tool_name: str | None = Field(default=None, description="Name of the tool to use if needed")
    reply: str = Field(default="", description="Direct answer if no tool is needed")


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


class FlowResolution(BaseModel):
    action: Literal["CONTINUE", "CREATE", "NO_FLOW_LOOKUP", "CLARIFY", "INVALID_ACTION"]
    flow_id: str | None = Field(default=None, description="Flow to apply the tool call on (CONTINUE)")
    reason: str | None = Field(default=None, description="Reason for CLARIFY / INVALID_ACTION")
    product_name: str | None = Field(default=None, description="Product name for CREATE")
    filters: Filter | None = Field(default=None, description="Filters for CREATE")


structured_model = model.with_structured_output(CheckResult)
tool_picker = model.with_structured_output(ToolChoice)
flow_resolver_model = model.with_structured_output(FlowResolution)


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


# ============================================================
# Nodes
# ============================================================

def check_llm(state: AgentState) -> dict:
    result = structured_model.invoke(state.messages)
    return {
        "tool_calls": [{"name": result.tool_name}] if (result.needs_tool and result.tool_name) else [],
        "needs_tool": result.needs_tool,
        "reply": result.reply,
    }


def query_tool(state: AgentState) -> dict:
    call = state.tool_calls[0] if state.tool_calls else {}
    if not call:
        return {"tool_outputs": ["No tools provided."]}
    user_text = last_user_text(state)
    choice = tool_picker.invoke([
        HumanMessage(content=json.dumps({"requested_tool": call.get("name"), "user_request": user_text}))
    ])
    result = f"Selected tool: {choice.tool}. Args: {choice.arguments}"
    return {"tool_outputs": [result]}


def flow_resolver(state: AgentState) -> dict:
    mem = state.conversation_memory.model_copy(deep=True)
    flow_summaries = [
        {
            "flow_id": f.flow_id,
            "state": f.state.value if isinstance(f.state, FlowState) else f.state,
            "product": f.product_discovery.input.product_name if f.product_discovery and f.product_discovery.input else None,
            "order": f.order.model_dump(exclude_none=True) if f.order else None,
        }
        for f in mem.flows
    ]
    last_tool = state.tool_outputs[-1] if state.tool_outputs else (state.tool_calls[0].get("name") if state.tool_calls else "unknown")
    context = {
        "active_flow_id": mem.active_flow_id,
        "flows": flow_summaries,
        "selected_tool": last_tool,
        "user_request": last_user_text(state),
    }

    system = (
        "You resolve which conversation flow a tool call should be applied to. Rules:\n"
        "- If the user request does not belong to any flow, return NO_FLOW_LOOKUP.\n"
        "- If there are no flows, or the user is starting a new product search, return CREATE.\n"
        "- If an active flow is compatible with the request, return CONTINUE with its flow_id.\n"
        "- If the request refers to another existing flow, return CONTINUE with that flow_id.\n"
        "- If it is ambiguous which flow applies, return CLARIFY with a reason.\n"
        "- If the action is not valid for any flow, return INVALID_ACTION with a reason."
    )
    result = flow_resolver_model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(context, default=str)),
    ])

    resolved_flow_id = state.resolved_flow_id
    if result.action == "CREATE":
        new_flow_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        tool_name = state.tool_calls[0].get("name", "") if state.tool_calls else ""
        if tool_name in ("searchProducts", "recallPreviousProducts", "selectProduct", "getProductDetails", "suggestProducts"):
            flow_state = FlowState.PRODUCT_DISCOVERY
            product_discovery = ProductDiscoveryContext(input=ProductDiscoveryInput(product_name=result.product_name, filters=result.filters or Filter()))
        else:
            flow_state = FlowState.ORDER
            product_discovery = None
        mem.flows.append(Flow(
            flow_id=new_flow_id,
            state=flow_state,
            created_at=now,
            updated_at=now,
            product_discovery=product_discovery,
        ))
        mem.active_flow_id = new_flow_id
        resolved_flow_id = new_flow_id
    elif result.action == "CONTINUE":
        mem.active_flow_id = result.flow_id
        resolved_flow_id = result.flow_id
    elif result.action in ("NO_FLOW_LOOKUP", "CLARIFY", "INVALID_ACTION"):
        resolved_flow_id = None

    updates: dict[str, Any] = {
        "conversation_memory": mem,
        "resolved_flow_id": resolved_flow_id,
        "flow_action": result.action,
    }
    if result.action in ("NO_FLOW_LOOKUP", "CLARIFY", "INVALID_ACTION"):
        updates["tool_outputs"] = [f"{result.action}: {result.reason or 'Please clarify.'}"]
    if result.action == "CREATE":
        updates["tool_calls"] = []
    logger.info("flow_resolver -> action=%s flow_id=%s", result.action, resolved_flow_id)
    return updates


def calling_tool(state: AgentState) -> dict:
    user_text = last_user_text(state)
    tool_context = state.tool_outputs[-1] if state.tool_outputs else "No tool selected."
    system = (
        "You are an e-commerce assistant. Call the requested tool to fulfill the user's request."
        f"\nResolved flow id: {state.resolved_flow_id or 'none'}. Tool context: {tool_context}"
    )
    ai_msg = tool_model.invoke([SystemMessage(content=system), HumanMessage(content=user_text)])
    tool_messages = tool_node.invoke([ai_msg])
    return {
        "messages": [ai_msg, *tool_messages],
        "tool_outputs": [getattr(m, "content", str(m)) for m in tool_messages],
    }


def reply(state: AgentState) -> dict:
    if not state.needs_tool:
        text = state.reply if state.reply.strip() else "I understood your request."
    else:
        tool_result = state.tool_outputs[-1] if state.tool_outputs else "No tool output available."
        user_text = last_user_text(state)
        response = model.invoke([
            SystemMessage(content="You are a helpful e-commerce assistant. Answer the user's request based only on the tool result provided."),
            HumanMessage(content=f"Tool result:\n{tool_result}\n\nUser request: {user_text}"),
        ])
        text = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=text)]}


# ============================================================
# Routing
# ============================================================

def route(state: AgentState) -> str:
    return "query_tool" if state.needs_tool else "reply"


def flow_route(state: AgentState) -> str:
    if state.flow_action in ("NO_FLOW_LOOKUP", "CLARIFY", "INVALID_ACTION"):
        return "reply"
    return "calling_tool"


graph = StateGraph(AgentState)
graph.add_node("check_llm", check_llm)
graph.add_node("query_tool", query_tool)
graph.add_node("flow_resolver", flow_resolver)
graph.add_node("calling_tool", calling_tool)
graph.add_node("reply", reply)

graph.add_edge(START, "check_llm")
graph.add_conditional_edges("check_llm", route, {"query_tool": "query_tool", "reply": "reply"})
graph.add_edge("query_tool", "flow_resolver")
graph.add_conditional_edges("flow_resolver", flow_route, {"calling_tool": "calling_tool", "reply": "reply"})
graph.add_edge("calling_tool", "reply")
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
            "Turn done -> needs_tool=%s tool=%s flow=%s action=%s reply=%r",
            state["needs_tool"],
            state["tool_calls"],
            state.get("resolved_flow_id"),
            state.get("flow_action"),
            reply_text,
        )
        print("Assistant:", reply_text)

    from IPython.display import Image, display
    display(Image("graph.png"))