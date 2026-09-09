import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from rich import print
from langchain_core.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

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
    selected_product: Product | None = None
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


class ToolCallDraft(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    prereqs: dict[str, Any] = Field(default_factory=dict)
    status: Literal["drafting", "ready", "executed", "cancelled"] = "drafting"
    attempts: int = 0
    missing: list[str] = Field(default_factory=list)


class Flow(BaseModel):
    flow_id: str
    state: FlowState
    created_at: datetime
    updated_at: datetime
    product_discovery: ProductDiscoveryContext | None = None
    order: OrderContext | None = None
    shipping: ShippingContext | None = None
    tool_draft: ToolCallDraft | None = None


class GlobalInformation(BaseModel):
    customer_name: str | None = None
    wilaya: str | None = None
    commune: str | None = None
    address: str | None = None


class ConversationMemory(BaseModel):
    global_information: GlobalInformation = Field(default_factory=GlobalInformation)
    flows: list[Flow] = Field(default_factory=list)
    active_flow_id: str | None = None


# ============================================================
# Tools
# ============================================================

PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "data", "products.json")
_PRODUCTS_CACHE: list[dict[str, Any]] | None = None


def load_products() -> list[dict[str, Any]]:
    global _PRODUCTS_CACHE
    if _PRODUCTS_CACHE is None:
        try:
            with open(PRODUCTS_PATH, encoding="utf-8") as f:
                _PRODUCTS_CACHE = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load products from %s: %s", PRODUCTS_PATH, e)
            _PRODUCTS_CACHE = []
    return _PRODUCTS_CACHE


def _compact_product(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "price": p.get("price"),
        "currency": p.get("currency"),
        "category": p.get("category"),
        "stockStatus": p.get("stockStatus"),
    }


@tool
def searchProducts(query: str) -> str:
    """Search the product catalog by name, description or category."""
    q = query.strip().lower()
    if not q:
        return "No query provided."
    hits = [
        p for p in load_products()
        if q in (p.get("name") or "").lower()
        or q in (p.get("description") or "").lower()
        or q in (p.get("category") or "").lower()
    ]
    if not hits:
        return f"No products found for '{query}'."
    return json.dumps([_compact_product(p) for p in hits[:5]], ensure_ascii=False)


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
    """Get the full details for a specific product by its id."""
    for p in load_products():
        if p.get("id") == product_id:
            return json.dumps(p, ensure_ascii=False)
    return f"Product not found for id '{product_id}'."


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

TOOL_NAMES = tuple(t.name for t in TOOLS)
TOOL_DESCRIPTIONS = {
    "searchProducts": "Search the catalog for products matching a free-text query.",
    "recallPreviousProducts": "Bring back products shown in an earlier product search.",
    "selectProduct": "Select a specific product from the current search results.",
    "getProductDetails": "Return full details for a product by its id.",
    "suggestProducts": "Suggest products matching some criteria.",
    "calculateShipping": "Estimate shipping cost for an order to a location.",
    "getOrderStatus": "Check the status of an existing order.",
    "createOrder": "Place an order for the product already selected in the active flow (set quantity).",
    "confirmOrder": "Confirm a pending order.",
    "modifyOrder": "Modify an existing order.",
    "cancelOrder": "Cancel an existing order.",
    "escalateConversation": "Only when the request truly cannot be handled by the other tools.",
}


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
    flow_direction: str = "normal"
    flow_action: str | None = None
    escalation: bool = False
    proposed_intent: str | None = None


class CheckResult(BaseModel):
    needs_tool: bool = Field(default=False, description="Whether the query requires a tool")
    tool_name: str | None = Field(default=None, description="Name of the tool (or proposed intent name) to use if needed")
    reply: str = Field(default="", description="Direct answer if no tool is needed")


class ToolChoice(BaseModel):
    tool: Literal[*TOOL_NAMES] | None = Field(default=None, description="The tool to use for the user's request")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    proposed_new_tool: str | None = Field(default=None, description="Name of a missing intent/tool the request needs when NONE of the available tools fits")


class AddressFields(BaseModel):
    wilaya: str | None = Field(default=None, description="The wilaya (province) mentioned in the customer message.")
    commune: str | None = Field(default=None, description="The commune (city/town) mentioned in the customer message.")
    address: str | None = Field(default=None, description="The street or detailed delivery address mentioned in the customer message.")


class FlowResolution(BaseModel):
    action: Literal["CONTINUE", "CREATE", "NO_FLOW_LOOKUP", "CLARIFY", "INVALID_ACTION"]
    flow_id: str | None = Field(default=None, description="Flow to apply the tool call on (CONTINUE)")
    reason: str | None = Field(default=None, description="Reason for CLARIFY / INVALID_ACTION")
    product_name: str | None = Field(default=None, description="Product name for CREATE")
    filters: Filter | None = Field(default=None, description="Filters for CREATE")


class DraftRoute(BaseModel):
    decision: Literal["continue_draft", "new_request", "cancel"]


def struct_schema_hint(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single valid JSON object ONLY, with no prose, markdown, or extra text. "
        "Never use an OpenAI function call format (do not wrap the object in {\"name\": ..., "
        "\"arguments\": ...}). Output the raw JSON object directly.\n"
        "The JSON must conform exactly to this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )


def _content_text(raw: Any) -> str:
    content = getattr(raw, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", "") if c.get("type") == "text" else json.dumps(c))
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content or "")


def _extract_json_span(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_json_dict(text: str) -> dict[str, Any] | None:
    candidates = [text, _extract_json_span(text)]
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "name" in d and "arguments" in d:
            args = d["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            candidates.append(json.dumps(args))
    except (json.JSONDecodeError, TypeError):
        pass
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_json_obj(text: str, schema: type[BaseModel]) -> BaseModel | None:
    obj = _parse_json_dict(text)
    if obj is None:
        return None
    try:
        return schema(**obj)
    except (ValidationError, TypeError):
        return None


def _call_json(model, schema: type[BaseModel], fallback: BaseModel, messages: list) -> BaseModel:
    retry_msg = HumanMessage(
        content="That was not valid JSON output. Reply with ONLY the raw JSON object matching the "
        "schema, with no prose and no name/arguments wrapper."
    )
    for messages_ in (messages, [*messages, retry_msg]):
        try:
            raw = model.invoke(messages_)
        except Exception as e:
            logger.warning("structured call failed (%s)", e)
            continue
        parsed = _parse_json_obj(_content_text(raw), schema)
        if parsed is not None:
            return parsed
        logger.warning("structured response did not parse: %s", _content_text(raw)[:200])
    return fallback


# ============================================================
# Tool schema introspection + argument draft helpers
# ============================================================

import re as _re

_NUM_RE = _re.compile(r"\b\d+\b")
_AFFIRM_RE = _re.compile(
    r"\b(yes|yeah|yep|ok|okay|sure|fine|go|go ahead|do it|deal|let's go|lets go|nedi|ekhdem|zid|na'am|d'accord|aight)\b",
    _re.IGNORECASE,
)
_CANCEL_RE = _re.compile(
    r"\b(cancel|cancel it|forget|forget it|never mind|nevermind|nvm|stop|abort|leave it|drop it|"
    r"asba|khaleh|la|no thanks|not now|later)\b",
    _re.IGNORECASE,
)

_BUY_RE = _re.compile(
    r"\b(buy|order|take|get|want|grab|nedi|khoud|ekhdem|prefer)\b",
    _re.IGNORECASE,
)

_STOP_WORDS = {
    "the", "this", "that", "your", "please", "actually", "will", "would", "could",
    "with", "and", "for", "you", "i", "me", "just", "want",
}

_SCHEMA_CACHE: dict[str, tuple[list[str], dict[str, Any]]] = {}


def tool_for(tool_name: str, flow: Flow | None = None):
    if tool_name == "createOrder":
        return _make_create_order_tool(flow)
    return next((t for t in TOOLS if t.name == tool_name), None)


def _tool_schema_info(tool_name: str) -> tuple[list[str], dict[str, Any]]:
    if tool_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[tool_name]
    required: list[str] = []
    props: dict[str, Any] = {}
    try:
        tool = tool_for(tool_name)
        schema = tool.args_schema.model_json_schema()
        required = list(schema.get("required", []))
        props = schema.get("properties", {})
    except Exception as e:
        logger.warning("Could not introspect schema for %s: %s", tool_name, e)
    _SCHEMA_CACHE[tool_name] = (required, props)
    return required, props


def _coerce_args(raw: dict[str, Any], tool_name: str) -> dict[str, Any]:
    _, props = _tool_schema_info(tool_name)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in props or v is None or v == "":
            continue
        t = props[k].get("type")
        try:
            if t == "integer":
                out[k] = int(float(v))
            elif t == "number":
                out[k] = float(v)
            elif t == "boolean":
                if isinstance(v, bool):
                    out[k] = v
                elif str(v).lower() in ("true", "1"):
                    out[k] = True
                elif str(v).lower() in ("false", "0"):
                    out[k] = False
                else:
                    continue
            else:
                out[k] = str(v).strip()
        except (ValueError, TypeError):
            continue
    return out


def _field_missing(args: dict[str, Any], tool_name: str) -> list[str]:
    required, _ = _tool_schema_info(tool_name)
    return [f for f in required if args.get(f) is None or args.get(f) == ""]


ADDRESS_FIELD_DESCS = {
    "wilaya": "the wilaya (province) to ship to",
    "commune": "the commune (city/town) to ship to",
    "address": "the street or detailed delivery address",
}


def _prereqs_for(tool_name: str, mem: ConversationMemory) -> dict[str, Any]:
    if tool_name != "createOrder":
        return {}
    gi = mem.global_information
    prereqs: dict[str, Any] = {}
    for f in ADDRESS_FIELD_DESCS:
        if not getattr(gi, f, None):
            prereqs[f] = None
    return prereqs


def _missing_prereqs(prereqs: dict[str, Any]) -> list[str]:
    return [k for k, v in prereqs.items() if v is None or v == ""]


def _flow_draft(state: AgentState) -> tuple[Flow | None, ToolCallDraft | None]:
    flow = _active_flow(state.conversation_memory, state.resolved_flow_id)
    return flow, (flow.tool_draft if flow else None)


def _llm_phrase(system: str, human: str) -> str:
    try:
        msg = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = _content_text(msg).strip()
        return text or "Could you provide some more information, please?"
    except Exception as e:
        logger.warning("reply phrasing call failed (%s)", e)
        return "Could you provide some more information, please?"


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


def _recent_transcript(state: AgentState, n: int = 6) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for m in reversed(state.messages):
        if len(items) >= n:
            break
        if isinstance(m, dict):
            role = str(m.get("role", "?"))
            text = m.get("content", "")
        else:
            mtype = getattr(m, "type", "")
            if mtype == "human":
                role = "customer"
            elif mtype == "ai":
                role = "assistant"
            elif mtype == "tool":
                role = "tool"
            else:
                role = mtype or "?"
            text = getattr(m, "content", "") or ""
        if isinstance(text, list):
            parts = []
            for c in text:
                if isinstance(c, dict):
                    parts.append(c.get("text", "") if c.get("type") == "text" else json.dumps(c))
                else:
                    parts.append(str(c))
            text = " ".join(parts)
        items.append({"role": role, "text": str(text)[:300]})
    items.reverse()
    return items


# ============================================================
# Nodes
# ============================================================

def draft_gate(state: AgentState) -> dict:
    direction = "normal"
    _, draft = _flow_draft(state)
    if draft is not None and draft.status == "drafting":
        text = last_user_text(state)
        low = text.lower()
        if _CANCEL_RE.search(low):
            direction = "cancel"
        elif _NUM_RE.search(low) or _AFFIRM_RE.search(low):
            direction = "continue"
        else:
            system = (
                "A tool call is partially filled and we are waiting for missing arguments from the "
                "customer. Decide whether the incoming customer message is:\n"
                "- continue_draft: answering the pending question (providing one of the missing values, "
                "or affirming it)\n"
                "- cancel: abandoning/cancelling the pending step\n"
                "- new_request: an unrelated or new request\n"
                + struct_schema_hint(DraftRoute)
            )
            payload = json.dumps({
                "pending_tool": draft.tool_name,
                "missing_arguments": draft.missing,
                "already_collected": draft.args,
                "customer_message": text,
            }, ensure_ascii=False, default=str)
            dec = _call_json(
                model,
                DraftRoute,
                DraftRoute(decision="new_request"),
                [SystemMessage(content=system), HumanMessage(content=payload)],
            )
            direction = {
                "continue_draft": "continue",
                "cancel": "cancel",
                "new_request": "normal",
            }.get(dec.decision, "normal")
    updates: dict[str, Any] = {"flow_direction": direction}
    if direction == "cancel":
        mem = state.conversation_memory.model_copy(deep=True)
        flow = _active_flow(mem, state.resolved_flow_id)
        if flow is not None:
            flow.tool_draft = None
            flow.updated_at = datetime.now(timezone.utc)
        updates.update({
            "conversation_memory": mem,
            "needs_tool": False,
            "reply": _llm_phrase(
                "You are a helpful e-commerce assistant. Briefly and kindly confirm that you are "
                "cancelling the step in progress. Match the customer's language.",
                f"Cancelling step ({draft.tool_name}). Customer said: {text}",
            ),
        })
    logger.info("draft_gate -> direction=%s draft=%s", direction, draft.tool_name if draft else None)
    return updates


def extract_tool_args(state: AgentState) -> dict:
    mem = state.conversation_memory.model_copy(deep=True)
    flow = _active_flow(mem, state.resolved_flow_id)
    if flow is None or flow.tool_draft is None or flow.tool_draft.status != "drafting":
        return {}
    draft = flow.tool_draft
    tool = tool_for(draft.tool_name, flow)
    if tool is None:
        return {}
    selected = None
    if flow.product_discovery is not None and flow.product_discovery.selected_product is not None:
        selected = flow.product_discovery.selected_product.model_dump()
    system = (
        "You extract tool arguments from an e-commerce customer message. The target tool and its "
        "schema are given. Output ONLY a JSON object with the fields you can confidently determine "
        "from the customer message; leave every other field OUT. Do not invent values, do not repeat "
        "the schema itself.\n"
        + struct_schema_hint(tool.args_schema)
    )
    human = json.dumps({
        "tool": draft.tool_name,
        "already_collected": draft.args,
        "missing_so_far": draft.missing,
        "selected_product": selected,
        "customer_message": last_user_text(state),
    }, ensure_ascii=False, default=str)
    extracted: dict[str, Any] = {}
    try:
        msg = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = _parse_json_dict(_content_text(msg)) or {}
        extracted = _coerce_args(raw, draft.tool_name)
    except Exception as e:
        logger.warning("extract_tool_args call failed (%s)", e)
    before_args_missing = _field_missing(draft.args, draft.tool_name)
    before_prereq_missing = _missing_prereqs(_prereqs_for(draft.tool_name, mem))
    draft.args.update(extracted)

    prereq_missing = _missing_prereqs(_prereqs_for(draft.tool_name, mem))
    if prereq_missing:
        addr_system = (
            "You extract a shipping address from an e-commerce customer message. Output ONLY a JSON "
            "object with the fields you can confidently determine; leave every other field OUT. Do "
            "not invent values, do not repeat the schema itself.\n"
            + struct_schema_hint(AddressFields)
        )
        addr_human = json.dumps({
            "customer_message": last_user_text(state),
            "already_known": {f: getattr(mem.global_information, f, None) for f in ADDRESS_FIELD_DESCS},
        }, ensure_ascii=False, default=str)
        addr_raw: dict[str, Any] = {}
        try:
            msg = model.invoke([SystemMessage(content=addr_system), HumanMessage(content=addr_human)])
            addr_raw = _parse_json_dict(_content_text(msg)) or {}
        except Exception as e:
            logger.warning("address extraction call failed (%s)", e)
        for k in prereq_missing:
            v = addr_raw.get(k)
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            if v is not None and str(v).strip():
                draft.prereqs[k] = str(v).strip()
            else:
                draft.prereqs.setdefault(k, None)
        gi = mem.global_information
        if any(draft.prereqs.get(k) for k in ADDRESS_FIELD_DESCS):
            for k in ADDRESS_FIELD_DESCS:
                if draft.prereqs.get(k):
                    setattr(gi, k, str(draft.prereqs[k]).strip())
            flow.shipping = ShippingContext(
                wilaya=gi.wilaya,
                commune=gi.commune,
                address=gi.address,
                shipping_cost=flow.shipping.shipping_cost if flow.shipping else None,
            )
            logger.info("global address updated -> wilaya=%s commune=%s address=%s", gi.wilaya, gi.commune, gi.address)

    missing_args = _field_missing(draft.args, draft.tool_name)
    missing_prereqs = _missing_prereqs(_prereqs_for(draft.tool_name, mem))
    if (
        missing_args != before_args_missing
        or missing_prereqs != before_prereq_missing
        or extracted
    ):
        draft.attempts = 0
    else:
        draft.attempts += 1
    draft.missing = missing_args + missing_prereqs
    if not draft.missing:
        draft.status = "ready"
    flow.updated_at = datetime.now(timezone.utc)
    logger.info(
        "extract_tool_args -> tool=%s extracted=%s prereqs=%s missing=%s status=%s attempts=%s",
        draft.tool_name, extracted, draft.prereqs, draft.missing, draft.status, draft.attempts,
    )
    return {"conversation_memory": mem}


def ask_reply(state: AgentState) -> dict:
    flow, draft = _flow_draft(state)
    if draft is None or flow is None:
        return {}
    required, props = _tool_schema_info(draft.tool_name)
    missing = draft.missing or _field_missing(draft.args, draft.tool_name)
    fields = [
        f"{name}: {props.get(name, {}).get('description') or name}"
        for name in missing
        if name in props
    ]
    known_gi = state.conversation_memory.global_information
    prereq_missing = _missing_prereqs(_prereqs_for(draft.tool_name, state.conversation_memory))
    fields.extend(f"{name}: {desc}" for name, desc in ADDRESS_FIELD_DESCS.items() if name in prereq_missing)
    selected = None
    if flow.product_discovery is not None and flow.product_discovery.selected_product is not None:
        selected = flow.product_discovery.selected_product.model_dump()
    system = (
        "You are a helpful e-commerce assistant completing a multi-step task. The customer must "
        "provide a few missing pieces of information. Ask ONLY for those missing items, phrased "
        "naturally and conversationally in the customer's language. Never ask about information we "
        "already know, and never restate the full task."
    )
    human = json.dumps({
        "task_tool": draft.tool_name,
        "missing_items": fields,
        "already_known": draft.args,
        "known_address": {f: getattr(known_gi, f, None) for f in ADDRESS_FIELD_DESCS},
        "selected_product": selected,
        "customer_message": last_user_text(state),
    }, ensure_ascii=False, default=str)
    return {"needs_tool": False, "reply": _llm_phrase(system, human)}


def after_extract(state: AgentState) -> str:
    _, draft = _flow_draft(state)
    if draft is not None and draft.status == "ready":
        return "calling_tool"
    return "ask_reply"


def check_llm(state: AgentState) -> dict:
    system = (
        "You are the intent classifier of an e-commerce assistant. Available intents:\n"
        + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
        + "\nDecide whether the customer's latest message needs a tool-backed action or can be answered "
        "directly:\n"
        "- Plain conversation or lightweight questions -> needs_tool=false, set reply.\n"
        "- Needs an action covered by one of the available intents -> needs_tool=true, tool_name=that intent.\n"
        "- Needs an action NOT covered by any available intent -> needs_tool=true and set tool_name to a "
        "short proposed intent name (e.g. 'warrantyClaim', 'refundRequest') so it can be escalated.\n"
        + struct_schema_hint(CheckResult)
    )
    messages = [SystemMessage(content=system), *state.messages]
    fallback = CheckResult(
        needs_tool=False,
        reply="I couldn't process that request. Could you please rephrase?",
    )
    result = _call_json(model, CheckResult, fallback, messages)
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
    active = _active_flow(state.conversation_memory, state.resolved_flow_id)
    flow_context = None
    if active is not None:
        selected = None
        if active.product_discovery:
            if active.product_discovery.selected_product is not None:
                selected = active.product_discovery.selected_product.model_dump()
            elif active.product_discovery.tool_results:
                selected = active.product_discovery.tool_results[0].model_dump()
        flow_context = {
            "flow_id": active.flow_id,
            "state": active.state.value if isinstance(active.state, FlowState) else active.state,
            "selected_product": selected,
        }
    selected_flows = []
    for f in state.conversation_memory.flows:
        if f.product_discovery and f.product_discovery.selected_product is not None:
            p = f.product_discovery.selected_product
            selected_flows.append({
                "flow_id": f.flow_id,
                "state": f.state.value if isinstance(f.state, FlowState) else f.state,
                "product_name": p.product_name,
                "product_id": p.product_id,
            })

    low = user_text.lower()
    tokens = set(t for t in _re.findall(r"[a-z]{3,}", low) if t not in _STOP_WORDS)
    forced = None
    if _BUY_RE.search(low):
        for f in state.conversation_memory.flows:
            if f.product_discovery and f.product_discovery.selected_product is not None:
                p = f.product_discovery.selected_product
                p_tokens = set(_re.findall(r"[a-z]{3,}", p.product_name.lower()))
                if tokens & p_tokens:
                    forced = p
                    break
    if forced is not None:
        choice = ToolChoice(tool="createOrder", arguments={})
        result = f"Selected tool: {choice.tool}. Args: {choice.arguments} (deterministic buy fast-path)"
        logger.info("query_tool -> %s", result)
        return {
            "tool_outputs": [result],
            "tool_calls": [{"name": choice.tool, "arguments": choice.arguments}],
        }

    system = (
        "You pick the single best tool for the user's request. Available tools:\n"
        + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
        + "\nRules:\n"
        "- Read `recent_conversation` before deciding. The user's latest message often answers an "
        "earlier question or continues an earlier subject - use that history to infer the true intent "
        "of the latest message.\n"
        "- When the user wants to BUY or ORDER a product that is already selected or visible in the "
        "conversation, choose createOrder and set quantity accordingly.\n"
        "- If a product is selected in `flows_with_selected_product` and the user refers to it, choose "
        "createOrder even if it is not the currently active flow.\n"
        "- Never invent tools: if `requested_tool` is not in the list above, ignore it and remap the "
        "request to the correct tool from the list.\n"
        "- If the request needs an intent that NONE of the available tools supports, set "
        "`proposed_new_tool` to the missing intent/tool name (e.g. 'warrantyClaim', 'refundRequest') "
        "and leave `tool` as null. This triggers a human escalation and no tool is run.\n"
        "- Only choose escalateConversation if no other tool fits.\n"
        + struct_schema_hint(ToolChoice)
    )
    fallback_name = call.get("name") if call.get("name") in TOOL_NAMES else "searchProducts"
    fallback = ToolChoice(tool=fallback_name, arguments={})
    choice = _call_json(
        model,
        ToolChoice,
        fallback,
        [
            SystemMessage(content=system),
            HumanMessage(content=json.dumps({
                "requested_tool": call.get("name"),
                "user_request": user_text,
                "active_flow": flow_context,
                "flows_with_selected_product": selected_flows,
                "recent_conversation": _recent_transcript(state),
            }, ensure_ascii=False, default=str)),
        ],
    )
    proposed = (choice.proposed_new_tool or "").strip()
    if proposed or choice.tool is None or choice.tool == "escalateConversation":
        intent = proposed or (choice.tool or "unsupported")
        logger.warning("query_tool -> ESCALATE proposed_intent=%s", intent)
        return {
            "escalation": True,
            "proposed_intent": intent,
            "needs_tool": False,
            "reply": "",
            "tool_calls": [],
            "tool_outputs": [f"ESCALATE: {intent}"],
        }
    result = f"Selected tool: {choice.tool}. Args: {choice.arguments}"
    logger.info("query_tool -> %s", result)
    return {
        "tool_outputs": [result],
        "tool_calls": [{"name": choice.tool, "arguments": choice.arguments}],
    }


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
        "- A request to BUY or ORDER a product is a valid order action: if a flow already has the "
        "product selected, return CONTINUE with its flow_id; otherwise return CREATE (an order flow). "
        "Never return INVALID_ACTION for a buy/order request.\n"
        "- escalateConversation applies only when the matter is truly out of scope; return "
        "INVALID_ACTION with a reason.\n"
        "- If it is ambiguous which flow applies, return CLARIFY with a reason.\n"
        + struct_schema_hint(FlowResolution)
    )
    fallback = FlowResolution(action="CLARIFY", reason="Unable to resolve the flow automatically.")
    result = _call_json(
        model,
        FlowResolution,
        fallback,
        [
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(context, default=str)),
        ],
    )

    resolved_flow_id = state.resolved_flow_id
    now = datetime.now(timezone.utc)
    if result.action == "CREATE":
        new_flow_id = uuid.uuid4().hex
        tool_name = state.tool_calls[0].get("name", "") if state.tool_calls else ""
        if tool_name in ("searchProducts", "recallPreviousProducts", "selectProduct", "getProductDetails", "suggestProducts"):
            flow_state = FlowState.PRODUCT_DISCOVERY
            product_discovery = ProductDiscoveryContext(input=ProductDiscoveryInput(product_name=result.product_name, filters=result.filters or Filter()))
        else:
            flow_state = FlowState.ORDER
            product_discovery = None
        new_flow = Flow(
            flow_id=new_flow_id,
            state=flow_state,
            created_at=now,
            updated_at=now,
            product_discovery=product_discovery,
        )
        if tool_name:
            new_flow.tool_draft = ToolCallDraft(tool_name=tool_name)
        mem.flows.append(new_flow)
        mem.active_flow_id = new_flow_id
        resolved_flow_id = new_flow_id
    elif result.action == "CONTINUE":
        mem.active_flow_id = result.flow_id
        resolved_flow_id = result.flow_id
        flow = _active_flow(mem, resolved_flow_id)
        if flow is not None:
            tool_name = state.tool_calls[0].get("name", "") if state.tool_calls else ""
            if (
                tool_name
                and (flow.tool_draft is None or flow.tool_draft.status in ("executed", "cancelled") or flow.tool_draft.tool_name != tool_name)
            ):
                flow.tool_draft = ToolCallDraft(tool_name=tool_name)
            flow.updated_at = now
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


def _active_flow(mem: ConversationMemory, resolved_flow_id: str | None) -> Flow | None:
    flow_id = resolved_flow_id or mem.active_flow_id
    if not flow_id:
        return None
    return next((f for f in mem.flows if f.flow_id == flow_id), None)


def _make_create_order_tool(flow: Flow | None):
    @tool
    def createOrder(quantity: int) -> str:
        """Create an order for the product selected in the current flow."""
        product = None
        if flow is not None and flow.product_discovery is not None:
            if flow.product_discovery.selected_product is not None:
                product = flow.product_discovery.selected_product
            elif flow.product_discovery.tool_results:
                product = flow.product_discovery.tool_results[0]
        if product is None:
            return (
                f"No product selected in flow {flow.flow_id if flow else 'none'} to order. "
                "Run a product search first so the product is attached to the active flow."
            )
        shipping = flow.shipping if flow is not None else None
        payload: dict[str, Any] = {
            "tool": "createOrder",
            "product_id": product.product_id,
            "product_name": product.product_name,
            "price": product.price,
            "currency": "DZD",
            "quantity": quantity,
            "status": "order_created",
        }
        if shipping and (shipping.wilaya or shipping.commune or shipping.address):
            payload["shipping"] = shipping.model_dump()
        else:
            payload["shipping"] = None
            payload["status"] = "order_pending_address"
            payload["message"] = "Please provide your shipping address (wilaya and commune) so we can finalize the order."
        return json.dumps(payload, ensure_ascii=False)
    return createOrder


def _tool_result_products(content: str) -> list[Product] | None:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    products = []
    for item in data:
        pid = item.get("product_id") or item.get("id")
        name = item.get("product_name") or item.get("name")
        price = item.get("price")
        if not pid or not name or price is None:
            continue
        products.append(Product(product_id=str(pid), product_name=str(name), price=float(price)))
    return products or None


PRODUCT_RESULT_TOOLS = {"searchProducts", "getProductDetails", "recallPreviousProducts", "selectProduct", "suggestProducts"}


def calling_tool(state: AgentState) -> dict:
    mem = state.conversation_memory.model_copy(deep=True)
    flow = _active_flow(mem, state.resolved_flow_id)
    draft = flow.tool_draft if flow is not None else None

    ai_msg: Any = None
    tool_messages: list[Any] = []
    called_name = ""
    call_args: dict[str, Any] = {}

    if flow is not None:
        gi = mem.global_information
        if (gi.wilaya or gi.commune or gi.address) and (
            flow.shipping is None
            or flow.shipping.wilaya != gi.wilaya
            or flow.shipping.commune != gi.commune
            or flow.shipping.address != gi.address
        ):
            flow.shipping = ShippingContext(
                wilaya=gi.wilaya,
                commune=gi.commune,
                address=gi.address,
                shipping_cost=flow.shipping.shipping_cost if flow.shipping else None,
            )

    if draft is not None and draft.status == "ready":
        tool = tool_for(draft.tool_name, flow)
        if tool is None:
            return {}
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": tool.name,
                "args": dict(draft.args),
                "id": "draft-" + uuid.uuid4().hex[:8],
                "type": "tool_call",
            }],
        )
        try:
            tool_messages = ToolNode([tool]).invoke([ai_msg])
        except Exception as e:
            logger.warning("draft tool execution failed (%s)", e)
            tool_messages = [ToolMessage(
                content=f"Tool execution failed: {e}", tool_call_id="draft-exec-error"
            )]
        called_name = tool.name
        call_args = dict(draft.args)
        flow.tool_draft = None
        flow.updated_at = datetime.now(timezone.utc)
        logger.info("calling_tool executed ready draft -> %s %s", called_name, call_args)
    else:
        user_text = last_user_text(state)
        tool_context = state.tool_outputs[-1] if state.tool_outputs else "No tool selected."
        order_tool = _make_create_order_tool(flow)
        tools = [order_tool] + [t for t in TOOLS if t.name != "createOrder"]
        runtime_tool_node = ToolNode(tools)
        runtime_tool_model = model.bind_tools(tools)

        system = (
            "You are an e-commerce assistant. Call the requested tool to fulfill the user's request."
            f"\nResolved flow id: {state.resolved_flow_id or 'none'}. Tool context: {tool_context}"
        )
        ai_msg = runtime_tool_model.invoke([SystemMessage(content=system), HumanMessage(content=user_text)])
        if getattr(ai_msg, "tool_calls", None):
            tool_messages = runtime_tool_node.invoke([ai_msg])
        else:
            logger.warning("Tool model returned no tool call; skipping ToolNode")
            tool_messages = [
                ToolMessage(content="No tool was called by the assistant.", tool_call_id="no_tool_call")
            ]
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            called_name = ai_msg.tool_calls[0].get("name", "")
            call_args = ai_msg.tool_calls[0].get("args") or {}

    if called_name in PRODUCT_RESULT_TOOLS and tool_messages:
        prods = _tool_result_products(getattr(tool_messages[0], "content", ""))
        if prods and flow is not None:
            if flow.product_discovery is None:
                flow.product_discovery = ProductDiscoveryContext()
            flow.product_discovery.tool_results = prods
            selected = prods[0]
            if called_name == "selectProduct":
                selector = str(call_args.get("selector") or "").strip()
                selected = next(
                    (p for p in prods if p.product_id == selector or p.product_name.lower() == selector.lower()),
                    prods[0],
                )
            elif call_args.get("product_id"):
                selected = next((p for p in prods if p.product_id == str(call_args.get("product_id"))), prods[0])
            flow.product_discovery.selected_product = selected

    return {
        "messages": [ai_msg, *tool_messages],
        "tool_outputs": [getattr(m, "content", str(m)) for m in tool_messages],
        "conversation_memory": mem,
        "needs_tool": True,
    }


def reply(state: AgentState) -> dict:
    if not state.needs_tool:
        text = state.reply if state.reply.strip() else "I understood your request."
    else:
        tool_result = state.tool_outputs[-1] if state.tool_outputs else "No tool output available."
        user_text = last_user_text(state)
        response = model.invoke([
            SystemMessage(
                content=(
                    "You are a helpful e-commerce assistant. Answer the user's request based only on "
                    "the tool result provided. When the result lists products, present them as a short "
                    "bulleted list with name and price in the given currency, and note stock availability "
                    "if relevant. If the result says no products were found, acknowledge that politely. "
                    "Always use the exact currency code from the tool result (DZD for orders); never "
                    "invent or substitute a currency symbol."
                )
            ),
            HumanMessage(content=f"Tool result:\n{tool_result}\n\nUser request: {user_text}"),
        ])
        text = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=text)]}


# ============================================================
# InMemoryStore persistence (long-term memory)
# ============================================================

def _store_config(config: RunnableConfig) -> tuple[tuple[str, str], str]:
    cfg = (config or {}).get("configurable", {}) or {}
    store_key = cfg.get("store_key") or cfg.get("thread_id") or "default"
    return ("assistant", str(store_key)), str(store_key)


def _fresh_turn() -> dict[str, Any]:
    return {
        "escalation": False,
        "proposed_intent": None,
        "needs_tool": False,
        "reply": "",
        "flow_direction": "normal",
        "flow_action": None,
        "resolved_flow_id": None,
        "tool_calls": [],
        "tool_outputs": [],
    }


def hydrate(state: AgentState, *, store: BaseStore, config: RunnableConfig) -> dict:
    resets = _fresh_turn()
    if state.conversation_memory.flows:
        return resets
    ns, _ = _store_config(config)
    item = store.get(ns, "conversation_memory")
    if item is None:
        return resets
    return {**resets, "conversation_memory": ConversationMemory(**item.value)}


def persist(state: AgentState, *, store: BaseStore, config: RunnableConfig) -> dict:
    ns, _ = _store_config(config)
    store.put(ns, "conversation_memory", state.conversation_memory.model_dump(mode="json"))
    return {}


# ============================================================
# Routing
# ============================================================

def route(state: AgentState) -> str:
    return "query_tool" if state.needs_tool else "reply"


def draft_route(state: AgentState) -> str:
    return {"continue": "extract_tool_args", "cancel": "reply"}.get(state.flow_direction, "check_llm")


def query_escalate_route(state: AgentState) -> str:
    return "escalate" if state.escalation else "flow_resolver"


def escalate(state: AgentState) -> dict:
    logger.warning(
        "ESCALATION to human agent -> proposed_intent=%s",
        state.proposed_intent,
    )
    return {"needs_tool": False, "reply": ""}


def flow_route(state: AgentState) -> str:
    if state.flow_action in ("NO_FLOW_LOOKUP", "CLARIFY", "INVALID_ACTION"):
        return "reply"
    _, draft = _flow_draft(state)
    if draft is not None and draft.status == "drafting":
        return "extract_tool_args"
    return "calling_tool"


graph = StateGraph(AgentState)
graph.add_node("hydrate", hydrate)
graph.add_node("draft_gate", draft_gate)
graph.add_node("extract_tool_args", extract_tool_args)
graph.add_node("ask_reply", ask_reply)
graph.add_node("check_llm", check_llm)
graph.add_node("query_tool", query_tool)
graph.add_node("escalate", escalate)
graph.add_node("flow_resolver", flow_resolver)
graph.add_node("calling_tool", calling_tool)
graph.add_node("reply", reply)
graph.add_node("persist", persist)

graph.add_edge(START, "hydrate")
graph.add_edge("hydrate", "draft_gate")
graph.add_conditional_edges(
    "draft_gate",
    draft_route,
    {"extract_tool_args": "extract_tool_args", "reply": "reply", "check_llm": "check_llm"},
)
graph.add_conditional_edges(
    "extract_tool_args",
    after_extract,
    {"calling_tool": "calling_tool", "ask_reply": "ask_reply"},
)
graph.add_edge("ask_reply", "reply")
graph.add_conditional_edges("check_llm", route, {"query_tool": "query_tool", "reply": "reply"})
graph.add_conditional_edges(
    "query_tool",
    query_escalate_route,
    {"escalate": "escalate", "flow_resolver": "flow_resolver"},
)
graph.add_edge("escalate", "persist")
graph.add_conditional_edges(
    "flow_resolver",
    flow_route,
    {"calling_tool": "calling_tool", "reply": "reply", "extract_tool_args": "extract_tool_args"},
)
graph.add_edge("calling_tool", "reply")
graph.add_edge("reply", "persist")
graph.add_edge("persist", END)

store = InMemoryStore()
app = graph.compile(checkpointer=InMemorySaver(), store=store)



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
        try:
            with open("graph.mmd", "w") as f:
                f.write(app.get_graph().draw_mermaid())
            logger.info("Saved mermaid source to graph.mmd")
        except Exception as e2:
            logger.warning("mermaid source write failed (%s)", e2)
        try:
            print(app.get_graph().draw_ascii())
        except Exception as e3:
            logger.warning("ascii render unavailable (%s)", e3)
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
        print("The graph state: ", result)
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