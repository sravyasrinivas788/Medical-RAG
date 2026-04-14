from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, CallToolResult
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response
import json
import uvicorn
from query import search
from database import get_conn
from psycopg2.extras import RealDictCursor


server=Server("medical-kb-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description="""Search the medical knowledge base using hybrid search.
Searches both WHO Essential Medicines PDF and structured database records
including drugs, dosages, contraindications, lab ranges and clinic policies.
Use for broad questions where you don't know exactly where the answer is.
Do NOT use for specific drug lookups — use get_drug_record instead.
Do NOT use for lab ranges — use get_lab_range instead.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — be specific, include drug names and medical terms"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return, default 5",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_drug_record",
            description="""Get the complete structured record for a specific drug.
Returns name, category, indication, dosage, contraindications, side effects.
Use when a specific drug name is mentioned and you need precise clinical data.
Much faster and more precise than search for known drug names.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Exact or partial drug name e.g. metformin, ibuprofen, warfarin"
                    }
                },
                "required": ["drug_name"]
            }
        ),
        Tool(
            name="get_lab_range",
            description="""Get the normal reference range for a specific lab test.
Returns test name, normal range, unit, and clinical notes.
Use when asked about normal lab values, test thresholds, or interpretation.
Much faster and more precise than search for known lab test names.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "Lab test name e.g. HbA1c, creatinine, INR, hemoglobin"
                    }
                },
                "required": ["test_name"]
            }
        ),
        Tool(
            name="get_clinic_policy",
            description="""Get clinic policy information on a specific topic.
Returns policy description for topics like appointments, prescriptions,
referrals, walk-ins, and patient records.
Use when asked about clinic procedures or administrative policies.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Policy topic e.g. missed appointment, prescription refill, referral"
                    }
                },
                "required": ["topic"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    result = execute_tool(name, arguments)
    return [TextContent(type="text", text=result)]

def execute_tool(tool_name: str, args: dict) -> str:
    if tool_name == "search_knowledge_base":
        query  = args["query"]
        top_k  = args.get("top_k", 5)
        print(f"[MCP] search_knowledge_base: '{query}'")

        results = search(query)
        if not results:
            return "No relevant results found for this query."

        return "\n\n".join(
            f"Result {i+1} (score: {r.get('rerank_score', 0)}, "
            f"source: {r['source']}, section: {r.get('heading', 'N/A')}):\n{r['text']}"
            for i, r in enumerate(results[:top_k])
        )

    elif tool_name == "get_drug_record":
        drug_name = args["drug_name"]
        print(f"[MCP] get_drug_record: '{drug_name}'")

        conn = get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM drugs WHERE LOWER(name) LIKE LOWER(%s)",
            (f"%{drug_name}%",)
        )
        drug = cur.fetchone()
        cur.close()
        conn.close()

        if not drug:
            return f"No drug record found for '{drug_name}'."

        return (
            f"Drug: {drug['name']}\n"
            f"Category: {drug['category']}\n"
            f"Indication: {drug['indication']}\n"
            f"Dosage: {drug['dosage']}\n"
            f"Contraindications: {drug['contraindications']}\n"
            f"Side effects: {drug['side_effects']}"
        )

    elif tool_name == "get_lab_range":
        test_name = args["test_name"]
        print(f"[MCP] get_lab_range: '{test_name}'")

        conn = get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM lab_ranges WHERE LOWER(test_name) LIKE LOWER(%s)",
            (f"%{test_name}%",)
        )
        lab = cur.fetchone()
        cur.close()
        conn.close()

        if not lab:
            return f"No lab range found for '{test_name}'."

        return (
            f"Test: {lab['test_name']}\n"
            f"Normal range: {lab['normal_range']} {lab['unit']}\n"
            f"Notes: {lab['notes']}"
        )

    elif tool_name == "get_clinic_policy":
        topic = args["topic"]
        print(f"[MCP] get_clinic_policy: '{topic}'")

        conn = get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM clinic_policies WHERE LOWER(topic) LIKE LOWER(%s)",
            (f"%{topic}%",)
        )
        policy = cur.fetchone()
        cur.close()
        conn.close()

        if not policy:
            return f"No policy found for topic '{topic}'."

        return f"Policy — {policy['topic']}:\n{policy['description']}"

    return f"Unknown tool: {tool_name}"

sse = SseServerTransport("/messages")

class SSEEndpoint:
    async def __call__(self, scope, receive, send):
        async with sse.connect_sse(scope, receive, send) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options()
            )

class MessagesEndpoint:
    async def __call__(self, scope, receive, send):
        await sse.handle_post_message(scope, receive, send)

mcp_app = Starlette(
    routes=[
        Route("/sse", endpoint=SSEEndpoint()),
        Route("/messages", endpoint=MessagesEndpoint(), methods=["POST"])
    ]
)

