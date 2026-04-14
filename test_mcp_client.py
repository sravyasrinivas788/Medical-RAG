# test_mcp_client.py
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test():
    async with sse_client("http://localhost:8000/mcp/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

           
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # call a tool
            result = await session.call_tool(
                "get_drug_record",
                {"drug_name": "metformin"}
            )
            print("\nDrug record result:")
            print(result.content[0].text)

            # search
            result = await session.call_tool(
                "search_knowledge_base",
                {"query": "hypertension medicines", "top_k": 3}
            )
            print("\nSearch result:")
            print(result.content[0].text[:500])

asyncio.run(test())