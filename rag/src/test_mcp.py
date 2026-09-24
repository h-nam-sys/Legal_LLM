import asyncio
from pathlib import Path
from fastmcp import Client


async def main():

    client = Client(
        Path("src/mcp_server.py")
    )

    async with client:

        print("Connected to MCP Server!")

        # ====================================================
        # 1. List tools
        # ====================================================

        tools = await client.list_tools()

        print("\nAvailable tools:")

        for tool in tools:
            print("-", tool.name)

        # ====================================================
        # 2. Sync
        # ====================================================

        print("\nTesting sync_laws...")

        sync_result = await client.call_tool(
            "sync_laws",
            {}
        )

        print("\nSYNC RESULT:")
        print(sync_result)

        # ====================================================
        # 3. Search
        # ====================================================

        print("\nTesting search_laws...")

        search_result = await client.call_tool(
            "search_laws",
            {
                "query": "Đăng ký khai sinh cần những giấy tờ gì?"
            }
        )

        print("\nSEARCH RESULT:")
        print(search_result)


asyncio.run(main())