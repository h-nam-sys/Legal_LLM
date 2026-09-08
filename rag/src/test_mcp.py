import asyncio
from fastmcp import Client


async def main():
    async with Client("src/mcp_server.py") as client:

        # Kiểm tra server có kết nối được không
        print("=" * 70)
        print("MCP SERVER TEST")
        print("=" * 70)

        # Liệt kê các tool
        tools = await client.list_tools()

        print("\nAvailable tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # Gọi tool search_mock_laws
        print("\n" + "=" * 70)
        print("CALLING search_mock_laws()")
        print("=" * 70)

        result = await client.call_tool(
            "search_mock_laws",
            {"query": "nghỉ việc"}
        )

        print("\nResult:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())