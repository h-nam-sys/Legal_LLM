import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--with",
            "mcp==2.1.1",
            "mcp",
            "run",
            "mcp_server/server.py",
        ],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            # Lấy danh sách tools
            tools = await session.list_tools()

            print("Available tools:")
            for tool in tools.tools:
                print("-", tool.name)

            # Gọi get_current_time
            result = await session.call_tool(
                "get_current_time",
                {}
            )

            print("\nCurrent time:")
            print(result)

            # Gọi get_random_law
            result = await session.call_tool(
                "get_random_law",
                {}
            )

            print("\nRandom law:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())