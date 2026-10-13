"""Bundled MCP stdio server. No embedding model; one Brave Web Search tool."""
import html
import os
from pathlib import Path
import re
import sys
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rag.mcp_config import hosts, valid_source_url
load_dotenv(ROOT / 'backend/.env')
load_dotenv(ROOT / '.env')
server = FastMCP('Legal LLM Web Search')

def plain(value):
    return html.unescape(re.sub(r'<[^>]*>', '', str(value or ''))).strip()

@server.tool()
async def web_search(query: str, count: int = 5, allowed_hosts: list[str] | None = None) -> dict:
    """Search public web snippets. Returns results [{title, url, snippet}]; no invented answer."""
    if not query.strip() or len(query) > 4000 or not 1 <= count <= 10:
        raise ValueError('Invalid query or result count')
    allowed = hosts(allowed_hosts or [])
    key = os.getenv('BRAVE_API_KEY')
    if not key:
        raise ValueError('BRAVE_API_KEY is not configured on the server')
    restriction = ' (' + ' OR '.join('site:' + h for h in allowed) + ')' if allowed else ''
    async with httpx.AsyncClient(timeout=35) as client:
        response = await client.get('https://api.search.brave.com/res/v1/web/search',
            headers={'X-Subscription-Token': key, 'Accept': 'application/json'},
            params={'q': query + restriction, 'count': count, 'extra_snippets': 'true', 'safesearch': 'moderate'})
        # Never expose provider request headers or credentials through tool errors.
        if response.status_code != 200:
            raise ValueError(f'Search provider returned HTTP {response.status_code}')
        data = response.json()
    results = []
    for row in data.get('web', {}).get('results', []):
        url = row.get('url', '')
        if valid_source_url(url, allowed):
            extra = row.get('extra_snippets') or []
            snippet = plain(row.get('description')) + '\n' + '\n'.join(plain(s) for s in extra[:3])
            if snippet.strip():
                results.append(dict(title=plain(row.get('title'))[:300], url=url, snippet=snippet.strip()[:2500]))
    return {'results': results[:count]}

if __name__ == '__main__':
    server.run(transport='stdio')
