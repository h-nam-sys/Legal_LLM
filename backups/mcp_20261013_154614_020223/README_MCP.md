# MCP search increment

## Install on your current Windows project

Stop backend and admin. Extract this increment into a separate folder, then run there:

```bat
python install_increment.py --project "C:\Users\Reines\Documents\GitHub\Legal_LLM"
```

The installer checks checksums, backs up all existing files before replacement, and restores them if installation fails. It does not copy datasets, database files, model weights, or secrets. No re-ingestion is needed.

```bat
cd /d C:\Users\Reines\Documents\GitHub\Legal_LLM
env\Scripts\activate
python -m pip install -r requirements_mcp.txt
python -m pip check
```

Use Python 3.10 or newer, in the same environment as the backend and admin. This increment pins the tested MCP v1 SDK rather than relying on the incompatible v2 client API. The old standalone `fastmcp` requirement was unused in this project; the implementation uses `mcp.server.fastmcp` from the official MCP SDK.

## Enable the bundled search server

1. Get a **Brave Search API** key from https://api-dashboard.search.brave.com/ . A Gemini key is not a Brave key. Review your provider's current plan and quota before enabling it.
2. Add this line to your existing project `.env` or `backend/.env`:

   ```dotenv
   BRAVE_API_KEY=your_real_brave_search_api_key
   ```

   Do not put the key in uploaded profiles or the admin's JSON settings. Gemini ingestion profiling continues to use its existing GEMINI_API_KEY.
3. Start Qwen and backend with your usual commands. Start admin with your existing launcher, then press Ctrl+F5 in chat.
4. In admin → **Tìm kiếm trực tuyến**, enable search and select permitted domains.
5. Configure allowed website names for each selected domain. Legal data can use `dichvucong.gov.vn`. Campus data should use your school's actual official website; do not leave it restricted to the government portal. Leave the list empty only if you intend to search the wider public web. The UI retains edits when changing the domain selector.
6. **Lưu cấu hình MCP** publishes the settings. Then **Kiểm tra kết nối** verifies MCP discovery. Connection testing does not call Brave or verify the Brave key.
7. Open **Thử tra cứu**, enter a query, and choose **Tra cứu thử** to check the provider key, website filtering, and returned snippets. This performs a real provider search using the displayed draft, without saving it. It does not invoke Qwen.

The bundled server starts automatically for a search and exits afterward. No extra terminal or startup service is needed. It loads no embedding models. MCP is disabled until configured; missing keys no longer lead to an ungrounded Gemini answer.

## Try the user flow

Use a domain explicitly for this initial check so routing ambiguity is separate from MCP. Choose Strict and ask a domain-relevant question that is absent from its dataset but covered by an allowed website.

1. Internal retrieval returns no usable evidence.
2. Chat offers online search and displays **Có / Không**. No provider call has happened yet.
3. **Có** approves that specific offer. The backend checks ownership, offer ID, current domain, global toggle, and domain permission.
4. The MCP client initializes a session, discovers the configured tool, and calls it. The bundled `web_search` tool calls Brave and returns actual URL/title/snippet records.
5. The client filters URLs by allowed host, deduplicates and bounds evidence, and sends snippets to the existing local Qwen endpoint.
6. Chat shows the answer with numbered references and a Sources section containing actual URLs and snippets. Answer and sources are saved in conversation history and the audit log.

If no usable snippets are returned, the answer says evidence is insufficient and does not call Qwen. Connection/tool/model failures return a readable error and leave the offer available for retry. Sending a new question invalidates the previous offer. Reopening history restores a still-pending offer. Duplicate clicks are serialized within the existing backend process. Failed searches are recorded as failures, rather than as successful web searches.

Friendly direct conversation remains direct. It offers MCP only when it has chosen retrieval and that retrieval found no evidence. Domain-specific MCP permissions are configured in this new tab; they override the older profile's `allow_external_search` setting when a rule is saved here.

## Advanced settings

The collapsed panel contains result limit, MCP timeout, answer token limit, answer instructions, and optional remote-server configuration.

Remote servers must support **Streamable HTTP**. Set the server URL, tool name, query-argument name, optional count-argument name, and optional fixed JSON arguments. Bearer tokens are referenced by an environment-variable name; they are not stored in settings. HTTPS is required for remote hosts; HTTP is supported for localhost.

The selected tool must return either MCP `structuredContent` or a JSON text block with this shape:

```json
{
  "results": [
    {
      "title": "Library borrowing rules",
      "url": "https://example.edu/library",
      "snippet": "Students must present a student card at the library desk."
    }
  ]
}
```

A list of these records, or `{ "result": { "results": [...] } }`, is also accepted. `description` is accepted as an alternative to `snippet`. Plain generated prose without URL/snippet records is rejected. Generic MCP servers returning a different format need an adapter; arbitrary MCP tools cannot be assumed to be search tools.

Source restrictions are applied twice: site operators guide the search query, and returned URL hostnames are checked by the client. The bundled tool also checks hosts itself. It accepts a website's subdomains but rejects similarly named unrelated hosts.

## Scope and limits

- The first implementation uses search snippets, not downloaded full pages. Snippets may omit conditions or current details; the prompt requires acknowledging missing evidence.
- Qwen remains responsible for interpreting evidence. Numbered citations and valid URLs do not by themselves establish answer correctness.
- Search is attempted only after an explicit offer approval. The provider receives that search query, and location only if enabled for the domain. Conversation history is not sent to the search provider.
- There is no extra domain-classifier call introduced by this patch. Existing detection, switching, local RAG, ingestion, and domain management remain in place.
- Settings are stored separately in `data/mcp_settings.json`, validated before atomic publication, and read again on each action. Changing keys in `.env` requires restarting backend/admin.
- Existing login remains the project's mock-login mechanism. This increment does not add production authentication, multi-worker distributed locks, arbitrary local commands, OAuth, or a server registry.

## Verification performed here

- Real stdio MCP discovery and tool invocation, plus a real local Streamable HTTP server/client round trip.
- Provider HTTP responses and Qwen replies substituted for deterministic protocol checks; no live Brave account or local model is available here.
- Real SQLite offer → approval → stored answer flow; failure retention; stale offer and duplicate approval rejection.
- Streamlit AppTest renders the new admin panel, verifies collapsed advanced controls, and publishes settings.
- Configuration validation, host restrictions, structured/text tool results, empty-evidence handling, and domain lifecycle regression checks.

`tests/test_mcp_increment.py` contains the focused checks. Your actual Brave key, Windows subprocess behavior, and Qwen quality still need the local run above. See `MCP_VALIDATION.txt` for the exact verification results, including pre-existing test failures.

Protocol and provider references: https://github.com/modelcontextprotocol/python-sdk and https://api-dashboard.search.brave.com/app/documentation/web-search/get-started .
