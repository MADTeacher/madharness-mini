import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from madharness_mini.loop import run_agent
from madharness_mini.mcp import McpToolProvider
from madharness_mini.mcp.config import load_mcp_server_configs
from madharness_mini.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    JsonRpcBuilder,
    parse_response,
)
from madharness_mini.mcp.provider import exported_tool_name
from madharness_mini.mcp.stdio import StdioMcpClient
from madharness_mini.policy import Policy
from madharness_mini.tools import ToolRegistry

from tests.helpers import HarnessTestCase


FAKE_MCP_SERVER = r'''
import json
import os
import sys
from pathlib import Path

marker = Path(sys.argv[1]) if len(sys.argv) > 1 else None
legacy = len(sys.argv) > 2 and sys.argv[2] == "legacy"
old_version = len(sys.argv) > 2 and sys.argv[2] == "old_version"

REQUIRED_META_KEYS = (
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientCapabilities",
)


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


try:
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")
        meta = (message.get("params") or {}).get("_meta") or {}
        if any(key not in meta for key in REQUIRED_META_KEYS):
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": "missing required _meta"},
                }
            )
        elif method == "server/discover":
            if legacy:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                )
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "resultType": "complete",
                            "supportedVersions": (
                                ["2025-11-25"] if old_version else ["2026-07-28"]
                            ),
                            "capabilities": {"tools": {}},
                            "_meta": {
                                "io.modelcontextprotocol/serverInfo": {
                                    "name": "fake",
                                    "version": "1.0",
                                }
                            },
                        },
                    }
                )
        elif method == "tools/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resultType": "complete",
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo input.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "fail": {"type": "boolean"},
                                        "check_env": {"type": "boolean"},
                                        "require_input": {"type": "boolean"},
                                    },
                                    "additionalProperties": False,
                                },
                            }
                        ],
                        "ttlMs": 60000,
                        "cacheScope": "private",
                    },
                }
            )
        elif method == "tools/call":
            args = message.get("params", {}).get("arguments", {})
            if args.get("require_input"):
                result = {
                    "resultType": "input_required",
                    "inputRequests": {
                        "login": {
                            "method": "elicitation/create",
                            "params": {"message": "Please log in"},
                        }
                    },
                    "requestState": "opaque-blob",
                }
            elif args.get("check_env"):
                result = {
                    "resultType": "complete",
                    "content": [
                        {
                            "type": "text",
                            "text": "secret="
                            + os.environ.get("MADHARNESS_MINI_API_KEY", "")
                            + ";demo="
                            + os.environ.get("DEMO_MODE", ""),
                        }
                    ],
                }
            else:
                result = {
                    "resultType": "complete",
                    "isError": bool(args.get("fail")),
                    "content": [
                        {"type": "text", "text": "echo:" + str(args.get("text", ""))}
                    ],
                    "structuredContent": {"seen": args},
                }
            send({"jsonrpc": "2.0", "id": request_id, "result": result})
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "unknown method"},
                }
            )
finally:
    if marker:
        marker.write_text("closed", encoding="utf-8")
'''


class McpTests(HarnessTestCase):
    def write_fake_server(self):
        cfg = self.make_cfg()
        script = cfg.root / "fake_mcp_server.py"
        marker = cfg.root / "mcp_closed.txt"
        script.write_text(FAKE_MCP_SERVER, encoding="utf-8")
        return cfg, script, marker

    def write_mcp_config(self, cfg, script, marker, mode=None, **overrides):
        args = [str(script), str(marker)] + ([mode] if mode else [])
        data = {
            "servers": {
                "fake": {
                    "enabled": True,
                    "command": sys.executable,
                    "args": args,
                    "cwd": ".",
                    "env": {"DEMO_MODE": "1"},
                    "timeout_seconds": 5,
                    **overrides,
                }
            }
        }
        (cfg.state_dir / "mcp.json").write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def test_jsonrpc_request_has_incrementing_id(self):
        rpc = JsonRpcBuilder()

        first = rpc.request("tools/list", {})
        second = rpc.request("tools/call", {"name": "echo"})

        self.assertEqual(first["id"], 1)
        self.assertEqual(second["id"], 2)
        self.assertEqual(first["jsonrpc"], "2.0")

    def test_jsonrpc_request_carries_stateless_meta(self):
        rpc = JsonRpcBuilder()

        request = rpc.request("tools/list")

        meta = request["params"]["_meta"]
        self.assertEqual(
            meta["io.modelcontextprotocol/protocolVersion"],
            MCP_PROTOCOL_VERSION,
        )
        self.assertEqual(meta["io.modelcontextprotocol/clientCapabilities"], {})
        self.assertEqual(
            meta["io.modelcontextprotocol/clientInfo"]["name"], "madharness-mini"
        )

    def test_jsonrpc_error_raises_runtime_error(self):
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }

        with self.assertRaisesRegex(RuntimeError, "MCP JSON-RPC error -32601"):
            parse_response(response, 1)

    def test_parse_response_defaults_missing_resulttype_to_complete(self):
        response = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}

        result = parse_response(response, 1)

        self.assertEqual(result, {"tools": []})

    def test_parse_response_rejects_unknown_resulttype(self):
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"resultType": "extension/unknown"},
        }

        with self.assertRaisesRegex(RuntimeError, "unknown resultType"):
            parse_response(response, 1)

    def test_mcp_config_missing_file_means_no_servers(self):
        cfg = self.make_cfg()

        self.assertEqual(load_mcp_server_configs(cfg, Policy(cfg)), [])

    def test_mcp_config_rejects_unsafe_cwd(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker, cwd="../outside")

        with self.assertRaisesRegex(RuntimeError, "path outside workspace"):
            load_mcp_server_configs(cfg, Policy(cfg))

    def test_stdio_client_discover_and_list_tools(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)
        config = load_mcp_server_configs(cfg, Policy(cfg))[0]
        client = StdioMcpClient(config)
        self.addCleanup(client.close)

        tools = client.start()

        self.assertEqual(tools[0]["name"], "echo")

    def test_stdio_client_rejects_server_with_old_protocol_only(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker, mode="old_version")
        config = load_mcp_server_configs(cfg, Policy(cfg))[0]
        client = StdioMcpClient(config)
        self.addCleanup(client.close)

        with self.assertRaisesRegex(RuntimeError, "supports protocol versions"):
            client.start()

    def test_stdio_client_rejects_legacy_handshake_server(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker, mode="legacy")
        config = load_mcp_server_configs(cfg, Policy(cfg))[0]
        client = StdioMcpClient(config)
        self.addCleanup(client.close)

        with self.assertRaisesRegex(RuntimeError, "legacy initialize handshake"):
            client.start()

    def test_mcp_provider_exports_toolspecs(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)
        provider = McpToolProvider()
        registry = ToolRegistry(cfg, providers=[provider])
        self.addCleanup(registry.close)

        names = [item["function"]["name"] for item in registry.schemas()]

        self.assertIn("mcp__fake__echo", names)

    def test_mcp_tool_call_returns_ok_observation(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)
        registry = ToolRegistry(cfg, providers=[McpToolProvider()])
        self.addCleanup(registry.close)

        obs = registry.call("mcp__fake__echo", {"text": "hello"})

        self.assertTrue(obs["ok"])
        self.assertEqual(obs["tool"], "mcp__fake__echo")
        self.assertEqual(obs["content"], "echo:hello")
        self.assertEqual(obs["data"]["seen"]["text"], "hello")

    def test_mcp_tool_call_iserror_returns_fail(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)
        registry = ToolRegistry(cfg, providers=[McpToolProvider()])
        self.addCleanup(registry.close)

        obs = registry.call("mcp__fake__echo", {"fail": True})

        self.assertFalse(obs["ok"])
        self.assertEqual(obs["content"], "echo:")

    def test_mcp_tool_call_input_required_returns_fail(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)
        registry = ToolRegistry(cfg, providers=[McpToolProvider()])
        self.addCleanup(registry.close)

        obs = registry.call("mcp__fake__echo", {"require_input": True})

        self.assertFalse(obs["ok"])
        self.assertIn("requires additional client input", obs["summary"])
        self.assertEqual(obs["requested_inputs"], ["login: elicitation/create"])

    def test_mcp_env_does_not_inherit_model_api_key(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)

        with patch.dict(os.environ, {"MADHARNESS_MINI_API_KEY": "secret"}):
            registry = ToolRegistry(cfg, providers=[McpToolProvider()])
            self.addCleanup(registry.close)
            obs = registry.call("mcp__fake__echo", {"check_env": True})

        self.assertTrue(obs["ok"])
        self.assertEqual(obs["content"], "secret=;demo=1")

    def test_mcp_process_is_closed_after_registry_close(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)
        provider = McpToolProvider()
        registry = ToolRegistry(cfg, providers=[provider])

        client = provider.clients[0]
        registry.close()

        self.assertIsNotNone(client.process)
        self.assertIsNotNone(client.process.poll())
        self.assertEqual(marker.read_text(encoding="utf-8"), "closed")

    def test_run_agent_closes_mcp_process_after_final_answer(self):
        cfg, script, marker = self.write_fake_server()
        self.write_mcp_config(cfg, script, marker)

        with patch(
            "madharness_mini.loop.ModelClient.chat",
            return_value={"choices": [{"message": {"content": "done"}}]},
        ):
            result, _trace_path = run_agent("finish", cfg)

        self.assertEqual(result, "done")
        self.assertEqual(marker.read_text(encoding="utf-8"), "closed")

    def test_exported_tool_name_replaces_unsupported_chars(self):
        self.assertEqual(
            exported_tool_name("docs", "search.index"),
            "mcp__docs__search_index",
        )
