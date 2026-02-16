"""
HTTP Tool Server for terminal-using agents.

Runs in a background thread within the benchmark process, exposing all CAR-bench
tools as REST endpoints. Terminal agents call tools by posting to these endpoints
(via the MCP server proxy or directly).

The server shares context vars with the benchmark process via
contextvars.copy_context(), ensuring correct access to context_state,
fixed_context, tool_execution_errors_during_runtime, etc.
"""

import contextvars
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Type

from car_bench.envs.tool import Tool
from car_bench.types import RESPOND_ACTION_NAME, USER_AS_A_TOOL_ACTION_NAMES, Action


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""

    daemon_threads = True


class ToolServer:
    """
    Manages an HTTP server that exposes CAR-bench tools as REST endpoints.

    Records all tool invocations as Action objects and tracks state hashes
    for reward calculation compatibility.
    """

    def __init__(
        self,
        tools_map: Dict[str, Type[Tool]],
        data: Dict[str, Any],
        tools_info: List[Dict[str, Any]],
        env: Any,
    ):
        self.tools_map = tools_map
        self.data = data
        self.tools_info = tools_info
        self.env = env

        # Capture parent thread's context for contextvar propagation
        self._parent_context = contextvars.copy_context()

        # Action and state recording
        self._action_log: List[Action] = []
        self._interaction_log: List[Dict[str, Any]] = []
        self._log_lock = threading.Lock()
        self._interaction_index = 0

        # User interaction synchronization
        self._user_message_pending = threading.Event()
        self._user_response_ready = threading.Event()
        self._pending_user_message: Optional[str] = None
        self._user_response: Optional[str] = None
        self._done = threading.Event()

        self._server: Optional[_ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    def start(self) -> int:
        """Start the HTTP server in a background thread. Returns the port."""
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Suppress default request logging
                pass

            def do_GET(self):
                path = self.path.rstrip("/")

                if path == "/tools":
                    self._respond_json(200, server_ref.tools_info)
                elif path.startswith("/tools/"):
                    tool_name = path[len("/tools/"):]
                    tool_info = next(
                        (
                            t
                            for t in server_ref.tools_info
                            if t.get("function", {}).get("name") == tool_name
                        ),
                        None,
                    )
                    if tool_info:
                        self._respond_json(200, tool_info)
                    else:
                        self._respond_json(
                            404, {"error": f"Unknown tool: {tool_name}"}
                        )
                elif path == "/health":
                    self._respond_json(200, {"status": "ok"})
                else:
                    self._respond_json(404, {"error": "Not found"})

            def do_POST(self):
                path = self.path.rstrip("/")
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

                try:
                    kwargs = json.loads(body) if body else {}
                except json.JSONDecodeError as e:
                    self._respond_json(400, {"error": f"Invalid JSON: {e}"})
                    return

                if path.startswith("/tool/"):
                    tool_name = path[len("/tool/"):]
                    self._handle_tool_call(tool_name, kwargs)
                elif path == "/user/message":
                    self._handle_user_message(kwargs)
                else:
                    self._respond_json(404, {"error": "Not found"})

            def _handle_tool_call(self, tool_name: str, kwargs: Dict[str, Any]):
                # Record the action
                action = Action(name=tool_name, kwargs=kwargs)
                with server_ref._log_lock:
                    server_ref._action_log.append(action)
                    idx = server_ref._interaction_index
                    server_ref._interaction_index += 1

                if tool_name not in server_ref.tools_map:
                    observation = json.dumps(
                        {"error": f"Unknown tool: {tool_name}"}
                    )
                    with server_ref._log_lock:
                        server_ref._interaction_log.append(
                            {
                                "type": "tool_call",
                                "index": idx,
                                "name": tool_name,
                                "kwargs": kwargs,
                                "result": observation,
                            }
                        )
                    self._respond_json(404, {"error": f"Unknown tool: {tool_name}"})
                    return

                # Execute tool in parent context to access correct context vars
                try:
                    observation = server_ref._parent_context.run(
                        server_ref.tools_map[tool_name].invoke,
                        data=server_ref.data,
                        **kwargs,
                    )
                except Exception as e:
                    observation = json.dumps({"error": str(e)})

                # Record state hash after tool execution
                try:
                    server_ref._parent_context.run(
                        server_ref.env._record_state_hash_if_needed
                    )
                except Exception:
                    pass

                # Log the interaction
                with server_ref._log_lock:
                    server_ref._interaction_log.append(
                        {
                            "type": "tool_call",
                            "index": idx,
                            "name": tool_name,
                            "kwargs": kwargs,
                            "result": observation if isinstance(observation, str) else json.dumps(observation),
                        }
                    )

                # Return result
                if isinstance(observation, str):
                    try:
                        result = json.loads(observation)
                    except json.JSONDecodeError:
                        result = {"result": observation}
                else:
                    result = observation

                self._respond_json(200, result)

            def _handle_user_message(self, kwargs: Dict[str, Any]):
                message = kwargs.get("message", "")

                if not message:
                    self._respond_json(
                        400, {"error": "Missing 'message' field"}
                    )
                    return

                # Record as a respond action
                action = Action(
                    name=RESPOND_ACTION_NAME, kwargs={"content": message}
                )
                with server_ref._log_lock:
                    server_ref._action_log.append(action)
                    idx = server_ref._interaction_index
                    server_ref._interaction_index += 1

                # Signal orchestrator that agent wants to talk to user
                server_ref._pending_user_message = message
                server_ref._user_response_ready.clear()
                server_ref._user_message_pending.set()

                # Wait for user response (or done signal)
                while not server_ref._done.is_set():
                    if server_ref._user_response_ready.wait(timeout=1.0):
                        break

                user_response = server_ref._user_response or ""
                is_done = server_ref._done.is_set()

                # Log user interaction
                with server_ref._log_lock:
                    server_ref._interaction_log.append(
                        {
                            "type": "user_message",
                            "index": idx,
                            "agent_said": message,
                            "user_replied": user_response,
                        }
                    )

                self._respond_json(
                    200,
                    {
                        "user_response": user_response,
                        "done": is_done,
                    },
                )

            def _respond_json(self, status_code: int, data: Any):
                response = json.dumps(data)
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

        # Bind to random available port
        self._server = _ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]

        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()

        return port

    def stop(self):
        """Shut down the HTTP server."""
        self._done.set()
        self._user_message_pending.set()  # Unblock any waiting threads
        self._user_response_ready.set()
        if self._server:
            self._server.shutdown()

    def get_actions(self) -> List[Action]:
        """Return all recorded actions."""
        with self._log_lock:
            return list(self._action_log)

    def get_interaction_log(self) -> List[Dict[str, Any]]:
        """Return the full interaction log for message reconstruction."""
        with self._log_lock:
            return list(self._interaction_log)

    def get_pending_user_message(self) -> Optional[str]:
        """
        Block until the agent sends a message to the user.
        Returns None if the server has been stopped.
        """
        while not self._done.is_set():
            if self._user_message_pending.wait(timeout=1.0):
                self._user_message_pending.clear()
                return self._pending_user_message
        return None

    def set_user_response(self, response: str):
        """Set the user's response to unblock the agent's /user/message request."""
        self._user_response = response
        self._user_response_ready.set()

    def signal_done(self):
        """Signal that the conversation is done (e.g., user said ###STOP###)."""
        self._done.set()
        self._user_message_pending.set()
        self._user_response_ready.set()
