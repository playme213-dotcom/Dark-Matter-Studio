"""
Dark Matter Studio — Swarm Multi-Agent Module

Implements the OpenAI Swarm pattern (Agent + handoffs) adapted for Colibri/Dark
Matter Studio's local tool backend. Stateless between runs, like OpenAI Swarm.

Core primitives:
  - Agent: name, model, instructions, functions (tool names), handoffs (list of Agent or handoff functions)
  - Swarm: client.run(agent, messages, context_variables, max_turns, stream) -> Response
  - Response: messages (with 'sender' field), agent, context_variables

Handoff: an Agent function that returns another Agent object (or a Result with
agent/context_variables/value) to transfer execution.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

# Reuse Dark Matter Studio's tool dispatch for executing tools inside swarm agents.
from agent_tools import dispatch_tool_call


class Result:
    """Return value from an Agent function. Can carry a value, an agent handoff,
    and context variable updates — matching OpenAI Swarm semantics."""

    def __init__(
        self,
        value: Any = None,
        agent: Optional["Agent"] = None,
        context_variables: Optional[Dict[str, Any]] = None,
    ):
        self.value = value
        self.agent = agent
        self.context_variables = context_variables or {}


class Agent:
    """An Agent encapsulates instructions + a set of tools, and can hand off to
    another Agent at any point."""

    def __init__(
        self,
        name: str = "Agent",
        model: Optional[str] = None,
        instructions: Union[str, Callable[[Dict[str, Any]], str]] = "You are a helpful agent.",
        functions: Optional[List[Callable]] = None,
        handoffs: Optional[List[Union["Agent", Callable]]] = None,
        tool_choice: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.functions = functions or []
        self.handoffs = handoffs or []
        self.tool_choice = tool_choice

    def get_instructions(self, context_variables: Dict[str, Any]) -> str:
        if callable(self.instructions):
            return self.instructions(context_variables)
        return str(self.instructions)


class Response:
    """Result of a Swarm run. messages include a 'sender' field indicating which
    agent originated each message."""

    def __init__(
        self,
        messages: List[Dict[str, Any]],
        agent: Agent,
        context_variables: Dict[str, Any],
    ):
        self.messages = messages
        self.agent = agent
        self.context_variables = context_variables


class Swarm:
    """Stateless multi-agent runner. Implements the Swarm loop:

      1. Get completion from current Agent
      2. Execute tool calls and append results
      3. Switch Agent if a handoff occurred
      4. Update context variables
      5. Repeat until no new function calls or max_turns reached
    """

    def __init__(self, client=None, debug: bool = False):
        self.client = client
        self.debug = debug

    def _log(self, message: str):
        if self.debug:
            print(f"[Swarm] {message}")

    def run(
        self,
        agent: Agent,
        messages: List[Dict[str, Any]],
        context_variables: Optional[Dict[str, Any]] = None,
        max_turns: int = 10,
        stream: bool = False,
    ) -> Response:
        context_variables = dict(context_variables or {})
        messages = list(messages)
        current_agent = agent
        turn = 0

        # Build the system prompt from the current agent's instructions
        system_prompt = current_agent.get_instructions(context_variables)

        while turn < max_turns:
            turn += 1
            self._log(f"Turn {turn}: agent={current_agent.name}")

            # Construct request messages: system + history
            request_messages = [{"role": "system", "content": system_prompt}] + messages

            # Call the inference backend (Dark Matter Studio's stream_colibri_completion)
            # We collect the full response text and parse tool calls.
            from app import stream_colibri_completion, extract_all_tool_calls

            opts = {
                "loader": "Colibri",  # will be overridden by LOADER_STATE in app context
                "session_id": "swarm-" + str(uuid.uuid4().hex[:8]),
                "streamed_tool_calls_out": {},
                "metrics_out": {},
            }

            # Build options from LOADER_STATE if available
            try:
                from app import LOADER_STATE
                opts["loader"] = LOADER_STATE.get("loader", "Colibri")
                settings = dict(LOADER_STATE.get("settings") or {})
                opts.update(settings)
            except Exception:
                pass

            # Stream and collect the full assistant reply
            collected = ""
            streamed_tcs: Dict[int, Dict[str, str]] = {}
            metrics: Dict[str, Any] = {}
            opts["streamed_tool_calls_out"] = streamed_tcs
            opts["metrics_out"] = metrics

            try:
                for token in stream_colibri_completion(
                    request_messages, current_agent.model or "qwen3-coder-30b-a3b-colibri", opts
                ):
                    if token:
                        collected += token
            except Exception as exc:
                self._log(f"Inference error: {exc}")
                messages.append({
                    "role": "assistant",
                    "content": f"[Swarm inference error: {exc}]",
                    "sender": current_agent.name,
                })
                break

            # Parse tool calls from the reply
            tool_calls = extract_all_tool_calls(collected, list(streamed_tcs.values()))

            # Store assistant message
            assistant_msg = {
                "role": "assistant",
                "content": collected,
                "sender": current_agent.name,
            }
            messages.append(assistant_msg)

            if not tool_calls:
                # No tool calls — run is complete
                self._log(f"No tool calls on turn {turn}; run complete.")
                break

            # Execute tool calls in order
            handoff_agent: Optional[Agent] = None
            handoff_context: Dict[str, Any] = {}
            tool_result_texts: List[str] = []

            for tool_name, args in tool_calls:
                self._log(f"Executing tool: {tool_name} with args {args}")

                # Check if this is a handoff function
                handoff_result = self._check_handoff(tool_name, args, current_agent)
                if handoff_result is not None:
                    # handoff_result is (Agent, context_variables_dict)
                    handoff_agent, handoff_context = handoff_result
                    continue

                # Execute via Dark Matter Studio's dispatch_tool_call
                try:
                    canonical, target, result = dispatch_tool_call(tool_name, args or {})
                    result_text = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as exc:
                    result_text = json.dumps({"status": "error", "output": str(exc)}, ensure_ascii=False)

                tool_result_texts.append(f"[{canonical}] {result_text}")
                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "name": canonical,
                    "sender": current_agent.name,
                })

            # Append tool results as a user-turn summary so the agent sees them
            if tool_result_texts:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(tool_result_texts),
                    "sender": current_agent.name,
                })

            # Handle handoff
            if handoff_agent is not None:
                self._log(f"Handoff: {current_agent.name} -> {handoff_agent.name}")
                current_agent = handoff_agent
                system_prompt = current_agent.get_instructions(context_variables)
                context_variables.update(handoff_context)
                messages.append({
                    "role": "system",
                    "content": f"[Handoff to {current_agent.name}]",
                    "sender": "system",
                })
                continue

        return Response(messages=messages, agent=current_agent, context_variables=context_variables)

    def _check_handoff(
        self, tool_name: str, args: Dict[str, Any], current_agent: Agent
    ) -> Optional[tuple]:
        """Check if a tool call is a handoff to another agent.

        Handoff functions are named 'transfer_to_<AgentName>' or match an agent
        in current_agent.handoffs.
        """
        # Direct handoff via function name pattern
        if tool_name.startswith("transfer_to_"):
            target_name = tool_name[len("transfer_to_"):]
            for h in current_agent.handoffs:
                if isinstance(h, Agent) and h.name == target_name:
                    return (h, {})
                if callable(h) and not isinstance(h, Agent):
                    # It's a function — call it to see if it returns an Agent
                    try:
                        ret = h(**args) if args else h()
                        if isinstance(ret, Agent):
                            return (ret, {})
                        if isinstance(ret, Result) and ret.agent is not None:
                            return (ret.agent, ret.context_variables or {})
                    except Exception:
                        pass
            return None

        # Handoff via Result object from a regular tool call
        return None


def make_handoff_function(target_agent: Agent, context_variables: Optional[Dict[str, Any]] = None) -> Callable:
    """Create a handoff function that transfers to target_agent.

    Usage:
        def transfer_to_sales():
            return sales_agent

        # or via make_handoff_function:
        transfer_to_sales = make_handoff_function(sales_agent, {"department": "sales"})
    """
    cv = context_variables or {}

    def _handoff(**kwargs):
        return Result(value=f"Handed off to {target_agent.name}", agent=target_agent, context_variables=cv)

    _handoff.__name__ = f"transfer_to_{target_agent.name}"
    _handoff.__doc__ = f"Transfer to {target_agent.name}"
    return _handoff


def create_swarm_agent(
    name: str,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[str]] = None,
    handoff_agents: Optional[List[Agent]] = None,
) -> Agent:
    """Factory to create a Dark Matter Studio swarm agent with tool-name functions.

    `tools` is a list of Dark Matter Studio tool names (e.g. ["read_workspace_file", "grep_workspace"]).
    Each becomes a thin wrapper function that dispatch_tool_call can execute.
    """
    funcs: List[Callable] = []
    if tools:
        for tname in tools:
            def _tool_func(_tn=tname, **kwargs):
                # This is a placeholder; actual dispatch happens in Swarm.run via dispatch_tool_call
                return {"tool": _tn, "args": kwargs}
            _tool_func.__name__ = tname
            _tool_func.__doc__ = f"Dark Matter Studio tool: {_tn}"
            funcs.append(_tool_func)

    handoffs: List[Union[Agent, Callable]] = []
    if handoff_agents:
        for ha in handoff_agents:
            handoffs.append(make_handoff_function(ha))

    return Agent(
        name=name,
        model=model,
        instructions=instructions or f"You are {name}, a helpful agent in the Dark Matter Studio swarm.",
        functions=funcs,
        handoffs=handoffs,
    )
