"""Simulate tool responses for the W13 routing agent's Path C (TIER_3) workflow.

Provide mock tool handlers for web_search, calculator, and unit_converter so
the harness can resolve tool calls without hitting external services.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate (e.g. '2 + 3 * 4')",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert a numeric value from one unit to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit (e.g. 'km', 'C', 'kg')",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit (e.g. 'mi', 'F', 'lb')",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, callable] = {}  # populated at module bottom


def simulate_tool_call(tool_name: str, arguments: dict) -> str:
    """Dispatch *tool_name* to the appropriate handler and return a JSON string."""
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        log.warning("Unknown tool requested: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = handler(**arguments)
        return json.dumps({"result": result})
    except Exception as exc:  # noqa: BLE001
        log.debug("Tool %s raised %s: %s", tool_name, type(exc).__name__, exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: list[tuple[list[str], str]] = [
    (
        ["usd", "eur", "exchange rate", "dollar euro"],
        "The current USD to EUR exchange rate is approximately 0.92 EUR per 1 USD.",
    ),
    (
        ["usd", "gbp", "exchange rate", "dollar pound"],
        "The current USD to GBP exchange rate is approximately 0.79 GBP per 1 USD.",
    ),
    (
        ["eur", "gbp", "exchange rate"],
        "The current EUR to GBP exchange rate is approximately 0.86 GBP per 1 EUR.",
    ),
    (
        ["weather", "new york"],
        "New York currently has partly cloudy skies with a temperature of 22°C (72°F).",
    ),
    (
        ["weather", "london"],
        "London is experiencing light rain with a temperature of 14°C (57°F).",
    ),
    (
        ["weather", "tokyo"],
        "Tokyo is sunny with a temperature of 28°C (82°F) and moderate humidity.",
    ),
    (
        ["population", "world"],
        "The current world population is approximately 8.1 billion people.",
    ),
    (
        ["population", "united states", "usa", "us population"],
        "The population of the United States is approximately 335 million people.",
    ),
    (
        ["population", "china"],
        "China's population is approximately 1.41 billion people.",
    ),
    (
        ["population", "india"],
        "India's population is approximately 1.44 billion people, making it the most populous country.",
    ),
    (
        ["speed of light", "light speed"],
        "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
    ),
    (
        ["distance", "earth", "moon"],
        "The average distance from the Earth to the Moon is about 384,400 km (238,855 miles).",
    ),
    (
        ["distance", "earth", "sun"],
        "The average distance from the Earth to the Sun is about 149.6 million km (93 million miles).",
    ),
    (
        ["python", "programming language"],
        "Python is a high-level interpreted programming language created by Guido van Rossum, first released in 1991.",
    ),
    (
        ["javascript", "programming language"],
        "JavaScript is a dynamic programming language primarily used for web development, standardized as ECMAScript.",
    ),
    (
        ["openai", "chatgpt", "gpt"],
        "OpenAI is an AI research company that created the GPT series of large language models, including ChatGPT.",
    ),
    (
        ["anthropic", "claude"],
        "Anthropic is an AI safety company that develops the Claude family of large language models.",
    ),
    (
        ["boiling point", "water"],
        "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm).",
    ),
    (
        ["pi", "mathematical constant"],
        "Pi is approximately 3.14159265358979. It represents the ratio of a circle's circumference to its diameter.",
    ),
    (
        ["largest country", "area"],
        "Russia is the largest country in the world by area, covering about 17.1 million square kilometers.",
    ),
]


def _web_search(query: str) -> str:
    """Return a canned response matching the query against the knowledge base."""
    query_lower = query.lower()
    best_match: str | None = None
    best_score = 0

    for keywords, response in _KNOWLEDGE_BASE:
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_match = response

    if best_match is not None and best_score > 0:
        log.debug("web_search matched query %r (score=%d)", query, best_score)
        return best_match

    return (
        f"No specific results found for '{query}'. "
        "Try rephrasing your search with more specific keywords."
    )


# ---------------------------------------------------------------------------
# calculator — AST-based safe evaluator
# ---------------------------------------------------------------------------

# Allowed binary operators
_BIN_OPS: dict[type, callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: dict[type, callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed function names mapped to callables
_SAFE_FUNCTIONS: dict[str, callable] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
    "min": min,
    "max": max,
}

# Allowed constant names
_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _safe_eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node, allowing only safe numeric operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Guard against excessively large exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
            raise ValueError(f"Exponent too large: {right}")
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are supported")
        func_name = node.func.id
        func = _SAFE_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return func(*args)

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        raise ValueError(f"Unknown name: {name}")

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression via AST walking."""
    # Normalise whitespace and strip
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty expression")

    # Replace common symbols that users might type
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc

    result = _safe_eval_node(tree)

    # Format: drop trailing .0 for clean integer results
    if isinstance(result, float) and result == int(result) and math.isfinite(result):
        return str(int(result))
    return str(result)


# ---------------------------------------------------------------------------
# unit_converter
# ---------------------------------------------------------------------------

# Conversion table: (from_unit, to_unit) -> callable(value) -> converted value
# Each pair is registered in both directions.
_CONVERSIONS: dict[tuple[str, str], callable] = {}


def _register_pair(
    a: str, b: str, a_to_b: callable, b_to_a: callable
) -> None:
    """Register a bidirectional conversion."""
    _CONVERSIONS[(a, b)] = a_to_b
    _CONVERSIONS[(b, a)] = b_to_a


# Distance
_register_pair("km", "mi", lambda v: v * 0.621371, lambda v: v * 1.60934)
_register_pair("m", "ft", lambda v: v * 3.28084, lambda v: v * 0.3048)

# Temperature
_register_pair("C", "F", lambda v: v * 9 / 5 + 32, lambda v: (v - 32) * 5 / 9)

# Mass
_register_pair("kg", "lb", lambda v: v * 2.20462, lambda v: v * 0.453592)

# Currency (approximate)
_register_pair("USD", "EUR", lambda v: v * 0.92, lambda v: v / 0.92)

# Volume
_register_pair("L", "gal", lambda v: v * 0.264172, lambda v: v * 3.78541)

# Speed
_register_pair("km_h", "mph", lambda v: v * 0.621371, lambda v: v * 1.60934)

# Normalisation map for common alternative spellings
_UNIT_ALIASES: dict[str, str] = {
    "kilometers": "km",
    "kilometer": "km",
    "miles": "mi",
    "mile": "mi",
    "meters": "m",
    "meter": "m",
    "feet": "ft",
    "foot": "ft",
    "celsius": "C",
    "fahrenheit": "F",
    "kilograms": "kg",
    "kilogram": "kg",
    "pounds": "lb",
    "pound": "lb",
    "liters": "L",
    "liter": "L",
    "litres": "L",
    "litre": "L",
    "gallons": "gal",
    "gallon": "gal",
    "kph": "km_h",
    "km/h": "km_h",
    "kmh": "km_h",
}


def _normalise_unit(unit: str) -> str:
    """Resolve a unit string to its canonical form."""
    return _UNIT_ALIASES.get(unit.lower(), unit)


def _unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert *value* from *from_unit* to *to_unit* using the hardcoded table."""
    src = _normalise_unit(from_unit)
    dst = _normalise_unit(to_unit)

    if src == dst:
        return f"{value} {from_unit} = {value} {to_unit}"

    converter = _CONVERSIONS.get((src, dst))
    if converter is None:
        supported = sorted({f"{a}->{b}" for a, b in _CONVERSIONS})
        raise ValueError(
            f"Unsupported conversion: {from_unit} -> {to_unit}. "
            f"Supported pairs: {', '.join(supported)}"
        )

    result = converter(value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------

_HANDLERS.update(
    {
        "web_search": _web_search,
        "calculator": _calculator,
        "unit_converter": _unit_converter,
    }
)
