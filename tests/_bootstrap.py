"""单元测试引导模块：让 agent 子模块可离线导入（不触发 Chroma、不需真实 API Key）。

背景：`src/agent/__init__.py` 在导入时会实例化 EntityAlignment（打开 Chroma 向量库）
并构造 ChatDeepSeek（校验 DEEPSEEK_API_KEY）。这会带来两个问题：
  1) 与正在运行的后端争用向量库，违反“单实例铁律”，可能损坏 HNSW 索引；
  2) 无真实 Key 时直接导入失败。

本模块在导入任何 `agent.*` 之前做两件事：
  - 注入占位 DEEPSEEK_API_KEY：ChatDeepSeek 构造期只校验非空，不发起网络请求；
  - 预置一个 agent 包壳到 sys.modules：使 `import agent.guardrail` 等只加载子模块，
    跳过真实 `agent/__init__.py`，从而不打开 Chroma。

约定：所有测试文件必须**最先** `import _bootstrap`，再导入被测模块。
本方案对 unittest 与 pytest 均适用。
"""
import os
import sys
import types

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
SRC_DIR = os.path.join(_ROOT_DIR, "src")

# 占位 Key：仅供 ChatDeepSeek 构造期非空校验；单测不会真正调用大模型。
# 若系统环境已有真实 Key，setdefault 不覆盖。
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-dummy-for-unit-tests")

# 让 `from configuration import config` / `from agent.xxx import ...` 可解析
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 预置 agent 包壳，绕过真实 agent/__init__.py（避免打开 Chroma 向量库）
if "agent" not in sys.modules:
    _agent_pkg = types.ModuleType("agent")
    _agent_pkg.__path__ = [os.path.join(SRC_DIR, "agent")]
    sys.modules["agent"] = _agent_pkg
