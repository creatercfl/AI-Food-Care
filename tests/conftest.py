"""pytest 收集用例前先应用与 _bootstrap 相同的导入引导。

（使用标准库 unittest 运行时可忽略本文件；unittest 用例已在各自文件顶部
 显式 `import _bootstrap`。二者等效，保证 pytest / unittest 都能离线运行。）
"""
import _bootstrap  # noqa: F401
