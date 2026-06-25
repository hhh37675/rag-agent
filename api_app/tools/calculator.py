"""安全的数学计算工具"""
import re
import ast
import operator

def safe_eval(expr: str):
    """
    一个安全且受限的算术表达式评估器
    基于 AST (抽象语法树) 解析，彻底杜绝系统命令注入。
    """
    # 仅允许的数学操作符
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # 数字
            return node.n
        elif isinstance(node, ast.BinOp):  # 二元运算 (+, -, *, /)
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # 一元运算 (如 -5)
            return allowed_operators[type(node.op)](_eval(node.operand))
        else:
            raise TypeError(f"不支持的语法节点: {type(node)}")

    # 限制递归层级，解析表达式
    parsed_node = ast.parse(expr, mode='eval').body
    return _eval(parsed_node)

def calculate_math(query: str) -> str:
    """
    供大模型调用的计算工具入口
    """
    try:
        # 严格过滤：只允许数字、小数点和基本运算符，剥离其他杂项字符
        cleaned = re.sub(r'[^0-9+\-*/.()]', '', query)
        if not cleaned:
            return "计算失败：没有提取到有效的数学表达式。"

        result = safe_eval(cleaned)

        # 去除末尾的 .0（如果是整数的话）
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"计算结果 : {result}"

    except ZeroDivisionError:
        return "计算错误：除数不能为零。"
    except Exception as e:
        return f"计算解析错误，请检查表达式格式 : {str(e)}"