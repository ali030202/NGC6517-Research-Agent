# skills/code_generator.py
"""
从零代码生成器 — 无模板模式的核心组件。
LLM 根据项目上下文从零生成 model_proto.py 模型架构代码。
约束：必须保留 NGC6517Encoder 类名、forward(batch) -> Tensor 接口、
      prototypical_logits 和 deterministic_multi_prototypes 函数签名。
train_fewshot.py 和 dataset_ngc6517.py 保持不变。
"""
import json
import os
import re
from typing import Optional

from skills.llm_client import (
    create_llm_client,
    get_response_from_llm,
    quick_llm_call,
)

SKILL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CODE_PATH = os.path.join(SKILL_ROOT, "model_proto.py")
KNOWLEDGE_FILE = os.path.join(SKILL_ROOT, "model_knowledge.md")

# ==================== 接口约束模板 ====================

INTERFACE_CONSTRAINTS = """
## 必须遵守的接口约束（不可更改）

train_fewshot.py 通过以下方式导入和使用模型：
```python
from model_proto import NGC6517Encoder, prototypical_logits

# 创建模型
model = NGC6517Encoder(embed_dim=args.embed_dim, dropout=args.dropout, use_prior=not args.no_prior)

# 前向传播（batch 是字典，包含 profile, dm, image, meta, aux 键）
z = model(batch)  # -> Tensor [B, embed_dim]

# 原型网络 logits 计算
logits = prototypical_logits(z_support, y_support, z_query, temperature, psr_k, rfi_k)
```

### 必须保留的签名：
1. `class NGC6517Encoder(nn.Module)` — 构造函数: `__init__(self, embed_dim=128, dropout=0.1, use_prior=True)`
2. `forward(self, batch) -> torch.Tensor` — batch 字典键: profile(float,[B,1,256]), dm(float,[B,1,256]), image(float,[B,2,32,32]), meta(float,[B,5]), aux(float,[B,27])
3. `def prototypical_logits(z_support, y_support, z_query, temperature=10.0, psr_k=1, rfi_k=1) -> torch.Tensor`
4. `def deterministic_multi_prototypes(z, k) -> torch.Tensor`

### 可用依赖（不可引入其他）：
- torch, torch.nn, torch.nn.functional

### 输出要求：
- 输出完整的 model_proto.py 代码（从 import 到最后一个函数）
- 代码必须语法正确，可直接被 Python 解释器执行
"""

# ==================== Prompt 模板 ====================

GENERATE_SYSTEM_PROMPT = """你是一名资深深度学习架构师，精通 PyTorch 和脉冲星信号处理。
你需要为 NGC6517 脉冲星/RFI 小样本分类任务设计全新的模型架构。
你的设计必须有创新性，同时保证代码可以直接运行。"""

GENERATE_PROMPT = """## 任务描述
为 NGC6517 脉冲星小样本分类设计一个**全新的模型架构**。

## 项目上下文
- 数据集：NGC6517 球状星团脉冲星观测数据
- 任务：二分类（脉冲星 PSR vs 射频干扰 RFI），小样本场景
- 训练方式：原型网络（Prototypical Network）+ episodic training
- 每个 episode: support set (k-shot per class) + query set

## 输入数据格式
- profile: [B, 1, 256] — 脉冲轮廓（1D时间序列）
- dm: [B, 1, 256] — DM 曲线（色散测量）
- image: [B, 2, 32, 32] — 子积分/子频带图像
- meta: [B, 5] — 观测元数据
- aux: [B, 27] — 辅助特征（含 profile统计9 + DM统计8 + 图像统计10）

## 领域知识摘要
{knowledge}

## 设计方向建议
{design_hints}

## 接口约束
{constraints}

请设计一个创新的模型架构，输出完整代码。使用以下格式：

DESIGN_RATIONALE:
<设计理由和创新点>

COMPLETE_CODE:
```python
<完整的 model_proto.py 代码>
```"""

MODIFY_PROMPT = """## 当前模型代码
```python
{current_code}
```

## 训练结果
- 当前 F1: {current_f1}
- 最佳 F1: {best_f1}

## 改进方向
{improvement_direction}

## 接口约束
{constraints}

请基于当前代码进行**大幅度架构改进**（不是微调参数），输出完整代码。使用以下格式：

DESIGN_RATIONALE:
<改进理由>

COMPLETE_CODE:
```python
<完整的 model_proto.py 代码>
```"""

# ==================== 辅助函数 ====================

def _load_knowledge() -> str:
    """加载领域知识文件。"""
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                text = f.read()
            return text[:3000]  # 截断避免过长
    except Exception:
        pass
    return "（领域知识文件不可用）"


def _extract_code_block(llm_output: str) -> str:
    """从 LLM 输出中提取代码块。"""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, llm_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    pattern = r"```\n(.*?)```"
    matches = re.findall(pattern, llm_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def _validate_code(code: str) -> tuple:
    from ngc_agent.code import validate_model
    try:
        validate_model(code)
        return True, "Static syntax/interface checks passed; runtime behavior is unverified"
    except (ValueError, SyntaxError) as exc:
        return False, str(exc)


# ==================== 核心函数 ====================

def generate_model_from_scratch(
    design_hints: str = "",
    max_retries: int = 2,
) -> dict:
    """
    从零生成 model_proto.py 代码。

    Args:
        design_hints: 设计方向提示（可选，如 "尝试注意力机制"）
        max_retries: 验证失败时的最大重试次数

    Returns:
        {
            "status": "success" | "failed",
            "code": str,           # 生成的完整代码
            "rationale": str,      # 设计理由
            "validation": str,     # 验证结果
        }
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"status": "failed", "error": f"LLM 客户端创建失败: {e}"}

    knowledge = _load_knowledge()
    constraints = INTERFACE_CONSTRAINTS

    for attempt in range(max_retries + 1):
        print(f"[CODEGEN] 尝试生成 {attempt + 1}/{max_retries + 1}...")

        prompt = GENERATE_PROMPT.format(
            knowledge=knowledge,
            design_hints=design_hints or "自由发挥，追求高分类精度",
            constraints=constraints,
        )

        try:
            text, _ = get_response_from_llm(
                msg=prompt,
                client=client,
                model=model,
                system_message=GENERATE_SYSTEM_PROMPT,
                temperature=0.85,
            )
        except Exception as e:
            return {"status": "failed", "error": f"LLM 调用失败: {e}"}

        code = _extract_code_block(text)
        if not code:
            print(f"[CODEGEN] 未找到代码块，重试...")
            continue

        # 提取设计理由
        rationale = ""
        rationale_match = re.search(r"DESIGN_RATIONALE:\s*(.*?)(?=COMPLETE_CODE:|$)", text, re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        # 验证
        is_valid, msg = _validate_code(code)
        if is_valid:
            print(f"[CODEGEN] 生成成功，代码验证通过")
            return {
                "status": "success",
                "code": code,
                "rationale": rationale,
                "validation": msg,
            }
        else:
            print(f"[CODEGEN] 验证失败: {msg}")

    return {"status": "failed", "error": f"生成 {max_retries + 1} 次均未通过验证"}


def generate_model_variant(
    current_code: str,
    current_f1: float,
    improvement_direction: str,
    max_retries: int = 2,
) -> dict:
    """
    基于当前代码生成架构变体（用于 BFTS 树搜索的节点展开）。

    Args:
        current_code: 当前 model_proto.py 完整代码
        current_f1: 当前模型的 F1 分数
        improvement_direction: 改进方向描述
        max_retries: 最大重试次数

    Returns:
        同 generate_model_from_scratch
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"status": "failed", "error": f"LLM 客户端创建失败: {e}"}

    best_f1 = current_f1  # 简化：使用传入的 F1
    constraints = INTERFACE_CONSTRAINTS

    for attempt in range(max_retries + 1):
        prompt = MODIFY_PROMPT.format(
            current_code=current_code,
            current_f1=current_f1,
            best_f1=best_f1,
            improvement_direction=improvement_direction,
            constraints=constraints,
        )

        try:
            text, _ = get_response_from_llm(
                msg=prompt,
                client=client,
                model=model,
                system_message=GENERATE_SYSTEM_PROMPT,
                temperature=0.9,
            )
        except Exception as e:
            return {"status": "failed", "error": f"LLM 调用失败: {e}"}

        code = _extract_code_block(text)
        if not code:
            continue

        rationale = ""
        rationale_match = re.search(r"DESIGN_RATIONALE:\s*(.*?)(?=COMPLETE_CODE:|$)", text, re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        is_valid, msg = _validate_code(code)
        if is_valid:
            return {
                "status": "success",
                "code": code,
                "rationale": rationale,
                "validation": msg,
            }

    return {"status": "failed", "error": f"变体生成 {max_retries + 1} 次均未通过验证"}


def generate_architecture_ideas(
    current_code: str,
    current_f1: float,
    num_ideas: int = 3,
) -> list:
    """
    让 LLM 提出多个不同的架构改进方向（用于 BFTS 分支生成）。

    Returns:
        [{"direction": str, "description": str, "expected_benefit": str}, ...]
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return [{"error": f"LLM 客户端创建失败: {e}"}]

    prompt = f"""## 当前模型架构代码（摘要）
```python
{current_code[:2000]}
```

## 当前性能
- F1: {current_f1}

## 任务
提出 {num_ideas} 个**不同方向**的架构改进方案。每个方案应该尝试不同的技术路线。

可能的方向包括但不限于：
- 注意力机制（self-attention, cross-attention）
- 对比学习辅助任务
- 多尺度特征融合策略
- 残差连接变体
- 频域/时频域特征提取
- 图神经网络
- 自适应温度/度量学习

请用 JSON 格式输出：
```json
[
  {{
    "direction": "简短方向名称",
    "description": "具体改进描述（2-3句）",
    "expected_benefit": "预期收益"
  }}
]
```"""

    try:
        text, _ = get_response_from_llm(
            msg=prompt,
            client=client,
            model=model,
            system_message="你是深度学习架构师，擅长设计创新的神经网络模型。",
            temperature=0.95,
        )

        from skills.llm_client import extract_json_between_markers
        ideas = extract_json_between_markers(text)
        if isinstance(ideas, list):
            return ideas
        # 尝试提取 JSON 数组
        arr_match = re.search(r"\[.*\]", text, re.DOTALL)
        if arr_match:
            return json.loads(arr_match.group())
        return [{"direction": "自由探索", "description": text[:200], "expected_benefit": "未知"}]

    except Exception as e:
        return [{"error": str(e)}]
