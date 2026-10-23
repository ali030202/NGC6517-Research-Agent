# skills/agent_ideas.py
"""
阶段1: 假设生成智能体 — 仿效 AI Scientist 的 generate_ideas.py。
基于训练日志 + 领域知识库 + 模型代码，LLM 批量生成改进假设，并做新颖性查重。
"""
import json
import os
from typing import Annotated, List, Dict
from pydantic import Field

from skills.llm_client import (
    create_llm_client,
    get_response_from_llm,
    extract_json_between_markers,
)

SKILL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(SKILL_ROOT, "train_log.json")
KNOWLEDGE_FILE = os.path.join(SKILL_ROOT, "model_knowledge.md")
MODEL_CODE_PATH = os.path.join(SKILL_ROOT, "model_proto.py")

# ==================== Prompt 模板 ====================

IDEA_SYSTEM_PROMPT = """你是一名资深的深度学习研究员，专注于脉冲星信号分类和少样本学习领域。
你的任务是为 NGC6517 脉冲星小样本分类原型网络提出创新性的改进方案。
你需要基于当前模型结构、历史实验数据和领域知识，生成具有研究价值的改进假设。
每个假设必须包含具体的代码修改方案，可以直接落地实现。"""

IDEA_FIRST_PROMPT = """以下是当前 NGC6517 脉冲星小样本分类项目的完整上下文：

## 1. 当前模型结构代码 (model_proto.py)
```python
{model_code}
```

## 2. 历史实验数据
{train_logs}

## 3. 领域知识库摘要
{knowledge}

## 4. 当前最佳实验结果
- 最佳 Validation F1: {best_f1}
- 目标 F1: {target_f1}
- 差距: {gap}

## 5. 已生成的假设（避免重复）
{prev_ideas_string}

---

请生成下一个有创意的研究假设，用于改进原型网络 NGC6517Encoder 以提升分类 F1 分数。

注意：
- 必须针对 model_proto.py 中的 NGC6517Encoder 类提出具体代码修改
- 不要过度拟合特定数据集，确保方案有普适意义
- 修改方案需要可行，不能引入不存在的外部资源

按以下格式回复：

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

在 <THOUGHT> 中，简要讨论你的直觉和动机、高层设计方案、必要的设计选择和预期实验结果。
说明你的假设与已有假设的区别。

在 <JSON> 中，提供以下字段：
- "Name": 假设简称，小写+下划线，如 "attention_fusion"
- "Title": 假设标题
- "Experiment": 具体实现方案，说明需要修改哪些函数/类、如何修改
- "Interestingness": 1-10 评分
- "Feasibility": 1-10 评分
- "Novelty": 1-10 评分

请确保 JSON 格式精确，可被自动解析。
你将有 {num_reflections} 轮迭代来优化假设，不需要全部使用。"""

IDEA_REFLECTION_PROMPT = """第 {current_round}/{num_reflections} 轮。
请仔细审视你刚创建的假设的质量、新颖性和可行性。
在下一轮尝试中，改进和优化你的假设。
坚持原始假设的核心思路，除非存在明显问题。

按同样格式回复：
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

如果没有需要改进的地方，在 THOUGHT 末尾加上 "I am done"，然后原样输出之前的 JSON。"""

NOVELTY_SYSTEM_PROMPT = """你是一名有抱负的 AI 研究员，希望发表一篇对领域有显著贡献的论文。
你有一个假设，需要检查它是否新颖——即是否与已有文献大幅重叠或已被充分探索。
对新颖性要求严格，确保该假设对会议或研讨会论文有足够的贡献。"""

NOVELTY_PROMPT = """第 {current_round}/{num_rounds} 轮。
你有以下假设：

\"\"\"
{idea}
\"\"\"

上一轮搜索到的论文（第一轮为空）：
\"\"\"
{last_query_results}
\"\"\"

按以下格式回复：

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

在 <THOUGHT> 中，简要推理假设的新颖性，并确定搜索查询。
如果已做出决定，在 THOUGHT 中添加 "Decision made: novel." 或 "Decision made: not novel."。

在 <JSON> 中，仅包含以下字段：
- "Query": 文献搜索关键词（如 "attention mechanism few-shot prototype"）。
  如果尚未做出决定，必须提供一个搜索查询。"""


# ==================== 核心函数 ====================

def _load_context() -> dict:
    """加载项目上下文：训练日志、知识库、模型代码。"""
    ctx = {"train_logs": "暂无训练记录", "knowledge": "", "model_code": ""}

    from skills.train_tools import get_train_logs
    logs = get_train_logs()
    if logs:
        ctx["train_logs"] = json.dumps(logs, indent=2, ensure_ascii=False)

    # 知识库（截取前 2000 字符，避免过长）
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                ctx["knowledge"] = f.read()[:2000]
    except Exception:
        pass

    # 模型代码
    try:
        if os.path.exists(MODEL_CODE_PATH):
            with open(MODEL_CODE_PATH, "r", encoding="utf-8") as f:
                ctx["model_code"] = f.read()
    except Exception:
        pass

    return ctx


def _get_best_f1() -> float:
    from skills.train_tools import evaluate_progress
    return evaluate_progress()["best_f1"]


def generate_ideas(
    target_f1: Annotated[float, Field(description="目标F1分数", ge=0.0, le=1.0)] = 0.9,
    max_num_ideas: Annotated[int, Field(description="生成假设数量", ge=1, le=20)] = 5,
    num_reflections: Annotated[int, Field(description="每条假设的反思迭代轮数", ge=1, le=5)] = 3,
) -> dict:
    """
    基于训练日志 + 知识库 + 模型代码，LLM 批量生成改进假设。
    返回结构化假设列表，每条包含 Name, Title, Experiment, 评分等字段。
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"error": f"LLM 客户端创建失败: {str(e)}"}

    ctx = _load_context()
    best_f1 = _get_best_f1()
    gap = round(target_f1 - best_f1, 4)

    idea_archive = []
    idea_str_archive = []

    for gen_idx in range(max_num_ideas):
        print(f"[IDEA] 生成假设 {gen_idx + 1}/{max_num_ideas}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive) if idea_str_archive else "暂无"
            msg_history = []

            # 第一轮生成
            text, msg_history = get_response_from_llm(
                msg=IDEA_FIRST_PROMPT.format(
                    model_code=ctx["model_code"],
                    train_logs=ctx["train_logs"],
                    knowledge=ctx["knowledge"],
                    best_f1=best_f1,
                    target_f1=target_f1,
                    gap=gap,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=IDEA_SYSTEM_PROMPT,
                msg_history=msg_history,
            )

            json_output = extract_json_between_markers(text)
            if json_output is None:
                print(f"[IDEA] 第 {gen_idx + 1} 条假设 JSON 解析失败，跳过")
                continue

            print(f"[IDEA] 初始假设: {json_output.get('Name', 'N/A')}")

            # 反思迭代
            if num_reflections > 1:
                for ref_round in range(num_reflections - 1):
                    text, msg_history = get_response_from_llm(
                        msg=IDEA_REFLECTION_PROMPT.format(
                            current_round=ref_round + 2,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=IDEA_SYSTEM_PROMPT,
                        msg_history=msg_history,
                    )
                    new_json = extract_json_between_markers(text)
                    if new_json is not None:
                        json_output = new_json
                    if "I am done" in text:
                        print(f"[IDEA] 假设在第 {ref_round + 2} 轮收敛")
                        break

            idea_archive.append(json_output)
            idea_str_archive.append(json.dumps(json_output, ensure_ascii=False))
            print(f"[IDEA] 已生成: {json_output.get('Name')} | "
                  f"Interestingness={json_output.get('Interestingness')} "
                  f"Feasibility={json_output.get('Feasibility')} "
                  f"Novelty={json_output.get('Novelty')}")

        except Exception as e:
            print(f"[IDEA] 生成假设 {gen_idx + 1} 失败: {e}")
            continue

    # 保存到 ideas.json
    ideas_path = os.path.join(SKILL_ROOT, "ideas.json")
    try:
        with open(ideas_path, "w", encoding="utf-8") as f:
            json.dump(idea_archive, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return {
        "status": "success",
        "total_generated": len(idea_archive),
        "target_f1": target_f1,
        "current_best_f1": best_f1,
        "gap": gap,
        "ideas": idea_archive,
    }


def check_idea_novelty(
    ideas: Annotated[list, Field(description="待检查的假设列表")],
    max_rounds: Annotated[int, Field(description="新颖性检查最大轮数", ge=1, le=10)] = 5,
) -> dict:
    """
    用 ArXiv 搜索做新颖性查重，为每条假设标记 novel=True/False。
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"error": f"LLM 客户端创建失败: {str(e)}"}

    # 导入 ArXiv 搜索（复用已有工具）
    try:
        from skills.paper_tools import search_arxiv_paper
    except ImportError:
        return {"error": "无法导入 search_arxiv_paper"}

    results = []
    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            results.append(idea)
            continue

        print(f"[NOVELTY] 检查假设 {idx}: {idea.get('Name', 'N/A')}")
        novel = False
        msg_history = []
        papers_str = ""

        for round_idx in range(max_rounds):
            try:
                text, msg_history = get_response_from_llm(
                    msg=NOVELTY_PROMPT.format(
                        current_round=round_idx + 1,
                        num_rounds=max_rounds,
                        idea=json.dumps(idea, ensure_ascii=False),
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=NOVELTY_SYSTEM_PROMPT,
                    msg_history=msg_history,
                )

                if "decision made: novel" in text.lower():
                    novel = True
                    print(f"[NOVELTY] 第 {round_idx + 1} 轮判定: novel")
                    break
                if "decision made: not novel" in text.lower():
                    print(f"[NOVELTY] 第 {round_idx + 1} 轮判定: not novel")
                    break

                # 解析搜索查询
                json_out = extract_json_between_markers(text)
                if json_out and "Query" in json_out:
                    query = json_out["Query"]
                    papers = search_arxiv_paper(topic=query, max_count=5)
                    if papers and not (isinstance(papers[0], dict) and "error" in papers[0]):
                        paper_lines = []
                        for i, p in enumerate(papers):
                            paper_lines.append(
                                f"{i}: {p.get('title','')} ({p.get('publish_date','')})\n"
                                f"   Abstract: {p.get('abstract','')[:200]}"
                            )
                        papers_str = "\n\n".join(paper_lines)
                    else:
                        papers_str = "No papers found."
            except Exception as e:
                print(f"[NOVELTY] 第 {round_idx + 1} 轮异常: {e}")
                continue

        idea["novel"] = novel
        results.append(idea)

    # 保存更新后的假设
    ideas_path = os.path.join(SKILL_ROOT, "ideas.json")
    try:
        with open(ideas_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    novel_count = sum(1 for i in results if i.get("novel", False))
    return {
        "status": "success",
        "total_ideas": len(results),
        "novel_count": novel_count,
        "ideas": results,
    }
