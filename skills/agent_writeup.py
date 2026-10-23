# skills/agent_writeup.py
"""
阶段3: 论文撰写智能体 — 仿效 AI Scientist 的 perform_writeup.py。
读取全部实验日志和图表，调用 LLM 逐章节生成 Markdown 论文，并自动补充参考文献。
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

WRITEUP_SYSTEM_PROMPT = """你是一名专业的学术论文撰写助手，擅长撰写深度学习和小样本学习领域的研究论文。
你需要基于实验数据和模型信息，撰写一篇结构完整、逻辑清晰的学术论文。
论文使用 Markdown 格式输出，语言风格严谨、专业。"""

WRITEUP_ABSTRACT_PROMPT = """请为以下研究撰写论文摘要（Abstract），200-300字。

## 研究背景
NGC6517 脉冲星信号小样本分类，使用原型网络（Prototypical Network）进行少样本学习。

## 核心方法
{method_summary}

## 实验结果摘要
{experiment_summary}

## 最佳实验配置与指标
{best_config}

请撰写一段精炼的摘要，涵盖研究动机、方法概述、主要实验结论。
直接输出摘要文本，不需要标题。"""

WRITEUP_INTRO_PROMPT = """请撰写论文的 Introduction 部分（引言），500-800字。

## 研究领域
脉冲星信号分类，小样本学习

## 研究问题
NGC6517 球状星团中脉冲星信号样本稀少，传统深度学习方法难以有效分类。

## 方法概述
{method_summary}

## 主要贡献
1. 提出了多模态原型网络 NGC6517Encoder
2. 实现了脉冲星信号的小样本分类
3. 通过自动科研迭代优化模型性能

## 实验最佳结果
最佳 Validation F1: {best_f1}

请撰写引言，包含：研究背景、现有方法的不足、本文方法的优势、论文结构概述。
使用 Markdown 格式，一级标题为 "# 1. Introduction"。"""

WRITEUP_RELATED_WORK_PROMPT = """请撰写论文的 Related Work 部分（相关工作），400-600字。

## 领域知识库参考
{knowledge}

## 检索到的相关论文
{papers}

请撰写相关工作部分，涵盖：
1. 脉冲星信号检测与分类的传统方法
2. 小样本学习（Few-shot Learning）
3. 原型网络（Prototypical Networks）及其变体
4. 多模态特征融合方法

使用 Markdown 格式，一级标题为 "# 2. Related Work"。
在适当位置用 [1], [2] 等引用标记引用参考文献。"""

WRITEUP_METHOD_PROMPT = """请撰写论文的 Method 部分（方法），600-1000字。

## 模型代码
```python
{model_code}
```

## 方法概述
{method_summary}

请详细描述 NGC6517Encoder 原型网络的方法，包括：
1. 整体架构概述
2. 多模态编码器设计（profile、dm、image、meta 四路编码）
3. 特征融合策略
4. 原型计算与分类机制
5. 训练策略（episodic training）

使用 Markdown 格式，一级标题为 "# 3. Method"。
可以适当使用数学公式（LaTeX 格式）描述关键计算过程。"""

WRITEUP_EXPERIMENTS_PROMPT = """请撰写论文的 Experiments 部分（实验），600-800字。

## 全部实验记录
{experiment_logs}

## 最佳实验配置
{best_config}

## 最佳实验结果
- Validation F1: {best_f1}
- 目标 F1: {target_f1}

请撰写实验部分，包含：
1. 数据集描述（NGC6517 脉冲星数据）
2. 实验设置（超参数范围、评估指标）
3. 实验结果表格（Markdown 表格）
4. 结果分析与讨论
5. 消融实验讨论（如有多组实验数据）

使用 Markdown 格式，一级标题为 "# 4. Experiments"。
包含至少一个 Markdown 表格展示实验结果对比。"""

WRITEUP_CONCLUSION_PROMPT = """请撰写论文的 Conclusion 部分（结论），300-500字。

## 研究总结
- 方法: {method_summary}
- 最佳结果: Validation F1 = {best_f1}
- 目标: {target_f1}

## 实验关键发现
{key_findings}

请撰写结论部分，包含：
1. 研究主要成果总结
2. 方法的优势与局限性
3. 未来工作展望

使用 Markdown 格式，一级标题为 "# 5. Conclusion"。"""

CITATION_SYSTEM_PROMPT = """你是一名学术文献管理专家。
你需要为论文补充准确的参考文献列表。"""

CITATION_PROMPT = """以下是一篇论文的 Related Work 部分：

{related_work_text}

请为文中引用的文献生成参考文献列表。

## 可用的论文检索结果
{search_results}

## 领域知识库中的参考信息
{knowledge_refs}

请输出 Markdown 格式的参考文献列表，标题为 "# References"。
每条参考文献包含：编号、作者、标题、发表年份、来源（arxiv/会议/期刊）。
格式示例：
[1] Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""


# ==================== 辅助函数 ====================

def _load_experiment_logs() -> list:
    from skills.train_tools import get_train_logs
    return get_train_logs()


def _load_knowledge() -> str:
    """加载领域知识库。"""
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return f.read()[:3000]
    except Exception:
        pass
    return ""


def _load_model_code() -> str:
    """加载模型代码。"""
    try:
        if os.path.exists(MODEL_CODE_PATH):
            with open(MODEL_CODE_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""


def _summarize_experiments(logs: list) -> str:
    """生成实验摘要文本。"""
    if not logs:
        return "暂无实验数据"
    lines = []
    for i, log in enumerate(logs):
        f1 = log.get("val_f1", "N/A")
        lr = log.get("lr", "N/A")
        embed = log.get("embed_dim", "N/A")
        ep = log.get("episodes", "N/A")
        lines.append(f"实验 {i+1}: lr={lr}, embed_dim={embed}, episodes={ep}, val_f1={f1}")
    return "\n".join(lines)


def _get_best_experiment(logs: list) -> dict:
    """获取最佳实验记录。"""
    if not logs:
        return {}
    valid = [x for x in logs if "val_f1" in x]
    if not valid:
        return {}
    return max(valid, key=lambda x: x["val_f1"])


def _search_related_papers(keywords: list) -> str:
    """搜索相关论文用于参考文献。"""
    try:
        from skills.paper_tools import search_arxiv_paper
    except ImportError:
        return "无法导入论文搜索工具"

    all_papers = []
    for kw in keywords[:3]:
        try:
            papers = search_arxiv_paper(topic=kw, max_count=3)
            if papers and not (isinstance(papers[0], dict) and "error" in papers[0]):
                all_papers.extend(papers)
        except Exception:
            continue

    if not all_papers:
        return "未检索到相关论文"

    lines = []
    for i, p in enumerate(all_papers):
        lines.append(
            f"[{i+1}] {p.get('title', 'N/A')} ({p.get('publish_date', 'N/A')})\n"
            f"    摘要: {p.get('abstract', '')[:200]}"
        )
    return "\n\n".join(lines)


# ==================== 核心函数 ====================

def perform_writeup(
    experiment_logs: Annotated[list, Field(description="实验日志列表，每条包含超参数和F1等指标")] = None,
    idea: Annotated[dict, Field(description="研究假设，包含 Title, Experiment 等字段")] = None,
) -> dict:
    """
    读取全部实验日志和模型代码，调用 LLM 逐章节生成 Markdown 论文。
    输出保存为 ngc6517_paper.md。
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"error": f"LLM 客户端创建失败: {str(e)}"}

    # 加载数据
    logs = experiment_logs if experiment_logs else _load_experiment_logs()
    if not logs:
        return {"status": "deferred", "reason": "没有真实实验记录；不能生成性能结论。"}
    knowledge = _load_knowledge()
    model_code = _load_model_code()
    best_exp = _get_best_experiment(logs)
    best_f1 = best_exp.get("val_f1", 0)
    if best_exp.get("job_id"):
        from ngc_agent.service import Agent
        source = Agent().run_directory(best_exp["job_id"]) / "source" / "model_proto.py"
        model_code = source.read_text(encoding="utf-8-sig")
    target_f1 = 0.9

    # 方法摘要
    method_summary = "多模态原型网络 NGC6517Encoder，集成 profile/dm/image/meta 四路编码器，" \
                     "通过特征融合层和原型计算实现脉冲星信号的小样本分类。"
    if idea and idea.get("Title"):
        method_summary += f"\n改进方案: {idea['Title']}"

    experiment_summary = _summarize_experiments(logs)
    best_config_str = json.dumps(best_exp, indent=2, ensure_ascii=False, default=str) if best_exp else "暂无"

    paper_sections = {}

    # --- Abstract ---
    print("[WRITEUP] 生成 Abstract...")
    try:
        abstract_text, _ = get_response_from_llm(
            msg=WRITEUP_ABSTRACT_PROMPT.format(
                method_summary=method_summary,
                experiment_summary=experiment_summary,
                best_config=best_config_str,
            ),
            client=client, model=model,
            system_message=WRITEUP_SYSTEM_PROMPT + "\n所有数值只能来自给定日志。验证指标必须标注验证集；没有独立测试就写未评估。引用需有可追溯来源；不得虚构文献或实验收益。",
        )
        paper_sections["abstract"] = f"# Abstract\n\n{abstract_text}"
    except Exception as e:
        paper_sections["abstract"] = f"# Abstract\n\n[生成失败: {str(e)}]"

    # --- Introduction ---
    print("[WRITEUP] 生成 Introduction...")
    try:
        intro_text, _ = get_response_from_llm(
            msg=WRITEUP_INTRO_PROMPT.format(
                method_summary=method_summary,
                best_f1=best_f1,
            ),
            client=client, model=model,
            system_message=WRITEUP_SYSTEM_PROMPT + "\n所有数值只能来自给定日志。验证指标必须标注验证集；没有独立测试就写未评估。引用需有可追溯来源；不得虚构文献或实验收益。",
        )
        paper_sections["introduction"] = intro_text
    except Exception as e:
        paper_sections["introduction"] = f"# 1. Introduction\n\n[生成失败: {str(e)}]"

    # --- Related Work ---
    print("[WRITEUP] 生成 Related Work...")
    search_keywords = [
        "few-shot learning pulsar classification",
        "prototypical network few-shot",
        "multi-modal feature fusion prototype",
    ]
    papers_text = _search_related_papers(search_keywords)

    try:
        related_text, _ = get_response_from_llm(
            msg=WRITEUP_RELATED_WORK_PROMPT.format(
                knowledge=knowledge[:1500],
                papers=papers_text,
            ),
            client=client, model=model,
            system_message=WRITEUP_SYSTEM_PROMPT + "\n所有数值只能来自给定日志。验证指标必须标注验证集；没有独立测试就写未评估。引用需有可追溯来源；不得虚构文献或实验收益。",
        )
        paper_sections["related_work"] = related_text
    except Exception as e:
        paper_sections["related_work"] = f"# 2. Related Work\n\n[生成失败: {str(e)}]"

    # --- Method ---
    print("[WRITEUP] 生成 Method...")
    try:
        method_text, _ = get_response_from_llm(
            msg=WRITEUP_METHOD_PROMPT.format(
                model_code=model_code[:4000],
                method_summary=method_summary,
            ),
            client=client, model=model,
            system_message=WRITEUP_SYSTEM_PROMPT + "\n所有数值只能来自给定日志。验证指标必须标注验证集；没有独立测试就写未评估。引用需有可追溯来源；不得虚构文献或实验收益。",
        )
        paper_sections["method"] = method_text
    except Exception as e:
        paper_sections["method"] = f"# 3. Method\n\n[生成失败: {str(e)}]"

    # --- Experiments ---
    print("[WRITEUP] 生成 Experiments...")
    try:
        exp_text, _ = get_response_from_llm(
            msg=WRITEUP_EXPERIMENTS_PROMPT.format(
                experiment_logs=json.dumps(logs, indent=2, ensure_ascii=False, default=str),
                best_config=best_config_str,
                best_f1=best_f1,
                target_f1=target_f1,
            ),
            client=client, model=model,
            system_message=WRITEUP_SYSTEM_PROMPT + "\n所有数值只能来自给定日志。验证指标必须标注验证集；没有独立测试就写未评估。引用需有可追溯来源；不得虚构文献或实验收益。",
        )
        paper_sections["experiments"] = exp_text
    except Exception as e:
        paper_sections["experiments"] = f"# 4. Experiments\n\n[生成失败: {str(e)}]"

    # --- Conclusion ---
    print("[WRITEUP] 生成 Conclusion...")
    key_findings = experiment_summary
    try:
        conclusion_text, _ = get_response_from_llm(
            msg=WRITEUP_CONCLUSION_PROMPT.format(
                method_summary=method_summary,
                best_f1=best_f1,
                target_f1=target_f1,
                key_findings=key_findings,
            ),
            client=client, model=model,
            system_message=WRITEUP_SYSTEM_PROMPT + "\n所有数值只能来自给定日志。验证指标必须标注验证集；没有独立测试就写未评估。引用需有可追溯来源；不得虚构文献或实验收益。",
        )
        paper_sections["conclusion"] = conclusion_text
    except Exception as e:
        paper_sections["conclusion"] = f"# 5. Conclusion\n\n[生成失败: {str(e)}]"

    # --- References ---
    print("[WRITEUP] 生成 References...")
    try:
        knowledge_refs = knowledge[:1000] if knowledge else "暂无"
        ref_text, _ = get_response_from_llm(
            msg=CITATION_PROMPT.format(
                related_work_text=paper_sections.get("related_work", "")[:2000],
                search_results=papers_text,
                knowledge_refs=knowledge_refs,
            ),
            client=client, model=model,
            system_message=CITATION_SYSTEM_PROMPT,
        )
        paper_sections["references"] = ref_text
    except Exception as e:
        paper_sections["references"] = f"# References\n\n[生成失败: {str(e)}]"

    # 组装完整论文
    title = "# NGC6517 Pulsar Few-Shot Classification with Multi-Modal Prototypical Networks\n\n"
    full_paper = title
    section_order = ["abstract", "introduction", "related_work", "method",
                     "experiments", "conclusion", "references"]
    for sec in section_order:
        full_paper += paper_sections.get(sec, "") + "\n\n"

    # 保存文件
    output_path = os.path.join(SKILL_ROOT, "ngc6517_paper.md")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_paper)
        print(f"[WRITEUP] 论文已保存至: {output_path}")
    except Exception as e:
        return {"error": f"论文保存失败: {str(e)}", "paper_text": full_paper}

    return {
        "status": "success",
        "output_path": output_path,
        "sections_generated": list(paper_sections.keys()),
        "paper_length": len(full_paper),
        "best_f1": best_f1,
    }


def add_citations(
    draft: Annotated[str, Field(description="待补充引用的论文草稿文本")],
) -> dict:
    """
    为论文草稿自动补充参考文献。
    调用 ArXiv 搜索 + 本地知识库查找相关文献。
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"error": f"LLM 客户端创建失败: {str(e)}"}

    # 搜索相关论文
    search_keywords = [
        "prototypical network few-shot learning",
        "pulsar classification deep learning",
        "multi-modal fusion prototype",
    ]
    papers_text = _search_related_papers(search_keywords)
    if not logs:
        return {"status": "deferred", "reason": "没有真实实验记录；不能生成性能结论。"}
    knowledge = _load_knowledge()

    try:
        ref_text, _ = get_response_from_llm(
            msg=CITATION_PROMPT.format(
                related_work_text=draft[:3000],
                search_results=papers_text,
                knowledge_refs=knowledge[:1000] if knowledge else "暂无",
            ),
            client=client, model=model,
            system_message=CITATION_SYSTEM_PROMPT,
        )
        return {
            "status": "success",
            "references": ref_text,
        }
    except Exception as e:
        return {"error": f"引用生成失败: {str(e)}"}
