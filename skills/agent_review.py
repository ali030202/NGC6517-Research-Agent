# skills/agent_review.py
"""
阶段4: 自动审稿智能体 — 仿效 AI Scientist 的 perform_review.py。
LLM 模拟多个审稿人进行同行评审，支持反思迭代和多审稿人集成，
并根据审稿意见修改论文。
"""
import json
import os
from typing import Annotated, List, Dict
from pydantic import Field

from skills.llm_client import (
    create_llm_client,
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

SKILL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ==================== Prompt 模板 ====================

REVIEWER_SYSTEM_PROMPT = """你是一名严格的学术会议审稿人，专注于深度学习和小样本学习领域。
你需要对提交的论文进行全面、客观的评审。
你的评审应该具有建设性，指出论文的优点和不足，并给出具体的改进建议。
请保持专业和公正的态度。"""

REVIEW_PROMPT = """请对以下论文进行全面评审。

---

{paper_text}

---

请从以下维度进行评审，并严格按照指定的 JSON 格式输出：

1. **Summary**: 用 2-3 句话概括论文的核心贡献
2. **Strengths**: 列出论文的 3-5 个优点
3. **Weaknesses**: 列出论文的 3-5 个不足之处
4. **Questions**: 列出 2-3 个需要作者回答的问题
5. **Overall Score**: 1-10 分（1=很差, 10=优秀）
6. **Soundness**: 技术正确性 1-4 分
7. **Novelty**: 新颖性 1-4 分
8. **Significance**: 重要性 1-4 分
9. **Confidence**: 你对该领域的了解程度 1-4 分
10. **Decision**: Accept 或 Reject

请使用以下 JSON 格式输出：

```json
{{
    "Summary": "...",
    "Strengths": ["...", "...", "..."],
    "Weaknesses": ["...", "...", "..."],
    "Questions": ["...", "..."],
    "Overall": 7,
    "Soundness": 3,
    "Novelty": 3,
    "Significance": 3,
    "Confidence": 3,
    "Decision": "Accept/Reject"
}}
```"""

REVIEW_REFLECTION_PROMPT = """第 {current_round}/{num_reflections} 轮反思。

你之前的评审意见如下：
{previous_review}

请重新审视你的评审：
1. 评审是否公正、客观？
2. 是否遗漏了重要问题？
3. 评分是否合理？

如果需要修改，请输出更新后的完整评审 JSON。
如果评审已经足够好，请在回复中包含 "Review finalized." 并重复之前的 JSON。

论文原文（供参考）：
{paper_excerpt}"""

META_REVIEW_SYSTEM_PROMPT = """你是一名资深领域主席（Area Chair），负责汇总多位审稿人的意见，
形成最终的 meta-review 决定。你需要综合考虑所有审稿意见，做出公正的评判。"""

META_REVIEW_PROMPT = """以下是 {num_reviews} 位审稿人对同一篇论文的独立评审意见：

{all_reviews}

---

请作为领域主席，综合所有审稿意见，生成 meta-review：

```json
{{
    "Meta_Summary": "论文总体评价摘要",
    "Consensus_Strengths": ["共识优点1", "共识优点2"],
    "Consensus_Weaknesses": ["共识不足1", "共识不足2"],
    "Conflicting_Views": ["分歧观点1"],
    "Meta_Score": 7,
    "Meta_Decision": "Accept/Reject",
    "Revision_Suggestions": ["修改建议1", "修改建议2", "修改建议3"],
    "Reasoning": "最终决定的推理过程"
}}
```"""

IMPROVEMENT_SYSTEM_PROMPT = """你是一名经验丰富的学术论文修改专家。
你需要根据审稿人的意见，对论文进行有针对性的修改和完善。
修改后的内容必须直接解决审稿人提出的问题，同时保持论文的整体结构。"""

IMPROVEMENT_PROMPT = """## 审稿意见（Meta-Review）
{review_summary}

## 具体修改建议
{revision_suggestions}

## 原始论文
{paper_text}

---

请根据以上审稿意见，修改论文中需要改进的部分。

修改要求：
1. 针对每条修改建议，在论文中做出对应的改进
2. 保持论文的整体结构和格式不变
3. 确保修改后的内容逻辑连贯、学术规范
4. 对于无法满足的建议（如需要额外实验数据），在论文中增加讨论说明

请输出修改后的完整论文（Markdown 格式）。
如果某些章节不需要修改，保持原样输出。"""


# ==================== 核心函数 ====================

def perform_review(
    paper_text: Annotated[str, Field(description="待审稿的论文全文（Markdown 格式）")],
    num_reflections: Annotated[int, Field(description="每位审稿人的反思迭代轮数", ge=1, le=5)] = 3,
    num_reviews_ensemble: Annotated[int, Field(description="独立审稿人数量", ge=1, le=5)] = 3,
) -> dict:
    """
    模拟同行评审：多位审稿人独立评审 + 反思迭代 + meta-review 聚合。
    返回结构化评审结果。
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"error": f"LLM 客户端创建失败: {str(e)}"}

    print(f"[REVIEW] 开始审稿: {num_reviews_ensemble} 位审稿人, "
          f"每人 {num_reflections} 轮反思")

    all_reviews = []

    # 多位审稿人独立评审
    for reviewer_idx in range(num_reviews_ensemble):
        print(f"[REVIEW] --- 审稿人 {reviewer_idx + 1}/{num_reviews_ensemble} ---")

        try:
            # 初始评审
            text, msg_history = get_response_from_llm(
                msg=REVIEW_PROMPT.format(paper_text=paper_text[:8000]),
                client=client, model=model,
                system_message=REVIEWER_SYSTEM_PROMPT,
                temperature=0.7 + reviewer_idx * 0.1,  # 略微不同的温度增加多样性
            )

            review_json = extract_json_between_markers(text)
            if review_json is None:
                print(f"[REVIEW] 审稿人 {reviewer_idx + 1}: JSON 解析失败")
                # 尝试从文本中提取分数
                review_json = {
                    "Summary": text[:300],
                    "Strengths": [],
                    "Weaknesses": [],
                    "Overall": 5,
                    "Decision": "Reject",
                    "raw_text": text,
                }

            # 反思迭代
            if num_reflections > 1:
                for ref_round in range(num_reflections - 1):
                    prev_review_str = json.dumps(review_json, indent=2, ensure_ascii=False)
                    paper_excerpt = paper_text[:3000] + "\n... (省略)"

                    text, msg_history = get_response_from_llm(
                        msg=REVIEW_REFLECTION_PROMPT.format(
                            current_round=ref_round + 2,
                            num_reflections=num_reflections,
                            previous_review=prev_review_str,
                            paper_excerpt=paper_excerpt,
                        ),
                        client=client, model=model,
                        system_message=REVIEWER_SYSTEM_PROMPT,
                        msg_history=msg_history,
                    )

                    new_review = extract_json_between_markers(text)
                    if new_review is not None:
                        review_json = new_review

                    if "review finalized" in text.lower():
                        print(f"[REVIEW] 审稿人 {reviewer_idx + 1}: "
                              f"第 {ref_round + 2} 轮收敛")
                        break

            review_json["reviewer_id"] = reviewer_idx + 1
            all_reviews.append(review_json)
            print(f"[REVIEW] 审稿人 {reviewer_idx + 1}: "
                  f"Overall={review_json.get('Overall', 'N/A')}, "
                  f"Decision={review_json.get('Decision', 'N/A')}")

        except Exception as e:
            print(f"[REVIEW] 审稿人 {reviewer_idx + 1} 异常: {e}")
            all_reviews.append({
                "reviewer_id": reviewer_idx + 1,
                "error": str(e),
                "Overall": 5,
                "Decision": "Reject",
            })

    # 生成 Meta-Review
    print("[REVIEW] 生成 Meta-Review...")
    meta_review = get_meta_review(all_reviews, client, model)

    return {
        "status": "success",
        "num_reviewers": num_reviews_ensemble,
        "reviews": all_reviews,
        "meta_review": meta_review,
        "avg_score": round(
            sum(r.get("Overall", 5) for r in all_reviews) / max(len(all_reviews), 1), 2
        ),
    }


def get_meta_review(
    reviews: Annotated[list, Field(description="多位审稿人的独立评审结果列表")],
    _client=None,
    _model=None,
) -> dict:
    """
    聚合多个审稿意见为 meta-review。
    可由外部传入 client/model，也可自动创建。
    """
    own_client = False
    if _client is None or _model is None:
        try:
            _client, _model = create_llm_client()
            own_client = True
        except Exception as e:
            return {"error": f"LLM 客户端创建失败: {str(e)}"}

    if not reviews:
        return {"error": "无审稿意见可聚合"}

    # 格式化所有审稿意见
    all_reviews_text = ""
    for i, r in enumerate(reviews):
        all_reviews_text += f"\n### 审稿人 {i + 1}\n"
        all_reviews_text += json.dumps(r, indent=2, ensure_ascii=False) + "\n"

    try:
        text, _ = get_response_from_llm(
            msg=META_REVIEW_PROMPT.format(
                num_reviews=len(reviews),
                all_reviews=all_reviews_text,
            ),
            client=_client, model=_model,
            system_message=META_REVIEW_SYSTEM_PROMPT,
        )

        meta = extract_json_between_markers(text)
        if meta is None:
            return {"raw_response": text, "status": "parse_failed"}

        meta["status"] = "success"
        return meta

    except Exception as e:
        return {"error": f"Meta-Review 生成失败: {str(e)}"}


def perform_improvement(
    review: Annotated[dict, Field(description="审稿意见（meta-review 或单条审稿意见）")],
    paper_text: Annotated[str, Field(description="原始论文文本（Markdown 格式）")],
) -> dict:
    """
    根据审稿意见修改论文，返回修改后的论文文本。
    """
    try:
        client, model = create_llm_client()
    except Exception as e:
        return {"error": f"LLM 客户端创建失败: {str(e)}"}

    # 提取审稿摘要和修改建议
    review_summary = ""
    revision_suggestions = ""

    if "Meta_Summary" in review:
        review_summary = review.get("Meta_Summary", "")
        suggestions = review.get("Revision_Suggestions", [])
        revision_suggestions = "\n".join(f"- {s}" for s in suggestions)
    else:
        review_summary = review.get("Summary", json.dumps(review, ensure_ascii=False))
        weaknesses = review.get("Weaknesses", [])
        revision_suggestions = "\n".join(f"- 修复: {w}" for w in weaknesses)

    if not revision_suggestions:
        revision_suggestions = "- 请改进论文的整体质量和清晰度"

    print(f"[IMPROVE] 根据审稿意见修改论文...")

    try:
        improved_text, _ = get_response_from_llm(
            msg=IMPROVEMENT_PROMPT.format(
                review_summary=review_summary,
                revision_suggestions=revision_suggestions,
                paper_text=paper_text[:8000],
            ),
            client=client, model=model,
            system_message=IMPROVEMENT_SYSTEM_PROMPT,
        )

        # 保存修改后的论文
        output_path = os.path.join(SKILL_ROOT, "ngc6517_paper_revised.md")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(improved_text)
            print(f"[IMPROVE] 修改后论文已保存至: {output_path}")
        except Exception:
            pass

        return {
            "status": "success",
            "improved_paper": improved_text,
            "output_path": output_path,
            "paper_length": len(improved_text),
        }

    except Exception as e:
        return {"error": f"论文修改失败: {str(e)}"}
