"""Evidence-based prompts and local references; no synthetic expected gains."""
from skills.train_tools import evaluate_progress
from skills.rag_tools import search_local_paper_knowledge


def set_mcp_instance(instance):
    return None


def generate_research_hypothesis(target_f1, defect_query):
    return {"experiment_status": evaluate_progress(target_f1),
            "reference_papers": search_local_paper_knowledge(defect_query, 3),
            "prompt_for_ai": "提出可验证假设，区分事实、推测和待验证结论。给出对照与消融；没有实验证据就不要填写预期 F1 增益。问题：" + defect_query}


def extract_model_improve_from_paper(paper_text):
    return "从给定资料提取结构、损失函数和 episode 策略的可消融改动。保留来源，不推断已有性能收益。\\n" + paper_text
