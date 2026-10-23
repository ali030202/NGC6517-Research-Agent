# skills/agent_bfts.py
"""
BFTS (Best-First Tree Search) 实验探索引擎 — 无模板模式的核心。
以树结构管理实验分支，每个节点是一次架构变体实验。
Best-first 策略：选择得分最高的未展开节点继续探索。
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

SKILL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CODE_PATH = os.path.join(SKILL_ROOT, "model_proto.py")
BACKUP_PATH = os.path.join(SKILL_ROOT, "model_proto.py.bak")
BFTS_LOG_PATH = os.path.join(SKILL_ROOT, "bfts_tree.json")


# ==================== 树节点 ====================

@dataclass
class ExperimentNode:
    """实验树节点。"""
    node_id: str
    parent_id: Optional[str]
    depth: int
    code: str                          # model_proto.py 完整代码
    direction: str                     # 改进方向描述
    rationale: str = ""                # 设计理由
    f1_score: float = 0.0             # 该节点的实验 F1
    val_f1: float = 0.0               # 验证 F1
    train_status: str = "pending"      # pending / success / failed / skipped
    children: List[str] = field(default_factory=list)
    expanded: bool = False
    created_at: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "code": self.code,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "direction": self.direction,
            "rationale": self.rationale[:200],
            "f1_score": self.f1_score,
            "val_f1": self.val_f1,
            "train_status": self.train_status,
            "children": self.children,
            "expanded": self.expanded,
            "created_at": self.created_at,
            "notes": self.notes,
        }


# ==================== BFTS 探索器 ====================

class BFTSExplorer:
    """
    Best-First Tree Search 实验探索引擎。

    工作流：
    1. 初始化根节点（基线代码 或 从零生成的代码）
    2. 循环：
       a. 选择最佳未展开节点（best-first）
       b. LLM 生成 N 个架构改进方向
       c. 对每个方向：生成代码 → 写入 → 训练 → 评分
       d. 创建子节点
    3. 当达到最大深度或最大实验数时停止
    """

    def __init__(self, root_code: str, root_f1: float = 0.0):
        self.nodes: Dict[str, ExperimentNode] = {}
        self.experiment_count = 0
        self.best_f1 = root_f1
        self.best_node_id = "root"
        self.start_time = 0.0

        # 创建根节点
        root = ExperimentNode(
            node_id="root",
            parent_id=None,
            depth=0,
            code=root_code,
            direction="基线模型",
            f1_score=root_f1,
            train_status="success" if root_f1 > 0 else "baseline",
            created_at=time.time(),
        )
        self.nodes["root"] = root

    def _next_id(self) -> str:
        self.experiment_count += 1
        return f"exp_{self.experiment_count:03d}"

    def _select_best_unexplored(self) -> Optional[ExperimentNode]:
        """Best-first: 选择 F1 最高且未展开的叶子节点。"""
        candidates = [
            n for n in self.nodes.values()
            if not n.expanded
            and n.train_status == "success"
            and n.depth < self._max_depth
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.f1_score)

    def _save_checkpoint(self):
        """保存树状态到文件。"""
        try:
            tree_data = {
                "experiment_count": self.experiment_count,
                "best_f1": self.best_f1,
                "best_node_id": self.best_node_id,
                "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            }
            from ngc_agent.code import atomic_text
            atomic_text(BFTS_LOG_PATH, json.dumps(tree_data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[BFTS] 保存检查点失败: {e}")

    def _write_code_and_train(self, code: str, direction: str) -> dict:
        from ngc_agent.service import Agent
        from skills.train_tools import wait_job
        agent = Agent()
        try:
            proposal = agent.code.stage(code, direction)
            job = agent.submit(proposal_id=proposal["id"])
            result = wait_job(agent, job["id"])
            return {"status": "success", "f1": result["val_f1"], "val_f1": result["val_f1"], "result": result}
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "f1": 0.0}

    def _expand_node(self, node: ExperimentNode, branches_per_node: int) -> List[ExperimentNode]:
        """展开一个节点：生成多个架构变体，逐个训练评估。"""
        from skills.code_generator import generate_architecture_ideas, generate_model_variant

        print(f"\n[BFTS] === 展开节点 {node.node_id} (depth={node.depth}, F1={node.f1_score}) ===")

        # 1. LLM 生成改进方向
        print(f"[BFTS] 生成 {branches_per_node} 个改进方向...")
        ideas = generate_architecture_ideas(
            current_code=node.code,
            current_f1=node.f1_score,
            num_ideas=branches_per_node,
        )

        if not ideas or (len(ideas) == 1 and "error" in ideas[0]):
            print(f"[BFTS] 生成改进方向失败: {ideas}")
            node.expanded = True
            return []

        children = []
        for i, idea in enumerate(ideas[:branches_per_node]):
            if "error" in idea:
                continue

            direction = idea.get("direction", f"变体_{i+1}")
            description = idea.get("description", "")

            # 检查实验数限制
            if self.experiment_count >= self._max_experiments:
                print(f"[BFTS] 达到最大实验数 {self._max_experiments}，停止展开")
                break

            node_id = self._next_id()
            print(f"\n[BFTS] --- 实验 {node_id}: {direction} ---")

            # 2. 生成代码变体
            gen_result = generate_model_variant(
                current_code=node.code,
                current_f1=node.f1_score,
                improvement_direction=description,
                max_retries=1,
            )

            if gen_result.get("status") != "success":
                print(f"[BFTS] 代码生成失败: {gen_result.get('error')}")
                child = ExperimentNode(
                    node_id=node_id,
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    code="",
                    direction=direction,
                    train_status="code_gen_failed",
                    created_at=time.time(),
                    notes=gen_result.get("error", ""),
                )
                self.nodes[node_id] = child
                node.children.append(node_id)
                children.append(child)
                continue

            new_code = gen_result["code"]
            rationale = gen_result.get("rationale", "")

            # 3. 写入代码 + 训练
            train_result = self._write_code_and_train(new_code, direction)

            # 4. 创建子节点
            child = ExperimentNode(
                node_id=node_id,
                parent_id=node.node_id,
                depth=node.depth + 1,
                code=new_code,
                direction=direction,
                rationale=rationale,
                f1_score=train_result.get("f1", 0.0),
                val_f1=train_result.get("val_f1", 0.0),
                train_status=train_result.get("status", "unknown"),
                created_at=time.time(),
                notes=description,
            )

            self.nodes[node_id] = child
            node.children.append(node_id)
            children.append(child)

            # 更新最佳
            if child.f1_score > self.best_f1:
                self.best_f1 = child.f1_score
                self.best_node_id = child.node_id
                print(f"[BFTS] 新最佳! node={node_id}, F1={child.f1_score}")

            # 保存检查点
            self._save_checkpoint()

        node.expanded = True
        self._save_checkpoint()
        return children

    def explore(
        self,
        max_depth: int = 3,
        max_experiments: int = 15,
        branches_per_node: int = 3,
    ) -> dict:
        """
        执行 BFTS 搜索。

        Returns:
            {
                "status": "completed" | "target_reached" | "budget_exhausted",
                "best_f1": float,
                "best_node_id": str,
                "total_experiments": int,
                "tree_summary": str,
                "tree_data": dict,
            }
        """
        if not 1 <= max_depth <= 10 or not 1 <= max_experiments <= 30 or not 1 <= branches_per_node <= 5:
            raise ValueError("Invalid BFTS budget")
        self._max_depth = max_depth
        self._max_experiments = max_experiments
        self.start_time = time.time()

        print(f"\n{'#' * 60}")
        print(f"# BFTS 探索启动")
        print(f"# 最大深度: {max_depth}")
        print(f"# 最大实验数: {max_experiments}")
        print(f"# 每节点分支: {branches_per_node}")
        print(f"{'#' * 60}\n")

        # 如果根节点还没训练，先训练它
        root = self.nodes["root"]
        if root.train_status == "baseline" and root.f1_score == 0.0:
            print("[BFTS] 训练基线模型...")
            train_result = self._write_code_and_train(root.code, "基线模型")
            self.experiment_count += 1
            root.f1_score = train_result.get("f1", 0.0)
            root.val_f1 = train_result.get("val_f1", 0.0)
            root.train_status = train_result.get("status", "failed")
            if root.f1_score > self.best_f1:
                self.best_f1 = root.f1_score
            self._save_checkpoint()

        # 主循环
        iteration = 0
        while self.experiment_count < max_experiments:
            iteration += 1
            print(f"\n[BFTS] === 迭代 {iteration} | 已用实验: {self.experiment_count}/{max_experiments} ===")

            # 选择最佳未展开节点
            best_node = self._select_best_unexplored()
            if best_node is None:
                print("[BFTS] 无未展开节点，搜索结束")
                break

            print(f"[BFTS] 选择节点: {best_node.node_id} "
                  f"(depth={best_node.depth}, F1={best_node.f1_score}, "
                  f"direction={best_node.direction})")

            # 展开节点
            children = self._expand_node(best_node, branches_per_node)

            # 检查是否达标
            if self.best_f1 >= 0.9:
                print(f"\n[BFTS] 目标达成! best_f1={self.best_f1}")
                elapsed = round(time.time() - self.start_time, 2)
                return {
                    "status": "target_reached",
                    "best_f1": self.best_f1,
                    "best_node_id": self.best_node_id,
                    "total_experiments": self.experiment_count,
                    "tree_summary": self.get_tree_summary(),
                    "tree_data": {k: v.to_dict() for k, v in self.nodes.items()},
                    "elapsed_seconds": elapsed,
                }

        elapsed = round(time.time() - self.start_time, 2)
        status = "completed" if self.best_f1 > self.nodes["root"].f1_score else "no_improvement"

        print(f"\n{'#' * 60}")
        print(f"# BFTS 探索结束")
        print(f"# 状态: {status}")
        print(f"# 最佳 F1: {self.best_f1} (node: {self.best_node_id})")
        print(f"# 总实验: {self.experiment_count}")
        print(f"# 耗时: {elapsed}s")
        print(f"{'#' * 60}")

        return {
            "status": status,
            "best_f1": self.best_f1,
            "best_node_id": self.best_node_id,
            "total_experiments": self.experiment_count,
            "tree_summary": self.get_tree_summary(),
            "tree_data": {k: v.to_dict() for k, v in self.nodes.items()},
            "elapsed_seconds": elapsed,
        }

    def get_tree_summary(self) -> str:
        """生成可展示的树结构摘要。"""
        lines = ["## BFTS 实验树", ""]

        def _render(node_id: str, prefix: str = "", is_last: bool = True):
            node = self.nodes.get(node_id)
            if not node:
                return
            connector = "`-- " if is_last else "|-- "
            status_icon = {
                "success": "+",
                "failed": "x",
                "baseline": "*",
                "pending": "?",
            }.get(node.train_status, "-")

            f1_str = f"{node.f1_score:.4f}" if node.f1_score else "---"
            line = f"{prefix}{connector}[{status_icon}] {node.node_id}: {node.direction} (F1={f1_str})"
            lines.append(line)

            child_prefix = prefix + ("    " if is_last else "|   ")
            for i, child_id in enumerate(node.children):
                _render(child_id, child_prefix, i == len(node.children) - 1)

        _render("root")

        lines.append("")
        lines.append(f"**最佳:** {self.best_node_id} (F1={self.best_f1:.4f})")
        lines.append(f"**总实验:** {self.experiment_count}")

        return "\n".join(lines)

    def restore_best_code(self):
        """Explicit application of the selected model, with a new versioned backup."""
        from ngc_agent.service import Agent
        best = self.nodes.get(self.best_node_id)
        if best is None or not best.code or best.train_status != "success":
            return False
        agent = Agent()
        proposal = agent.code.stage(best.code, f"BFTS selected {best.node_id}; validation F1={best.f1_score}")
        agent.code.apply(proposal["id"])
        return True
