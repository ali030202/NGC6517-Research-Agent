#!/usr/bin/env python3
"""Local control panel backed by the same Agent service as CLI and MCP."""
import argparse
import json
from ngc_agent.config import ROOT, TrainConfig
from ngc_agent.service import Agent


def create_app(root=ROOT):
    import gradio as gr
    agent = Agent(root)
    baseline = json.dumps(TrainConfig().to_dict(), ensure_ascii=False, indent=2)

    def submit(config_text, budget, target, search_plan):
        try:
            config = TrainConfig.from_dict(json.loads(config_text))
            job = agent.run_plan(config, int(budget), float(target)) if search_plan else agent.submit(config)
            return job["id"], f"任务已提交：{job['id']}。关闭页面后后台任务仍会继续。"
        except Exception as exc:
            return gr.skip(), f"无法提交：{exc}"

    def poll(job_id):
        capabilities = agent.capabilities()
        if not job_id:
            return {"capabilities": capabilities, "jobs": agent.store.jobs(20)}, "尚未选择任务。"
        try:
            return agent.store.get(job_id.strip()), agent.logs(job_id.strip())
        except (ValueError, OSError) as exc:
            return {"error": str(exc)}, ""

    def action(job_id, name):
        try:
            if not job_id.strip():
                raise ValueError("请先填写或提交任务 ID")
            if name == "cancel":
                agent.store.cancel(job_id.strip())
                return gr.skip(), "已请求取消；后台工作进程会停止训练并保留已有日志。"
            job = getattr(agent, name)(job_id.strip())
            return job["id"], f"{name}：{job['state']}，任务 {job['id']}"
        except Exception as exc:
            return gr.skip(), str(exc)

    def chat(message, history, job_id):
        history = list(history or [])
        try:
            result = agent.chat(message, job_id or None)
            reply, active = result["reply"], result["last_job"] or job_id
        except Exception as exc:
            reply, active = f"处理失败：{exc}", job_id
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        return "", history[-80:], active

    def inspect(config_text):
        try:
            return agent.inspect(TrainConfig.from_dict(json.loads(config_text)))
        except Exception as exc:
            return {"error": str(exc)}

    def plan(config_text, budget, target):
        try:
            return agent.plan(TrainConfig.from_dict(json.loads(config_text)), int(budget), float(target))
        except Exception as exc:
            return {"error": str(exc)}

    def stage(code, reason):
        try:
            result = agent.code.stage(code, reason)
            patch = (agent.code.path(result["id"]) / "changes.diff").read_text(encoding="utf-8")
            return result["id"], patch or "提案与当前模型相同。", result
        except Exception as exc:
            return "", "", {"error": str(exc)}

    def apply(proposal_id):
        try:
            return agent.code.apply(proposal_id.strip())
        except Exception as exc:
            return {"error": str(exc)}

    def proposed_training(config_text, proposal_id):
        try:
            job = agent.submit(TrainConfig.from_dict(json.loads(config_text)), proposal_id.strip())
            return job["id"], "已提交提案训练，使用独立代码快照。"
        except Exception as exc:
            return gr.skip(), str(exc)

    def rollback(sha):
        try:
            return agent.code.rollback(sha.strip())
        except Exception as exc:
            return {"error": str(exc)}

    def report():
        result = agent.report()
        return result["markdown"], result["path"]

    with gr.Blocks(title="NGC6517 Research Agent") as demo:
        gr.Markdown("# NGC6517 科研智能体\n本地实验、资料检索、模型提案和证据报告。API 凭据配置暂缓；本地功能可直接使用。")
        with gr.Row():
            job_id = gr.Textbox(label="当前任务 ID", placeholder="提交后自动填入；也可粘贴历史任务 ID", scale=4)
            cancel = gr.Button("取消任务")
            retry = gr.Button("重新运行")
            recover = gr.Button("标记已中断任务")
        notice = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("对话入口"):
                chatbot = gr.Chatbot(type="messages", label="本地智能体", height=430)
                message = gr.Textbox(label="输入命令", placeholder="帮助、状态、计划、/search 多尺度卷积")
                send = gr.Button("发送")
            with gr.Tab("实验管理"):
                gr.Markdown("先检查数据并查看计划，再提交训练。自动实验按验证 F1 选择结果，最多运行指定次数。")
                config_text = gr.Code(value=baseline, language="json", label="训练配置")
                with gr.Row():
                    budget = gr.Slider(1, 10, value=3, step=1, label="自动实验次数预算")
                    target = gr.Slider(0, 1, value=0.9, step=0.01, label="验证 F1 停止目标")
                with gr.Row():
                    inspect_button = gr.Button("检查训练/验证数据")
                    plan_button = gr.Button("查看实验计划")
                    train_button = gr.Button("提交一次训练", variant="primary")
                    plan_run_button = gr.Button("执行自动实验计划")
                inspect_output = gr.JSON(label="数据 / 计划")
                evaluate_button = gr.Button("对当前任务的冻结模型做最终测试评估")
                gr.Markdown("最终测试需要独立 test.csv；结果不进入自动搜索。频繁查看测试集会削弱其独立性。")
            with gr.Tab("任务状态与日志"):
                refresh = gr.Button("刷新")
                status = gr.JSON(label="状态（后台定时更新）")
                logs = gr.Textbox(label="最近日志", lines=22, max_lines=30, interactive=False)
            with gr.Tab("本地知识检索"):
                query = gr.Textbox(label="关键词")
                query_button = gr.Button("检索本地资料")
                results = gr.JSON(label="原文片段与来源")
                gr.Markdown("读取 model_knowledge.md、knowledge/ 和本地 papers_extracted/。检索无需 API 或向量模型。")
            with gr.Tab("模型代码提案"):
                gr.Markdown("粘贴完整 model_proto.py 后暂存，先查看差异。静态检查只检查语法和接口；训练会执行所选代码，请先人工阅读。")
                code = gr.Code(language="python", label="候选模型代码")
                reason = gr.Textbox(label="改动说明")
                stage_button = gr.Button("暂存并查看差异")
                proposal_id = gr.Textbox(label="提案 ID")
                diff = gr.Textbox(label="代码差异", lines=16, interactive=False)
                with gr.Row():
                    proposal_train = gr.Button("训练该提案")
                    apply_button = gr.Button("应用到当前模型文件")
                backup_sha = gr.Textbox(label="备份 SHA256（应用结果中提供）")
                rollback_button = gr.Button("恢复指定备份")
                code_result = gr.JSON(label="提案 / 应用 / 回滚结果")
            with gr.Tab("实验报告"):
                report_button = gr.Button("生成基于真实记录的报告")
                report_md = gr.Markdown()
                report_file = gr.File(label="下载 Markdown 报告", interactive=False)
        timer = gr.Timer(3)
        for event in (send.click, message.submit):
            event(chat, [message, chatbot, job_id], [message, chatbot, job_id])
        inspect_button.click(inspect, config_text, inspect_output)
        plan_button.click(plan, [config_text, budget, target], inspect_output)
        train_button.click(lambda c, b, t: submit(c, b, t, False), [config_text, budget, target], [job_id, notice])
        plan_run_button.click(lambda c, b, t: submit(c, b, t, True), [config_text, budget, target], [job_id, notice])
        cancel.click(lambda j: action(j, "cancel"), job_id, [job_id, notice])
        retry.click(lambda j: action(j, "retry"), job_id, [job_id, notice])
        recover.click(lambda j: action(j, "recover"), job_id, [job_id, notice])
        evaluate_button.click(lambda j: action(j, "evaluate"), job_id, [job_id, notice])
        refresh.click(poll, job_id, [status, logs])
        timer.tick(poll, job_id, [status, logs], queue=False)
        demo.load(poll, job_id, [status, logs])
        query_button.click(agent.knowledge, query, results)
        stage_button.click(stage, [code, reason], [proposal_id, diff, code_result])
        apply_button.click(apply, proposal_id, code_result)
        proposal_train.click(proposed_training, [config_text, proposal_id], [job_id, notice])
        rollback_button.click(rollback, backup_sha, code_result)
        report_button.click(report, None, [report_md, report_file])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NGC6517 local research console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_app().queue().launch(server_name=args.host, server_port=args.port, share=False)
