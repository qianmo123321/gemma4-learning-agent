# 与原项目的迁移映射

| 原项目文件/能力 | 服务器版对应位置 | 处理方式 |
|---|---|---|
| `app_enhanced.py` | `frontend/` + `backend/app/main.py` + `backend/app/providers.py` | 拆分 UI、RAG、模型调用、反馈接口 |
| `enhanced_rag.py` | `backend/app/rag.py` | 保留 word/char 双 TF-IDF、去冗余思想，移除 Streamlit 依赖 |
| `scripts/gemma4_lora_infer_cli.py` | `backend/app/providers.py` | 不再每次子进程加载模型，改为 FastAPI 调 vLLM 常驻模型服务 |
| `training/lora_train_gemma4.py` | `training/lora_train_gemma4_qlora.py` | Windows FP16 LoRA 改为 Linux/A100 4bit QLoRA |
| `runtime_data/qa_history.jsonl` | `backend/data/qa_history.jsonl` | 后端持久化多轮对话日志 |
| `runtime_data/feedback.jsonl` | `backend/data/feedback.jsonl` | 后端持久化反馈训练池 |
| `trained_models/gemma4_e2b_lora/adapter` | 不直接迁移 | E2B Adapter 不能给 12B 使用，需要 12B 重训 |
