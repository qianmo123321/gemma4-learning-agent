# 本项目服务器部署版：从 Streamlit 本地项目迁移到前后端分离

这套代码是按你上传的 **local_vertical_ai_demo_1000_plus** 项目重构的服务器版本。

原项目的核心能力：

- `app_enhanced.py`：Streamlit 页面、知识库上传、多 Agent、LoRA 启动与训练可视化；
- `enhanced_rag.py`：word TF-IDF + char TF-IDF + MMR；
- `training/lora_train_gemma4.py`：Gemma4-E2B 本地 FP16 LoRA；
- `scripts/gemma4_lora_infer_cli.py`：Windows 子进程推理。

服务器版将它们拆成：

```text
frontend/      浏览器端：ChatGPT 风格多轮对话、模型功能切换、知识库上传
backend/       FastAPI：RAG、对话历史、提示词、模型 API、反馈记录
training/      A100 40GB 训练：Gemma4-12B 4bit QLoRA
deploy/        systemd、Nginx、环境脚本
legacy_reference/  原项目关键文件备份，只做参考
```

## 1. 目标服务器目录

建议统一部署到：

```bash
/opt/gemma4_learning_agent
```

## 2. 上传与解压

本地上传压缩包到 AutoDL 后执行：

```bash
cd /opt
unzip /root/autodl-tmp/local_vertical_ai_demo_server.zip -d /opt
mv /opt/local_vertical_ai_demo_server /opt/gemma4_learning_agent
cd /opt/gemma4_learning_agent
chmod +x deploy/*.sh
```

若 AutoDL 没有 `/opt` 写权限，可改用：

```bash
/root/autodl-tmp/gemma4_learning_agent
```

后续命令中的项目路径一起替换即可。

## 3. 首先检查 GPU

```bash
nvidia-smi
```

确认是 A100 40GB，并确认没有其他训练进程占满显存。

## 4. 准备 Web/RAG 环境

```bash
cd /opt/gemma4_learning_agent
bash deploy/01_setup_web_env.sh /opt/gemma4_learning_agent
cp backend/.env.example backend/.env
nano backend/.env
```

第一次为了验证前后端通信，可以把：

```ini
MODEL_PROVIDER=mock
```

然后启动开发 API：

```bash
bash deploy/04_start_dev.sh /opt/gemma4_learning_agent
```

另开一个终端测试：

```bash
curl http://127.0.0.1:8000/api/health
```

此时前端可直接用一个静态服务器测试：

```bash
cd /opt/gemma4_learning_agent/frontend
python -m http.server 8080
```

浏览器访问：

```text
http://服务器IP:8080
```

注意：浏览器访问 `8080` 时，前端会请求 `服务器IP:8000/api`，所以 AutoDL 需要映射 8000 和 8080，或者直接按第 8 步使用 Nginx。

## 5. 导入你原项目的数据与模型 Adapter

从你的原项目复制以下内容到服务器项目中：

```text
原项目
├── data/lora_data/*.jsonl
├── private_data/knowledge_base/*
├── runtime_data/feedback.jsonl           （可选）
└── trained_models/gemma4_e2b_lora/adapter （仅本地 E2B Adapter，服务器 12B 不能直接复用）
```

复制后目标路径：

```text
服务器项目
├── data/lora_data/
├── backend/data/knowledge/
└── backend/data/feedback.jsonl
```

命令示例：

```bash
cp -r /root/autodl-tmp/uploaded/data/lora_data/* /opt/gemma4_learning_agent/data/lora_data/
cp -r /root/autodl-tmp/uploaded/private_data/knowledge_base/* /opt/gemma4_learning_agent/backend/data/knowledge/
```

> 重要：`gemma4_e2b_lora/adapter` 不能加载到 Gemma4-12B。LoRA Adapter 与基座模型尺寸、层数、权重形状绑定。你需要在 12B 基座上重新训练新 Adapter。

## 6. 准备 vLLM 环境与下载 12B 模型

```bash
bash deploy/02_setup_vllm_env.sh
bash deploy/03_download_gemma4_12b.sh /opt/gemma4_learning_agent
```

下载完成后确认：

```bash
ls -lh models/gemma/gemma-4-12B-it | head
```

## 7. 先启动 Gemma4-12B 基座推理

```bash
bash deploy/05_start_vllm_with_lora.sh /opt/gemma4_learning_agent
```

若未训练 12B Adapter，它会自动只启动基座模型。

测试：

```bash
curl http://127.0.0.1:8001/v1/models
```

编辑 `backend/.env`：

```ini
MODEL_PROVIDER=openai_compatible
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=google/gemma-4-12b-it
```

重启 API 后就会使用 Gemma4-12B 基座模型。

## 8. 正式部署 Nginx + systemd

```bash
apt update
apt install -y nginx

bash deploy/install_systemd.sh /opt/gemma4_learning_agent
systemctl start gemma4-learning-api
systemctl status gemma4-learning-api --no-pager
```

先使用 6006/8000 以外的独立 vLLM 端口 8001。Nginx 对外提供 80。

日志：

```bash
journalctl -u gemma4-learning-api -f
journalctl -u gemma4-vllm -f
```

## 9. A100 40GB 上训练 Gemma4-12B QLoRA

训练不要从网页触发。通过受控命令行运行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate gemma4_vllm

cd /opt/gemma4_learning_agent

python training/lora_train_gemma4_qlora.py \
  --model_path models/gemma/gemma-4-12B-it \
  --dataset data/lora_data/student_agent_lora_seed_50.jsonl \
  --output_dir trained_models/gemma4_12b_lora_test \
  --epochs 1 \
  --eval_size 1 \
  --max_length 256 \
  --lr 1e-4 \
  --batch 1 \
  --grad_accum 4 \
  --r 4 \
  --alpha 8 \
  --target attention
```

小样本跑通后，正式训练：

```bash
python training/lora_train_gemma4_qlora.py \
  --model_path models/gemma/gemma-4-12B-it \
  --dataset data/lora_data/student_agent_lora_full.jsonl \
  --output_dir trained_models/gemma4_12b_lora \
  --epochs 2 \
  --eval_size 20 \
  --max_length 512 \
  --lr 1e-4 \
  --batch 1 \
  --grad_accum 8 \
  --r 8 \
  --alpha 16 \
  --target attention
```

## 10. 训练后挂载 Adapter

训练输出：

```text
trained_models/gemma4_12b_lora/adapter
```

停止当前 vLLM 后，以 LoRA 形式重新启动：

```bash
pkill -f "vllm serve" || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate gemma4_vllm

vllm serve models/gemma/gemma-4-12B-it \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --enable-lora \
  --lora-modules gemma4-learning=trained_models/gemma4_12b_lora/adapter
```

然后把 `.env` 中的模型名改为：

```ini
VLLM_MODEL=gemma4-learning
```

重启 API：

```bash
systemctl restart gemma4-learning-api
```

## 11. 现场 Demo 顺序

1. 打开首页；
2. 知识库管理上传课程资料；
3. 在多轮问答中问“RAG 与 LoRA 在这个项目中分别负责什么？”；
4. 展开证据区，展示本地资料来源；
5. 追问“那我应该先优化哪个？”；
6. 切换学习路径生成两周计划；
7. 切换 AI 出题生成题目、答案和解析；
8. 提交反馈，说明反馈会整理为 JSONL 并重新训练 Gemma4-12B LoRA。

## 12. 当前版本范围

这个部署版**重点完成**：

- 前后端分离；
- Streamlit 多轮问答与提示词迁入 FastAPI；
- 用原项目 `enhanced_rag.py` 的检索思路做后端 RAG；
- A100 40GB 的 Gemma4-12B QLoRA 脚本；
- vLLM + LoRA + Nginx + systemd 的部署通路。

当前上传/解析先支持 TXT/MD/CSV。PDF、DOCX、XLSX 可在第二阶段接入解析器；你原本 Streamlit 版本的 CSV/TXT/MD 逻辑已完整保留到后端。
