# Gemma4 私有学习智能体

本项目是一个基于 **Gemma4 + vLLM + FastAPI + RAG + LoRA 数据闭环** 的私有学习智能体原型系统。

系统面向人工智能入门学习场景，支持本地大模型推理、私有知识库问答、网页交互、对话历史保存、回答质量反馈，以及高质量样本导出用于后续 LoRA 微调。

---

## 一、项目目标

本项目希望构建一个面向学习场景的闭环式 AI 助教系统：

```text
私有知识库
    ↓
RAG 检索增强
    ↓
Gemma4 本地推理
    ↓
网页学习助手
    ↓
对话记忆与质量反馈
    ↓
LoRA 微调数据导出
系统不仅用于回答问题，还可以持续收集高质量问答样本，为后续模型微调和行为风格优化提供数据基础。

二、核心功能
基于 vLLM 的本地 Gemma4 推理
FastAPI 后端服务
轻量级 Web 前端界面
RAG 私有知识库检索增强问答
Markdown 知识库文件支持
SQLite 持久化对话历史
会话重命名、删除与导出
回答质量评分
LoRA 训练样本标记
高质量样本导出为 JSONL
提供 Nginx、systemd、vLLM 等部署脚本
三、项目结构
gemma4_learning_agent/
├── app/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── rag.py
│   │   │   ├── providers.py
│   │   │   ├── schemas.py
│   │   │   ├── storage.py
│   │   │   └── conversation_store.py
│   │   ├── data/
│   │   │   └── knowledge/
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   └── build_lora_dataset_v1.py
│   │
│   ├── frontend/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   │
│   ├── deploy/
│   │   ├── 01_setup_web_env.sh
│   │   ├── 02_setup_vllm_env.sh
│   │   ├── 03_download_gemma4_12b.sh
│   │   ├── 04_start_dev.sh
│   │   ├── 05_start_vllm_with_lora.sh
│   │   ├── nginx.conf
│   │   ├── gemma4-learning-api.service
│   │   ├── gemma4-vllm.service
│   │   └── install_systemd.sh
│   │
│   ├── training/
│   │   ├── lora_train_gemma4_qlora.py
│   │   └── requirements-training.txt
│   │
│   ├── README_DEPLOY_CN.md
│   └── MIGRATION_MAP.md
│
├── .gitignore
└── README.md
四、主要模块
1. RAG 私有知识库问答

系统支持本地 Markdown 知识库。后端会对知识库文档进行加载、切块、索引和检索，并将相关内容作为上下文传入大模型，提高回答的依据性。

示例知识库目录：

app/backend/data/knowledge/

当前示例知识库主要面向人工智能小白学习，包括：

人工智能基础概念
Python 入门路线
数据处理基础
机器学习入门
模型评估指标
深度学习基础
大模型基础
RAG 检索增强生成
LoRA 微调入门
学习智能体设计
四周 AI 入门计划
AI 伦理与学习边界
2. Gemma4 本地推理

系统后端通过 OpenAI-compatible 接口调用本地 vLLM 服务。

示例启动方式：

vllm serve /path/to/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192

FastAPI 后端会调用该本地模型服务完成回答生成。

3. Web 前端学习助手

前端提供浏览器交互界面，支持：

与学习智能体对话
基于知识库进行问答
查看历史会话
重命名和删除会话
导出会话
对回答进行质量评分
标记回答是否可作为 LoRA 训练样本

前端文件位于：

app/frontend/
4. 对话记忆与反馈系统

系统使用 SQLite 保存对话历史和反馈信息，包括：

会话信息
用户消息
助手回答
RAG 检索证据
使用的模型信息
回答质量评分
是否加入 LoRA 训练样本

运行时数据库不会上传到 GitHub，避免泄露真实对话内容。

5. LoRA 数据闭环

用户可以对高质量回答进行评分，并勾选为训练样本。随后可以通过脚本将这些样本导出为 LoRA 微调数据集。

导出脚本：

app/backend/build_lora_dataset_v1.py

典型数据格式：

{
  "messages": [
    {
      "role": "user",
      "content": "RAG 是什么？"
    },
    {
      "role": "assistant",
      "content": "RAG 是检索增强生成，系统会先从知识库中检索相关资料，再让大模型基于资料生成回答。"
    }
  ],
  "meta": {
    "rating": 5,
    "agent_mode": "qa",
    "conversation_id": "..."
  }
}
五、快速开始
1. 克隆项目
git clone https://github.com/qianmo123321/gemma4-learning-agent.git
cd gemma4-learning-agent
2. 安装后端依赖
cd app/backend
pip install -r requirements.txt
3. 配置环境变量
cp .env.example .env
nano .env

根据自己的模型路径、vLLM 地址、数据库路径和知识库路径修改 .env。

4. 启动 vLLM
vllm serve /path/to/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192
5. 启动后端服务
cd app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
6. 部署前端

前端可以通过 Nginx 或任意静态文件服务部署。

部署说明可参考：

app/README_DEPLOY_CN.md
app/deploy/nginx.conf
六、LoRA 数据导出

在收集足够多的高质量反馈样本后，执行：

cd app/backend
python build_lora_dataset_v1.py

推荐数据规模：

小规模功能验证：50～100 条高质量样本
初步 LoRA 训练：300～1000 条高质量样本
稳定行为风格微调：1000 条以上高质量样本
七、仓库不包含的内容

出于安全和存储考虑，本仓库不包含：

Gemma4 模型权重
Hugging Face 缓存
Conda 环境
SQLite 运行数据库
私有 .env 配置文件
API Key 或访问 Token
大型训练检查点

这些文件均已通过 .gitignore 排除。

八、适用场景

本项目适合用于：

人工智能教学展示
私有学习智能体原型开发
RAG 课程资料问答
LoRA 微调数据采集
本地大模型应用开发实验
教学平台或智能助教项目验证

当前项目仍属于教育原型系统，不是生产级多用户平台。

九、GitHub 仓库
https://github.com/qianmo123321/gemma4-learning-agent
十、说明

本项目仅用于学习、研究和教学展示。正式部署到公网前，建议进一步补充用户权限管理、数据隔离、访问控制、日志审计和安全策略。
