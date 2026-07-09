from app.config import get_settings
from app.conversation_store import list_conversations, get_messages

import json
from pathlib import Path

settings = get_settings()

OUT_DIR = Path("/root/autodl-tmp/gemma4_learning_agent/training_data")
OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_FILE = OUT_DIR / "lora_train_v1.jsonl"

MIN_RATING = 4


def valid_pair(q, a):
    return q and a and len(a) > 30


rows = []

conversations = list_conversations(settings.conversation_db_path)

for conv in conversations:
    msgs = get_messages(settings.conversation_db_path, conv["conversation_id"])

    last_user = None

    for m in msgs:
        if m["role"] == "user":
            last_user = m["content"]

        if m["role"] != "assistant":
            continue

        # 获取评分（如果没有评分字段也不影响）
        feedback = m.get("quality_feedback", {})
        rating = feedback.get("rating", 0) if feedback else 0
        selected = feedback.get("training_selected", False) if feedback else False

        if not selected or rating < MIN_RATING:
            continue

        if not valid_pair(last_user, m["content"]):
            continue

        rows.append({
            "messages": [
                {"role": "user", "content": last_user},
                {"role": "assistant", "content": m["content"]}
            ],
            "meta": {
                "rating": rating,
                "agent_mode": conv.get("agent_mode", "qa"),
                "conversation_id": conv["conversation_id"]
            }
        })

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("完成训练集构建")
print("样本数：", len(rows))
print("输出路径：", OUT_FILE)
