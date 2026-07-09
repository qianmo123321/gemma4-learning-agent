const API = window.__API_BASE__ || "/api";

const state = {
  chats: [],
  active: null,
  mode: "qa",
};

const $ = (selector) => document.querySelector(selector);
const messagesBox = $("#chatMessages");
const historyBox = $("#historyList");

function esc(text = "") {
  return text.replace(
    /[&<>"']/g,
    (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    }[char]),
  );
}

function renderText(text = "") {
  return esc(text)
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
    .replace(/\n/g, "<br>");
}

function getChat() {
  return state.chats.find((chat) => chat.id === state.active) || null;
}

function setMode(mode) {
  state.mode = mode;

  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });

  const placeholderMap = {
    qa: "输入问题；你可以继续追问上一轮内容…",
    learning_path: "例如：我想在两周内入门机器学习，每天可学 2 小时。",
    quiz: "例如：围绕 RAG 基础生成 5 道中等难度选择题。",
    coach: "说说你目前学不懂的地方，我会陪你一步步梳理。",
  };

  $("#userInput").placeholder = placeholderMap[mode] || placeholderMap.qa;
}

function renderHistory() {
  historyBox.innerHTML = "";

  state.chats.forEach((chat) => {
    const row = document.createElement("div");
    row.className = `history-item ${
      chat.id === state.active ? "active" : ""
    }`;

    const titleButton = document.createElement("button");
    titleButton.type = "button";
    titleButton.className = "history-title";
    titleButton.textContent = chat.title || "新对话";
    titleButton.title = chat.title || "新对话";

    titleButton.onclick = async () => {
      try {
        await openConversation(chat.id);
      } catch (error) {
        alert(`读取会话失败：${error.message}`);
      }
    };

    const actions = document.createElement("div");
    actions.className = "history-actions";

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "history-action";
    renameButton.textContent = "✎";
    renameButton.title = "重命名会话";
    renameButton.setAttribute("aria-label", "重命名会话");

    renameButton.onclick = async (event) => {
      event.stopPropagation();

      try {
        await renameConversation(chat);
      } catch (error) {
        alert(`重命名失败：${error.message}`);
      }
    };

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "history-action danger";
    deleteButton.textContent = "×";
    deleteButton.title = "删除会话";
    deleteButton.setAttribute("aria-label", "删除会话");

    const exportButton = document.createElement("button");
    exportButton.type = "button";
    exportButton.className = "history-action";
    exportButton.textContent = "⇩";
    exportButton.title = "导出 Markdown";
    exportButton.setAttribute("aria-label", "导出 Markdown");

    exportButton.onclick = (event) => {
      event.stopPropagation();
      downloadConversation(chat, "markdown");
    };


    deleteButton.onclick = async (event) => {
      event.stopPropagation();

      try {
        await removeConversation(chat);
      } catch (error) {
        alert(`删除失败：${error.message}`);
      }
    };

    actions.append(renameButton, exportButton, deleteButton);
    row.append(titleButton, actions);
    historyBox.appendChild(row);
  });
}

async function renameConversation(chat) {
  const nextTitle = window.prompt(
    "请输入新的会话名称：",
    chat.title || "新对话",
  );

  if (nextTitle === null) {
    return;
  }

  const title = nextTitle.trim();

  if (!title) {
    throw new Error("会话名称不能为空。");
  }

  const response = await fetch(`${API}/conversations/${chat.id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      title,
    }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "服务器未能更新会话名称。");
  }

  chat.title = data.title;
  chat.updated_at = data.updated_at;

  renderHistory();
}


function downloadConversation(chat, format = "markdown") {
  const extension = format === "json" ? "json" : "md";

  const link = document.createElement("a");
  link.href = `${API}/conversations/${chat.id}/export?format=${format}`;
  link.download = `conversation_${chat.id}.${extension}`;

  document.body.appendChild(link);
  link.click();
  link.remove();
}


async function removeConversation(chat) {
  const title = chat.title || "这个会话";

  const confirmed = window.confirm(
    `确定删除“${title}”吗？\n\n删除后，该会话中的全部聊天记录和 RAG 证据都会从 SQLite 中移除，无法恢复。`,
  );

  if (!confirmed) {
    return;
  }

  const wasActive = state.active === chat.id;

  const response = await fetch(`${API}/conversations/${chat.id}`, {
    method: "DELETE",
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "服务器未能删除会话。");
  }

  state.chats = state.chats.filter((item) => item.id !== chat.id);

  if (wasActive) {
    state.active = null;

    if (state.chats.length > 0) {
      await openConversation(state.chats[0].id);
    } else {
      await newChat();
    }
  } else {
    renderHistory();
  }
}


function evidenceHtml(items = []) {
  if (!items.length) {
    return "";
  }

  return `
    <details>
      <summary>查看本轮 RAG 证据（${items.length} 条）</summary>
      ${items
        .map(
          (item) => `
            <div class="evidence">
              <small>
                ${esc(item.source_file)} · score=${item.score}
              </small>
              ${esc(item.text)}
            </div>
          `,
        )
        .join("")}
    </details>
  `;
}

function findMessageById(messageId) {
  for (const chat of state.chats) {
    const message = (chat.messages || []).find(
      (item) => item.message_id === messageId,
    );

    if (message) {
      return message;
    }
  }

  return null;
}


function qualityFeedbackHtml(message) {
  if (!message.message_id) {
    return "";
  }

  const quality = message.quality_feedback || null;
  const rating = quality?.rating || 0;
  const trainingSelected = Boolean(quality?.training_selected);

  const stars = Array.from(
    { length: 5 },
    (_, index) => {
      const star = index + 1;

      return `
        <button
          type="button"
          class="quality-star ${star <= rating ? "selected" : ""}"
          data-message-id="${message.message_id}"
          data-rate="${star}"
          title="评分 ${star} 星"
          aria-label="评分 ${star} 星"
        >★</button>
      `;
    },
  ).join("");

  const feedbackText = quality?.feedback
    ? "修改评价"
    : "填写评价";

  const selectedText = trainingSelected
    ? '<span class="training-tag">已加入训练候选</span>'
    : "";

  return `
    <div class="quality-panel" data-message-id="${message.message_id}">
      <div class="quality-topline">
        <span class="quality-label">回答质量</span>
        <div class="quality-stars">${stars}</div>
        <span class="quality-score">
          ${rating ? `${rating}/5` : "未评分"}
        </span>
      </div>

      <div class="quality-actions">
        <label class="training-check">
          <input
            class="training-select"
            type="checkbox"
            data-message-id="${message.message_id}"
            ${trainingSelected ? "checked" : ""}
          >
          <span>加入训练样本</span>
        </label>

        <button
          type="button"
          class="feedback-note-btn"
          data-message-id="${message.message_id}"
        >${feedbackText}</button>

        ${selectedText}
      </div>
    </div>
  `;
}


async function saveQualityFeedback(
  messageId,
  rating,
  feedback,
  trainingSelected,
) {
  const response = await fetch(
    `${API}/messages/${messageId}/feedback`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        rating,
        feedback,
        training_selected: trainingSelected,
      }),
    },
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "评分保存失败。");
  }

  const message = findMessageById(messageId);

  if (message) {
    message.quality_feedback = data;
  }

  renderMessages();
}


function bindQualityActions() {
  messagesBox.querySelectorAll(".quality-star").forEach((button) => {
    button.onclick = async () => {
      const messageId = button.dataset.messageId;
      const rating = Number(button.dataset.rate);
      const message = findMessageById(messageId);

      if (!message) {
        return;
      }

      const panel = button.closest(".quality-panel");
      const trainingSelected = Boolean(
        panel?.querySelector(".training-select")?.checked,
      );

      const currentFeedback =
        message.quality_feedback?.feedback || "";

      const feedback = window.prompt(
        "可选：填写这条回答的优点、问题或改进建议。",
        currentFeedback,
      );

      if (feedback === null) {
        return;
      }

      try {
        await saveQualityFeedback(
          messageId,
          rating,
          feedback.trim(),
          trainingSelected,
        );
      } catch (error) {
        alert(`评分保存失败：${error.message}`);
      }
    };
  });

  messagesBox.querySelectorAll(".training-select").forEach((checkbox) => {
    checkbox.onchange = async () => {
      const messageId = checkbox.dataset.messageId;
      const message = findMessageById(messageId);
      const rating = message?.quality_feedback?.rating || 0;

      if (!rating) {
        checkbox.checked = false;
        alert("请先为该回答选择 1 到 5 星评分。");
        return;
      }

      try {
        await saveQualityFeedback(
          messageId,
          rating,
          message.quality_feedback?.feedback || "",
          checkbox.checked,
        );
      } catch (error) {
        checkbox.checked = !checkbox.checked;
        alert(`训练样本状态保存失败：${error.message}`);
      }
    };
  });

  messagesBox.querySelectorAll(".feedback-note-btn").forEach((button) => {
    button.onclick = async () => {
      const messageId = button.dataset.messageId;
      const message = findMessageById(messageId);
      const rating = message?.quality_feedback?.rating || 0;

      if (!rating) {
        alert("请先为该回答选择 1 到 5 星评分。");
        return;
      }

      const feedback = window.prompt(
        "填写评价或改进建议：",
        message.quality_feedback?.feedback || "",
      );

      if (feedback === null) {
        return;
      }

      try {
        await saveQualityFeedback(
          messageId,
          rating,
          feedback.trim(),
          Boolean(message.quality_feedback?.training_selected),
        );
      } catch (error) {
        alert(`评价保存失败：${error.message}`);
      }
    };
  });
}


function renderMessages() {
  const chat = getChat();

  if (!chat) {
    messagesBox.innerHTML = "";
    return;
  }

  if (!chat.messages || !chat.messages.length) {
    messagesBox.innerHTML = `
      <div class="welcome">
        <span class="eyebrow">Gemma4 Learning Agent</span>
        <h3>今天想从哪里开始？</h3>
        <p>
          你可以基于本地课程资料连续追问，也可以切换到学习路径、
          AI 出题与 AI 陪练。
        </p>
        <div class="suggestions">
          <button data-q="如何学习 RAG 技术？">如何学习 RAG 技术？</button>
          <button data-q="RAG 和 LoRA 在这个项目里分别负责什么？">
            RAG 和 LoRA 分别负责什么？
          </button>
          <button data-q="请给我规划两周机器学习学习路径。">
            规划两周学习路径
          </button>
        </div>
      </div>
    `;

    document.querySelectorAll("[data-q]").forEach((button) => {
      button.onclick = () => {
        $("#userInput").value = button.dataset.q;
        $("#userInput").focus();
      };
    });
  } else {
    messagesBox.innerHTML = chat.messages
      .map(
        (message) => `
          <section class="message ${message.role}">
            <span class="who">
              ${message.role === "user" ? "你" : "Gemma4 学习助教"}
            </span>
            <div class="bubble">${renderText(message.content)}</div>
            ${
              message.role === "assistant"
                ? `${evidenceHtml(message.evidence || [])}${qualityFeedbackHtml(message)}`
                : ""
            }
          </section>
        `,
      )
      .join("");
  }
  bindQualityActions();
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function addMessage(message) {
  const chat = getChat();

  if (!chat) {
    return;
  }

  chat.messages.push(message);
  renderMessages();
}

async function loadConversationList() {
  const response = await fetch(`${API}/conversations`);

  if (!response.ok) {
    throw new Error("无法读取历史会话列表。");
  }

  const summaries = await response.json();
  const oldChats = new Map(state.chats.map((chat) => [chat.id, chat]));

  state.chats = summaries.map((item) => {
    const oldChat = oldChats.get(item.conversation_id);

    return {
      id: item.conversation_id,
      title: item.title,
      agent_mode: item.agent_mode,
      created_at: item.created_at,
      updated_at: item.updated_at,
      messages: oldChat?.messages || [],
    };
  });

  renderHistory();
}

async function openConversation(conversationId) {
  const response = await fetch(`${API}/conversations/${conversationId}`);

  if (!response.ok) {
    throw new Error("会话不存在或读取失败。");
  }

  const detail = await response.json();

  const chat = {
    id: detail.conversation_id,
    title: detail.title,
    agent_mode: detail.agent_mode || "qa",
    created_at: detail.created_at,
    updated_at: detail.updated_at,
    messages: (detail.messages || []).map((message) => ({
      message_id: message.message_id,
      role: message.role,
      content: message.content,
      evidence: message.evidence || [],
      model_used: message.model_used || null,
      quality_feedback: message.quality_feedback || null,
    })),
  };

  const index = state.chats.findIndex((item) => item.id === chat.id);

  if (index >= 0) {
    state.chats[index] = chat;
  } else {
    state.chats.unshift(chat);
  }

  state.active = chat.id;
  setMode(chat.agent_mode);
  renderHistory();
  renderMessages();
}

async function newChat() {
  const response = await fetch(`${API}/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      title: "新对话",
      agent_mode: state.mode,
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "创建会话失败。");
  }

  const item = await response.json();

  const chat = {
    id: item.conversation_id,
    title: item.title,
    agent_mode: item.agent_mode,
    created_at: item.created_at,
    updated_at: item.updated_at,
    messages: [],
  };

  state.chats.unshift(chat);
  state.active = chat.id;

  renderHistory();
  renderMessages();

  return chat;
}

async function send() {
  const input = $("#userInput");
  const raw = input.value.trim();
  const chat = getChat();

  if (!raw || !chat) {
    return;
  }

  input.value = "";

  addMessage({
    role: "user",
    content: raw,
  });

  const button = $("#sendBtn");
  button.disabled = true;
  button.textContent = "生成中…";

  try {
    const response = await fetch(`${API}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        conversation_id: chat.id,
        messages: [
          {
            role: "user",
            content: raw,
          },
        ],
        agent_mode: state.mode,
        use_rag: $("#useRag").checked,
        top_k: Number($("#topK").value),
        temperature: 0.35,
        max_tokens: Number($("#maxTokens").value),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "模型请求失败。");
    }

    chat.title = data.title || chat.title;
    chat.agent_mode = state.mode;

    addMessage({
      message_id: data.assistant_message_id || null,
      role: "assistant",
      content: data.answer,
      evidence: data.evidence || [],
      model_used: data.model_used || null,
      quality_feedback: null,
    });

    await loadConversationList();
    renderMessages();
  } catch (error) {
    addMessage({
      role: "assistant",
      content: `### 模型调用失败\n${error.message}\n\n请检查 FastAPI、vLLM 与模型服务日志。`,
      evidence: [],
    });
  } finally {
    button.disabled = false;
    button.textContent = "发送 ↗";
  }
}

async function refreshStatus() {
  try {
    const response = await fetch(`${API}/health`);
    const data = await response.json();

    $("#statusText").textContent = `在线 · ${data.provider}`;
    $("#fileMetric").textContent = data.knowledge_files;
    $("#chunkMetric").textContent = data.knowledge_chunks;
  } catch {
    $("#statusDot").style.background = "#ff7180";
    $("#statusText").textContent = "后端未连接";
  }
}

async function refreshKnowledge() {
  const response = await fetch(`${API}/knowledge/status`);
  const data = await response.json();

  $("#fileMetric").textContent = data.file_count;
  $("#chunkMetric").textContent = data.chunk_count;

  $("#sourceList").innerHTML = data.sources.length
    ? data.sources
        .map((source) => `<div class="source">${esc(source)}</div>`)
        .join("")
    : `<div class="source">暂未上传知识库文件。</div>`;
}

$("#newChatBtn").onclick = async () => {
  try {
    await newChat();
  } catch (error) {
    alert(`创建新对话失败：${error.message}`);
  }
};

$("#knowledgeBtn").onclick = async () => {
  await refreshKnowledge();
  $("#knowledgeDialog").showModal();
};

$("#closeKnowledgeBtn").onclick = () => {
  $("#knowledgeDialog").close();
};

$("#sendBtn").onclick = send;

$("#userInput").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    send();
  }
});

$("#topK").oninput = (event) => {
  $("#topKValue").textContent = event.target.value;
};

$("#maxTokens").oninput = (event) => {
  $("#maxTokensValue").textContent = event.target.value;
};

document.querySelectorAll(".tab").forEach((button) => {
  button.onclick = () => {
    setMode(button.dataset.mode);
  };
});

$("#uploadForm").onsubmit = async (event) => {
  event.preventDefault();

  const file = $("#knowledgeFile").files[0];

  if (!file) {
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "索引构建中…";

  try {
    const response = await fetch(`${API}/knowledge/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "上传失败");
    }

    await refreshKnowledge();
    alert(data.message);
  } catch (error) {
    alert(`上传失败：${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "上传并重建索引";
  }
};

async function boot() {
  try {
    await loadConversationList();

    if (state.chats.length > 0) {
      await openConversation(state.chats[0].id);
    } else {
      await newChat();
    }
  } catch (error) {
    messagesBox.innerHTML = `
      <div class="welcome">
        <h3>会话系统初始化失败</h3>
        <p>${esc(error.message)}</p>
      </div>
    `;
  }

  await refreshStatus();
}

setMode("qa");
boot();
