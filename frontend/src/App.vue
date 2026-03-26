<template>
  <div class="header">
    <h1>🤖 AI Research Agent</h1>
    <p>支持工具调用 · RAG 知识库 · 多轮对话记忆</p>
    <button class="new-chat-button" @click="startNewChat">🆕 新对话</button>
  </div>

  <div class="chat-container" ref="chatContainer">
    <div v-if="messages.length === 0" class="info">
      👋 你好！我是 AI Research Agent，可以帮你计算、搜索知识库。开始聊天吧！
    </div>
    
    <div v-for="(msg, index) in messages" :key="index" :class="['message', msg.role]">
      <div class="message-wrapper">
        <div class="message-label">{{ msg.role === 'user' ? '👤 你' : '🤖 Agent' }}</div>
        <div class="message-content">{{ msg.content }}</div>
      </div>
    </div>

    <div v-if="loading" class="loading">
      ⏳ Agent 正在思考...
    </div>
  </div>

  <div class="input-container">
    <input 
      v-model="userInput" 
      @keyup.enter="sendMessage"
      :disabled="loading"
      class="input-box" 
      type="text" 
      placeholder="输入你的问题..."
    />
    <button 
      @click="sendMessage" 
      :disabled="loading || !userInput.trim()"
      class="send-button"
    >
      {{ loading ? '发送中...' : '发送' }}
    </button>
  </div>

  <div class="info-footer">
    会话 ID: {{ threadId }} | 消息数: {{ messageCount }}
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import axios from 'axios'

const messages = ref([])
const userInput = ref('')
const loading = ref(false)
const threadId = ref(generateThreadId())
const messageCount = ref(0)
const chatContainer = ref(null)

function generateThreadId() {
  return 'web_' + Date.now()
}

function startNewChat() {
  if (confirm('确定要开始新对话吗？当前对话历史将被清空。')) {
    messages.value = []
    threadId.value = generateThreadId()
    messageCount.value = 0
  }
}

async function sendMessage() {
  const message = userInput.value.trim()
  if (!message || loading.value) return

  messages.value.push({
    role: 'user',
    content: message
  })

  userInput.value = ''
  loading.value = true

  nextTick(() => scrollToBottom())

  try {
    const response = await axios.post('/chat', {
      message: message,
      thread_id: threadId.value
    })

    messages.value.push({
      role: 'agent',
      content: response.data.answer
    })

    messageCount.value = response.data.message_count

    nextTick(() => scrollToBottom())

  } catch (error) {
    console.error('Error:', error)
    messages.value.push({
      role: 'agent',
      content: '❌ 抱歉，发生了错误: ' + error.message
    })
  } finally {
    loading.value = false
  }
}

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  text-align: center;
}

.header h1 {
  font-size: 24px;
  margin-bottom: 5px;
}

.header p {
  font-size: 14px;
  opacity: 0.9;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f7f7f7;
}

.message {
  margin-bottom: 16px;
  display: flex;
  animation: fadeIn 0.3s;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.user {
  justify-content: flex-end;
}

.message-wrapper {
  max-width: 70%;
  display: flex;
  flex-direction: column;
}

.message-content {
  padding: 12px 16px;
  border-radius: 12px;
  word-break: break-word;
  overflow-wrap: break-word;
}

.message.user .message-content {
  background: #667eea;
  color: white;
}

.message.agent .message-content {
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
}

.message-label {
  font-size: 12px;
  margin-bottom: 4px;
  opacity: 0.7;
}

.input-container {
  padding: 20px;
  background: white;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 10px;
}

.input-box {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e0e0e0;
  border-radius: 24px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.3s;
}

.input-box:focus {
  border-color: #667eea;
}

.send-button {
  padding: 12px 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 24px;
  font-size: 14px;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.send-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading {
  text-align: center;
  padding: 20px;
  color: #999;
}

.new-chat-button {
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 16px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.3s;
}

.new-chat-button:hover {
  background: rgba(255, 255, 255, 0.3);
}

.info-footer {
  font-size: 12px;
  color: #999;
  text-align: center;
  padding: 10px;
  background: white;
  border-top: 1px solid #e0e0e0;
}
</style>
