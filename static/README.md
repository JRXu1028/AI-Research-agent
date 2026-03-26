# 前端静态文件

## 文件说明

- **index.html** - Web 聊天界面（Vue.js 3）

## 技术栈

- Vue.js 3（CDN 版本，无需构建）
- 原生 JavaScript
- CSS3（渐变、动画）

## 自定义

### 修改样式

编辑 `index.html` 中的 `<style>` 部分：

```css
/* 修改主题色 */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* 修改消息气泡颜色 */
.message.user .message-content {
    background: #667eea;  /* 用户消息颜色 */
}
```

### 修改功能

编辑 `index.html` 中的 Vue 组件：

```javascript
methods: {
    // 添加新方法
    exportChat() {
        // 导出对话记录
    }
}
```

## 注意事项

- 使用 CDN 加载 Vue.js，需要网络连接
- 如需离线使用，下载 Vue.js 到本地
- API 地址硬编码为 `http://localhost:8000`
