/**
 * Orbit - Pharmaceutical Intelligence Chat Interface
 * Powered by Senture
 */

// State management
const state = {
    conversationId: generateId(),
    context: null,
    isLoading: false
};

// DOM Elements
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    autoResizeTextarea();
});

function setupEventListeners() {
    chatForm.addEventListener('submit', handleSubmit);
    userInput.addEventListener('keydown', handleKeydown);
    userInput.addEventListener('input', autoResizeTextarea);
}

function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
}

function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
}

async function handleSubmit(e) {
    e.preventDefault();

    const message = userInput.value.trim();
    if (!message || state.isLoading) return;

    // Clear welcome message if present
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    // Add user message
    addMessage(message, 'user');

    // Clear input
    userInput.value = '';
    autoResizeTextarea();

    // Send to API
    await sendMessage(message);
}

function sendSuggestion(text) {
    userInput.value = text;
    chatForm.dispatchEvent(new Event('submit'));
}

function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatarSvg = role === 'user'
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
             <circle cx="12" cy="7" r="4"/>
           </svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <circle cx="12" cy="12" r="10"/>
             <circle cx="12" cy="12" r="4"/>
             <path d="M12 2v2"/>
             <path d="M12 20v2"/>
             <path d="M2 12h2"/>
             <path d="M20 12h2"/>
           </svg>`;

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarSvg}</div>
        <div class="message-content">${formatMessage(content)}</div>
    `;

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator';

    typingDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <circle cx="12" cy="12" r="4"/>
                <path d="M12 2v2"/>
                <path d="M12 20v2"/>
                <path d="M2 12h2"/>
                <path d="M20 12h2"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    messagesContainer.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

async function sendMessage(message) {
    state.isLoading = true;
    sendBtn.disabled = true;

    addTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                conversation_id: state.conversationId,
                context: state.context
            })
        });

        removeTypingIndicator();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update context for multi-turn
        if (data.context) {
            state.context = data.context;
        }

        // Add assistant response
        addMessage(data.response || data.output || 'I apologize, but I encountered an issue processing your request.', 'assistant');

    } catch (error) {
        removeTypingIndicator();
        console.error('Error sending message:', error);
        addMessage('I apologize, but I encountered an error. Please try again.', 'assistant');
    } finally {
        state.isLoading = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

function formatMessage(content) {
    // Basic markdown-like formatting
    let formatted = escapeHtml(content);

    // Bold
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Code blocks
    formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // Inline code
    formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');

    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function generateId() {
    return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Expose sendSuggestion to window for onclick handlers
window.sendSuggestion = sendSuggestion;
