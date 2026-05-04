import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, ShieldCheck, AlertCircle, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Greeting. I am your Legal AI Assistant specialized in Indian Criminal Law. How can I assist you with the Indian Penal Code, Evidence Act, or CrPC today?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [status, setStatus] = useState('Online');
  const [sessionId, setSessionId] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Generate or retrieve session ID
    let sId = localStorage.getItem('legal_session_id');
    if (!sId) {
      sId = 'sess_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('legal_session_id', sId);
    }
    setSessionId(sId);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsTyping(true);
    setStatus('Analyzing legal context...');

    try {
      // Simulate real-time states
      setTimeout(() => {
        if (isTyping) setStatus('Consulting law books...');
      }, 600);
      
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: userMessage })
      });

      const data = await response.json();
      
      setStatus('Online');
      setIsTyping(false);

      // Simulate streaming response
      simulateStreaming(data.message, data.disclaimer);
    } catch (error) {
      console.error('Error:', error);
      setIsTyping(false);
      setStatus('Online');
      setMessages(prev => [...prev, { role: 'assistant', content: "I apologize, but I am currently unable to provide a response due to technical difficulties. Please consult a professional advisor." }]);
    }
  };

  const handleFeedback = async (index, type) => {
    try {
      await fetch('http://localhost:8000/api/v1/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message_index: index, feedback: type })
      });
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[index].feedbackProvided = true;
        return newMessages;
      });
    } catch (e) {
      console.error('Feedback error:', e);
    }
  };

  const simulateStreaming = (text, disclaimer) => {
    let currentText = '';
    const words = text.split(' ');
    let i = 0;
    
    setMessages(prev => [...prev, { role: 'assistant', content: '', disclaimer: '', isStreaming: true }]);
    
    const interval = setInterval(() => {
      if (i < words.length) {
        currentText += (i === 0 ? '' : ' ') + words[i];
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].content = currentText;
          return newMessages;
        });
        i++;
      } else {
        clearInterval(interval);
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].isStreaming = false;
          newMessages[newMessages.length - 1].disclaimer = disclaimer;
          return newMessages;
        });
      }
    }, 50);
  };

  return (
    <div className="app-container">
      <div className="chat-panel">
        <header className="chat-header" style={{ background: 'linear-gradient(135deg, #2c3e50 0%, #000000 100%)' }}>
          <div className="agent-status">
            <div className={`status-dot ${status !== 'Online' ? 'pulse' : ''}`}></div>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Legal Assistant AI</h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{status}</p>
            </div>
          </div>
          <div style={{ opacity: 0.8, color: '#f1c40f' }}>
            <ShieldCheck size={24} />
          </div>
        </header>

        <main className="chat-messages">
          <AnimatePresence>
            {messages.map((msg, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className={`message ${msg.role}`}
              >
                <div style={{ marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px', opacity: 0.6, fontSize: '0.75rem' }}>
                  {msg.role === 'assistant' ? <Bot size={14} /> : <User size={14} />}
                  <span>{msg.role === 'assistant' ? 'Legal AI' : 'Consultant'}</span>
                </div>
                <div className="message-content" style={{ whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </div>
                
                {msg.disclaimer && (
                  <div className="disclaimer-text" style={{ marginTop: '12px', fontSize: '0.7rem', color: '#f87171', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '8px' }}>
                    {msg.disclaimer}
                  </div>
                )}
                
                {msg.role === 'assistant' && !msg.isStreaming && !msg.feedbackProvided && index > 0 && (
                  <div style={{ display: 'flex', gap: '8px', marginTop: '8px', justifyContent: 'flex-end', opacity: 0.5 }}>
                    <button onClick={() => handleFeedback(index, 'positive')} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}><Send size={12} style={{ transform: 'rotate(-45deg)' }} /></button>
                    <button onClick={() => handleFeedback(index, 'negative')} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}><AlertCircle size={12} /></button>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          
          {isTyping && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="message assistant">
              <div className="thinking-indicator">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </main>

        <footer className="chat-input-area">
          <div className="input-wrapper">
            <input
              type="text"
              placeholder="Ask about IPC Section, Evidence Act, or Legal query..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            />
            <button className="send-button" onClick={handleSend} disabled={isTyping}>
              {isTyping ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
            </button>
          </div>
          <p style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
            Strictly grounded by official Indian Criminal Law documents.
          </p>
        </footer>
      </div>
    </div>
  );
};

export default App;
