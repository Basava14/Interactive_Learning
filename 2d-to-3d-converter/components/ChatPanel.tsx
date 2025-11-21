'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Download, Loader2, MessageCircle, Sparkles } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatPanelProps {
  imageName: string;
  summary: string;
  isGeneratingSummary: boolean;
}

export default function ChatPanel({ imageName, summary, isGeneratingSummary }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    
    const newMessages: Message[] = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_name: imageName,
          conversation_history: newMessages,
          user_message: userMessage,
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setMessages([...newMessages, { role: 'assistant', content: data.response }]);
      } else {
        throw new Error(data.detail || 'Failed to get response');
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages([
        ...newMessages,
        { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportReport = async () => {
    setIsExporting(true);
    try {
      const response = await fetch('http://localhost:8000/api/export-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_name: imageName,
          summary: summary,
          conversation_history: messages,
        }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `learning_report_${imageName.replace(/\s+/g, '_')}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        throw new Error('Failed to export report');
      }
    } catch (error) {
      console.error('Export error:', error);
      alert('Failed to export report. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-2xl border border-gray-200/50 overflow-hidden backdrop-blur-sm">
      {/* Header with Gradient */}
      <div className="px-6 py-5 border-b border-gray-200/50 bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 flex-shrink-0 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                Learning Assistant
                <Sparkles className="w-4 h-4 text-yellow-300 animate-pulse" />
              </h2>
              <p className="text-xs text-blue-100 mt-0.5">AI-Powered Educational Support</p>
            </div>
          </div>
          <button
            onClick={handleExportReport}
            disabled={isExporting || !summary}
            className="group flex items-center gap-2 px-4 py-2.5 bg-white text-blue-700 rounded-xl hover:bg-blue-50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-semibold shadow-lg hover:shadow-xl hover:scale-105 active:scale-95"
          >
            {isExporting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4 group-hover:animate-bounce" />
            )}
            Export
          </button>
        </div>
      </div>

      {/* Summary Section with Scrollable Content - FIXED */}
      <div className="border-b border-indigo-100/50 flex-shrink-0 shadow-sm" style={{ height: '220px' }}>
        <div className="px-6 py-5 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 h-full">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-6 bg-gradient-to-b from-blue-600 to-indigo-600 rounded-full"></div>
            <h3 className="text-sm font-bold text-gray-800 uppercase tracking-wider">
              Summary
            </h3>
          </div>
          {isGeneratingSummary ? (
            <div className="flex items-center gap-3 text-blue-600 bg-white/60 backdrop-blur-sm p-4 rounded-xl border border-blue-200/50">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm font-medium">Generating AI-powered summary...</span>
            </div>
          ) : (
            <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-indigo-200/50 shadow-sm" style={{ height: 'calc(100% - 25px)' }}>
              <div className="p-4 h-full overflow-y-auto">
                <p className="text-gray-700 text-sm leading-relaxed">
                  {summary || 'No summary available yet.'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chat Messages - Fixed height calculation */}
      <div 
        ref={chatContainerRef}
        className="overflow-y-auto px-6 py-5 space-y-4 bg-gradient-to-b from-transparent to-gray-50/30"
        style={{ height: 'calc(100vh - 12rem - 100px - 200px - 120px)' }}
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-indigo-400 rounded-full blur-2xl opacity-20 animate-pulse"></div>
              <div className="relative p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-3xl border-2 border-blue-200/50 shadow-lg">
                <MessageCircle className="w-16 h-16 text-blue-600" />
              </div>
            </div>
            <h4 className="text-lg font-bold text-gray-800 mb-2">Start Your Learning Journey</h4>
            <p className="text-sm text-gray-600 mb-4 max-w-xs">
              Ask questions about this image to deepen your understanding
            </p>
            <div className="flex flex-wrap gap-2 justify-center max-w-md">
              <span className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                "What are the main features?"
              </span>
              <span className="px-3 py-1.5 bg-indigo-100 text-indigo-700 rounded-full text-xs font-medium">
                "Explain this concept"
              </span>
              <span className="px-3 py-1.5 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                "Tell me more"
              </span>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-5 py-3.5 shadow-lg transition-all duration-300 hover:shadow-xl ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-br-sm'
                      : 'bg-white text-gray-800 rounded-bl-sm border border-gray-200'
                  }`}
                >
                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-200">
                      <Sparkles className="w-3.5 h-3.5 text-blue-600" />
                      <span className="text-xs font-semibold text-blue-600">AI Assistant</span>
                    </div>
                  )}
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start animate-fadeIn">
                <div className="bg-white rounded-2xl px-5 py-4 rounded-bl-sm border border-gray-200 shadow-lg">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                    <span className="text-sm text-gray-600">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area with Modern Design */}
      <div className="px-6 py-5 border-t border-gray-200/50 bg-gradient-to-r from-gray-50 to-white flex-shrink-0 shadow-lg" style={{ height: '120px' }}>
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about this image..."
              disabled={isLoading || isGeneratingSummary}
              className="w-full px-5 py-3.5 pr-12 border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed text-sm transition-all duration-300 shadow-sm hover:shadow-md bg-white"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <MessageCircle className="w-5 h-5 text-gray-400" />
            </div>
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading || isGeneratingSummary}
            className="group px-6 py-3.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 font-semibold shadow-lg hover:shadow-xl hover:scale-105 active:scale-95"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Send className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" />
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}