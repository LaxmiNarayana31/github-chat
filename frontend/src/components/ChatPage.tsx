import React, { useState, useCallback, useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Send, Github } from "lucide-react";
import { toast } from "react-hot-toast";
import Sidebar from "./Sidebar";
import ChatMessage from "./ChatMessage";

interface DocumentMetadata {
  file_path: string;
  type: string;
  is_code: boolean;
  is_implementation: boolean;
  title: string;
}

interface Document {
  text: string;
  meta_data: DocumentMetadata;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rationale?: string;
  contexts?: Document[];
}

interface Conversation {
  id: string;
  repoUrl: string;
  repoName: string;
  messages: Message[];
  createdAt: string;
}

const STORAGE_KEY = "github-chat-conversations";

const ChatPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);

  // Get repo URL from navigation state
  const repoUrl = (location.state as { repoUrl?: string })?.repoUrl || "";

  const getRepoName = (url: string) => {
    try {
      const parts = url.split("/");
      return parts[parts.length - 1] || parts[parts.length - 2] || "Repository";
    } catch {
      return "Repository";
    }
  };

  // Clear localStorage when browser/tab closes
  useEffect(() => {
    const handleBeforeUnload = () => {
      localStorage.removeItem(STORAGE_KEY);
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  // Load conversations from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    let loadedConversations: Conversation[] = [];
    
    if (stored) {
      try {
        loadedConversations = JSON.parse(stored);
      } catch (e) {
        console.error("Failed to parse stored conversations:", e);
        localStorage.removeItem(STORAGE_KEY);
      }
    }

    // If we have a repoUrl from navigation, create new conversation
    if (repoUrl) {
      const newConversation: Conversation = {
        id: crypto.randomUUID(),
        repoUrl,
        repoName: getRepoName(repoUrl),
        messages: [],
        createdAt: new Date().toISOString(),
      };
      loadedConversations = [newConversation, ...loadedConversations];
      setConversations(loadedConversations);
      setActiveConversationId(newConversation.id);
    } else if (loadedConversations.length > 0) {
      // Otherwise load stored and select most recent
      setConversations(loadedConversations);
      setActiveConversationId(loadedConversations[0].id);
    }
    
    setInitialized(true);
  }, [repoUrl]);

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    if (initialized && conversations.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    }
  }, [conversations, initialized]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, activeConversationId]);

  const activeConversation = conversations.find((c) => c.id === activeConversationId);

  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim() || !activeConversation || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: inputMessage.trim(),
    };

    // Add user message to conversation
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === activeConversationId
          ? { ...conv, messages: [...conv.messages, userMessage] }
          : conv
      )
    );
    setInputMessage("");
    setIsLoading(true);

    try {
      console.log("🚀 Sending request to backend...");
      console.log("📦 Request payload:", {
        repo_url: activeConversation.repoUrl,
        query: userMessage.content,
      });
      const startTime = performance.now();

      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          repo_url: activeConversation.repoUrl,
          query: userMessage.content,
        }),
      });

      const endTime = performance.now();
      console.log(`⏱️ Response received in ${(endTime - startTime).toFixed(2)}ms`);

      if (!response.ok) {
        console.error("❌ Response not OK:", response.status, response.statusText);
        throw new Error("Failed to get response");
      }

      const result = await response.json();
      console.log("✅ API Response:", result);
      console.log("📝 Answer length:", result.answer?.length || 0);
      console.log("📚 Contexts count:", result.contexts?.length || 0);

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer,
        rationale: result.rationale,
        contexts: result.contexts,
      };

      // Add assistant message to conversation
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === activeConversationId
            ? { ...conv, messages: [...conv.messages, assistantMessage] }
            : conv
        )
      );
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = error instanceof Error ? error.message : "Failed to get response";
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [inputMessage, activeConversation, activeConversationId, isLoading]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // New Chat: Create new conversation in the SAME repo and clear backend memory
  const handleNewChat = async () => {
    if (!activeConversation) return;
    
    // Clear backend memory for fresh conversation
    try {
      await fetch("http://localhost:8000/clear-memory", {
        method: "POST",
      });
      console.log("🧹 Backend memory cleared");
    } catch (error) {
      console.error("Failed to clear backend memory:", error);
    }
    
    const newConversation: Conversation = {
      id: crypto.randomUUID(),
      repoUrl: activeConversation.repoUrl,
      repoName: activeConversation.repoName,
      messages: [],
      createdAt: new Date().toISOString(),
    };
    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
  };

  // New Repo: Go back to landing page to enter different repo
  const handleNewRepo = () => {
    navigate("/");
  };

  // Select a conversation from sidebar - restore its context in backend
  const handleSelectConversation = async (id: string) => {
    const conv = conversations.find((c) => c.id === id);
    if (conv && id !== activeConversationId) {
      // Restore this conversation's context in backend memory
      try {
        const messages = conv.messages.map((m) => ({
          role: m.role,
          content: m.content,
        }));
        
        await fetch("http://localhost:8000/set-context", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(messages),
        });
        console.log(`📝 Context restored for conversation: ${conv.repoName}`);
      } catch (error) {
        console.error("Failed to restore context:", error);
      }
      
      setActiveConversationId(id);
    }
  };

  const handleDeleteConversation = (id: string) => {
    const remaining = conversations.filter((c) => c.id !== id);
    setConversations(remaining);
    
    if (activeConversationId === id) {
      if (remaining.length > 0) {
        setActiveConversationId(remaining[0].id);
      } else {
        // Clear localStorage and go to landing
        localStorage.removeItem(STORAGE_KEY);
        navigate("/");
      }
    }
  };

  // Redirect to home if no conversations
  if (initialized && conversations.length === 0) {
    navigate("/");
    return null;
  }

  return (
    <div className="flex h-screen bg-gray-950">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onNewRepo={handleNewRepo}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-16 border-b border-gray-800 flex items-center px-6 gap-3 bg-gray-900">
          <Github className="h-6 w-6 text-gray-400" />
          <div>
            <h1 className="text-white font-medium">
              {activeConversation?.repoName || "Chat"}
            </h1>
            <p className="text-xs text-gray-500 truncate max-w-md">
              {activeConversation?.repoUrl}
            </p>
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeConversation?.messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="p-4 bg-gray-800/50 rounded-2xl mb-4">
                <Github className="h-12 w-12 text-gray-500" />
              </div>
              <h2 className="text-xl font-medium text-white mb-2">
                Start exploring {activeConversation.repoName}
              </h2>
              <p className="text-gray-400 max-w-md">
                Ask any question about the repository's code, structure, dependencies, or implementation details.
              </p>
            </div>
          ) : (
            <>
              {activeConversation?.messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  rationale={message.rationale}
                  contexts={message.contexts}
                />
              ))}
              {isLoading && (
                <ChatMessage role="assistant" content="" isLoading />
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area - fixed alignment */}
        <div className="p-4 border-t border-gray-800 bg-gray-900">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-center">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about this repository..."
                rows={1}
                className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-transparent resize-none"
                style={{ height: "48px" }}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="h-12 w-12 flex items-center justify-center bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all border border-gray-600"
              >
                <Send className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
