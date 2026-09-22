import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Send,
  Plus,
  Trash2,
  CheckCircle2,
  FolderGit2,
  LogOut,
} from "lucide-react";
import ChatMessage from "./ChatMessage";
import { api, Document, AgentStep } from "../services/api";
import toast from "react-hot-toast";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rationale?: string;
  contexts?: Document[];
  steps?: AgentStep[];
}

interface Conversation {
  id: string;
  repoUrl: string;
  title: string;
  messages: Message[];
  createdAt: string;
}

interface ChatPageProps {
  initialRepoUrl?: string;
  onLogout?: () => void;
}

const STORAGE_KEY = "github_chat_conversations";
const ACTIVE_CONV_KEY = "github_chat_active_conversation";

const ChatPage: React.FC<ChatPageProps> = ({ initialRepoUrl, onLogout }) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(
    null,
  );
  const [agentStatus, setAgentStatus] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load conversations from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      const savedActiveId = localStorage.getItem(ACTIVE_CONV_KEY);

      if (saved) {
        const parsed: Conversation[] = JSON.parse(saved);
        setConversations(parsed);

        if (savedActiveId && parsed.some((c) => c.id === savedActiveId)) {
          setActiveConversationId(savedActiveId);
        } else if (parsed.length > 0) {
          setActiveConversationId(parsed[0].id);
        }
      } else if (initialRepoUrl) {
        const newConv: Conversation = {
          id: crypto.randomUUID(),
          repoUrl: initialRepoUrl,
          title:
            initialRepoUrl.split("/").slice(-2).join("/") || initialRepoUrl,
          messages: [],
          createdAt: new Date().toISOString(),
        };
        setConversations([newConv]);
        setActiveConversationId(newConv.id);
      }
    } catch (e) {
      console.error("Failed to load conversations from localStorage:", e);
    }
  }, [initialRepoUrl]);

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    if (conversations.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
      } catch (e) {
        console.error("Failed to save conversations to localStorage:", e);
      }
    }
  }, [conversations]);

  // Save active conversation ID
  useEffect(() => {
    if (activeConversationId) {
      localStorage.setItem(ACTIVE_CONV_KEY, activeConversationId);
    }
  }, [activeConversationId]);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversations, activeConversationId, streamingMessageId, agentStatus]);

  const activeConversation = conversations.find(
    (c) => c.id === activeConversationId,
  );

  // Switch conversation and sync backend context
  const handleSelectConversation = async (convId: string) => {
    if (convId === activeConversationId) return;

    setActiveConversationId(convId);
    const targetConv = conversations.find((c) => c.id === convId);

    if (targetConv && targetConv.messages.length > 0) {
      try {
        const contextMessages = targetConv.messages
          .filter((m) => m.content.trim().length > 0)
          .map((m) => ({ role: m.role, content: m.content }));

        if (contextMessages.length > 0) {
          await api.setContext(contextMessages);
          console.log(
            `[Context Sync] Restored ${contextMessages.length} turns for conversation ${convId}`,
          );
        } else {
          await api.clearMemory();
        }
      } catch (e) {
        console.warn("[Context Sync] Failed to sync context with backend:", e);
      }
    } else {
      try {
        await api.clearMemory();
      } catch (e) {
        console.warn("[Context Sync] Failed to clear memory:", e);
      }
    }
  };

  // Send a message using real-time SSE streaming
  const handleSendMessage = useCallback(async () => {
    if (
      !inputMessage.trim() ||
      !activeConversation ||
      !activeConversationId ||
      isLoading
    ) {
      return;
    }

    const userContent = inputMessage.trim();
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: userContent,
    };

    const assistantMessageId = crypto.randomUUID();
    const assistantPlaceholder: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      steps: [],
    };

    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === activeConversationId
          ? {
              ...conv,
              messages: [...conv.messages, userMessage, assistantPlaceholder],
            }
          : conv,
      ),
    );

    setInputMessage("");
    setIsLoading(true);
    setStreamingMessageId(assistantMessageId);
    setAgentStatus("Routing query intent...");

    try {
      const startTime = performance.now();

      await api.queryRepoStream(activeConversation.repoUrl, userContent, {
        onStatus: (status) => {
          setAgentStatus(status.message);
        },
        onStep: (step) => {
          setConversations((prev) =>
            prev.map((conv) => {
              if (conv.id !== activeConversationId) return conv;
              return {
                ...conv,
                messages: conv.messages.map((m) => {
                  if (m.id !== assistantMessageId) return m;
                  const existingSteps = m.steps ? [...m.steps] : [];
                  const stepIdx = existingSteps.findIndex(
                    (s) => s.id === step.id,
                  );
                  if (stepIdx >= 0) {
                    existingSteps[stepIdx] = step;
                  } else {
                    existingSteps.push(step);
                  }
                  return { ...m, steps: existingSteps };
                }),
              };
            }),
          );
          if (step.detail) {
            setAgentStatus(step.title);
          }
        },
        onToken: (token) => {
          setConversations((prev) =>
            prev.map((conv) => {
              if (conv.id !== activeConversationId) return conv;
              return {
                ...conv,
                messages: conv.messages.map((m) => {
                  if (m.id !== assistantMessageId) return m;
                  return { ...m, content: m.content + token };
                }),
              };
            }),
          );
        },
        onDone: (data) => {
          const endTime = performance.now();
          console.log(
            `[Stream Done] Completed in ${(endTime - startTime).toFixed(2)}ms`,
          );
          setConversations((prev) =>
            prev.map((conv) => {
              if (conv.id !== activeConversationId) return conv;
              return {
                ...conv,
                messages: conv.messages.map((m) => {
                  if (m.id !== assistantMessageId) return m;
                  return {
                    ...m,
                    content: data.answer || m.content,
                    rationale: data.rationale,
                    contexts: data.contexts,
                    steps: data.steps || m.steps,
                  };
                }),
              };
            }),
          );
          setIsLoading(false);
          setAgentStatus("");
          setStreamingMessageId(null);
        },
        onError: (errMsg) => {
          console.error("[Stream Error]", errMsg);
          toast.error(errMsg);
          setIsLoading(false);
          setAgentStatus("");
          setStreamingMessageId(null);
        },
      });
    } catch (error) {
      console.error("[Query Stream] Error:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Failed to get response";
      toast.error(errorMessage);
      setConversations((prev) =>
        prev.map((conv) => {
          if (conv.id !== activeConversationId) return conv;
          return {
            ...conv,
            messages: conv.messages.filter(
              (m) => m.id !== assistantMessageId || m.content.length > 0,
            ),
          };
        }),
      );
    } finally {
      setIsLoading(false);
      setAgentStatus("");
      setStreamingMessageId(null);
    }
  }, [inputMessage, activeConversation, activeConversationId, isLoading]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewChat = async () => {
    if (!activeConversation) return;

    try {
      await api.clearMemory();
    } catch (e) {
      console.warn("Failed to clear backend memory:", e);
    }

    const newConv: Conversation = {
      id: crypto.randomUUID(),
      repoUrl: activeConversation.repoUrl,
      title: activeConversation.title,
      messages: [],
      createdAt: new Date().toISOString(),
    };

    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
    toast.success("Started a new conversation");
  };

  const handleDeleteConversation = (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = conversations.filter((c) => c.id !== convId);
    setConversations(updated);

    if (updated.length === 0) {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(ACTIVE_CONV_KEY);
      setActiveConversationId(null);
      if (onLogout) onLogout();
    } else if (activeConversationId === convId) {
      setActiveConversationId(updated[0].id);
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <div className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FolderGit2 className="h-6 w-6 text-gray-300" />
            <span className="font-semibold text-gray-200">GithubChat</span>
          </div>
          {onLogout && (
            <button
              onClick={onLogout}
              title="Switch Repository"
              className="p-1.5 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-gray-200 transition-colors">
              <LogOut className="h-4 w-4" />
            </button>
          )}
        </div>

        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg text-sm font-medium transition-colors border border-gray-700">
            <Plus className="h-4 w-4" />
            <span>New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          <div className="px-2 py-1 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Conversations
          </div>
          {conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => handleSelectConversation(conv.id)}
              className={`group flex items-center justify-between px-3 py-2.5 rounded-lg text-sm cursor-pointer transition-colors ${
                conv.id === activeConversationId
                  ? "bg-gray-800 text-white font-medium"
                  : "text-gray-400 hover:bg-gray-800/60 hover:text-gray-200"
              }`}>
              <span className="truncate flex-1">{conv.title}</span>
              <button
                onClick={(e) => handleDeleteConversation(conv.id, e)}
                title="Delete conversation"
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity">
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>

        {activeConversation && (
          <div className="p-3 border-t border-gray-800 bg-gray-900/50">
            <div
              className="text-xs text-gray-500 truncate"
              title={activeConversation.repoUrl}>
              Active Repo:{" "}
              <span className="text-gray-300 font-mono">
                {activeConversation.repoUrl}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Main chat view */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between bg-gray-900">
          <div>
            <h2 className="font-semibold text-white">
              {activeConversation?.title || "Chat"}
            </h2>
            <p className="text-xs text-gray-500 truncate max-w-lg">
              {activeConversation?.repoUrl}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-3 py-1.5 rounded-full font-medium">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>Ready</span>
          </div>
        </div>

        {/* Message list */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {!activeConversation || activeConversation.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-8">
              <FolderGit2 className="h-12 w-12 text-gray-600 mb-4" />
              <h3 className="text-lg font-semibold text-gray-300 mb-2">
                Ask anything about this repository
              </h3>
              <p className="text-sm text-gray-500 max-w-md mb-6">
                Agentic RAG with Self-Correction, CRAG relevance grading, query
                routing, and real-time execution step streaming.
              </p>
            </div>
          ) : (
            <>
              {activeConversation.messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  rationale={message.rationale}
                  contexts={message.contexts}
                  steps={message.steps}
                  isLoading={message.id === streamingMessageId && isLoading}
                  statusMessage={
                    message.id === streamingMessageId ? agentStatus : undefined
                  }
                />
              ))}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="p-4 border-t border-gray-800 bg-gray-900">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-end bg-gray-800/80 border border-gray-700/80 rounded-2xl p-3 shadow-xl focus-within:border-blue-500/80 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all backdrop-blur-sm">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about this repository's code, architecture, or flow... (Shift + Enter for new line)"
                rows={3}
                className="flex-1 px-3 py-1.5 bg-transparent text-white placeholder-gray-500 focus:outline-none resize-none font-sans text-sm min-h-[76px] max-h-40 leading-relaxed scrollbar-thin scrollbar-thumb-gray-700"
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="h-11 w-11 flex items-center justify-center bg-blue-600 hover:bg-blue-500 text-white rounded-xl disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/40 flex-shrink-0 mb-0.5">
                <Send className="h-5 w-5" />
              </button>
            </div>
            <div className="flex justify-between items-center px-2 mt-2 text-[11px] text-gray-500 font-mono">
              <span>
                Press{" "}
                <kbd className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-gray-400 text-[10px]">
                  Enter
                </kbd>{" "}
                to send,{" "}
                <kbd className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-gray-400 text-[10px]">
                  Shift + Enter
                </kbd>{" "}
                for multi-line
              </span>
              <span>
                {inputMessage.length > 0
                  ? `${inputMessage.length} characters`
                  : ""}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
