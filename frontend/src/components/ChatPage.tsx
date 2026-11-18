import React, { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Send,
  Plus,
  Trash2,
  FolderGit2,
  LogOut,
  X,
  Loader2,
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
  const location = useLocation();
  const navigate = useNavigate();
  const stateRepoUrl = (location.state as { repoUrl?: string } | null)?.repoUrl;
  const effectiveRepoUrl = initialRepoUrl || stateRepoUrl;

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
  const [isRepoModalOpen, setIsRepoModalOpen] = useState(false);
  const [newRepoUrlInput, setNewRepoUrlInput] = useState("");
  const [isInitializingRepo, setIsInitializingRepo] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load conversations from localStorage or initialize with effectiveRepoUrl
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      const savedActiveId = localStorage.getItem(ACTIVE_CONV_KEY);
      let parsed: Conversation[] = [];

      if (saved) {
        try {
          parsed = JSON.parse(saved);
        } catch {
          parsed = [];
        }
      }

      let activeConv: Conversation | undefined;

      if (effectiveRepoUrl) {
        const existingConv =
          (savedActiveId &&
            parsed.find(
              (c) => c.id === savedActiveId && c.repoUrl === effectiveRepoUrl,
            )) ||
          parsed.find((c) => c.repoUrl === effectiveRepoUrl);

        if (existingConv) {
          setConversations(parsed);
          setActiveConversationId(existingConv.id);
          activeConv = existingConv;
        } else {
          const newConv: Conversation = {
            id: crypto.randomUUID(),
            repoUrl: effectiveRepoUrl,
            title:
              effectiveRepoUrl.split("/").slice(-2).join("/") || effectiveRepoUrl,
            messages: [],
            createdAt: new Date().toISOString(),
          };
          const updated = [newConv, ...parsed];
          setConversations(updated);
          setActiveConversationId(newConv.id);
          activeConv = newConv;
        }
      } else if (parsed.length > 0) {
        setConversations(parsed);
        const selectedId =
          savedActiveId && parsed.some((c) => c.id === savedActiveId)
            ? savedActiveId
            : parsed[0].id;
        setActiveConversationId(selectedId);
        activeConv = parsed.find((c) => c.id === selectedId);
      } else {
        navigate("/", { replace: true });
        return;
      }

      // Sync backend conversation context on load/refresh if messages exist
      if (activeConv && activeConv.messages.length > 0) {
        const contextMessages = activeConv.messages
          .filter((m) => m.content.trim().length > 0)
          .map((m) => ({ role: m.role, content: m.content }));

        if (contextMessages.length > 0) {
          api.setContext(contextMessages).catch((err) =>
            console.warn("[Init Context] Could not restore context:", err),
          );
        }
      }
    } catch (e) {
      console.error("Failed to load conversations from localStorage:", e);
    }
  }, [effectiveRepoUrl, navigate]);

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
    const trimmedInput = inputMessage.trim();
    if (
      !trimmedInput ||
      isLoading
    ) {
      return;
    }

    if (!activeConversation || !activeConversationId) {
      toast.error("No active conversation found. Please initialize a repository first.");
      return;
    }

    const userContent = trimmedInput;
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    } else {
      navigate("/");
    }
  };

  const handleConnectNewRepo = async (targetUrl?: string) => {
    const urlToUse = (targetUrl || newRepoUrlInput).trim();
    if (!urlToUse) {
      toast.error("Please enter a valid GitHub repository URL");
      return;
    }
    if (!urlToUse.startsWith("http://") && !urlToUse.startsWith("https://")) {
      toast.error("Repository URL must start with https://");
      return;
    }

    setIsInitializingRepo(true);
    try {
      await api.initRepo(urlToUse);
      toast.success("Repository connected! Starting new chat session...");

      try {
        await api.clearMemory();
      } catch (e) {
        console.warn("Failed to clear backend memory:", e);
      }

      const repoTitle =
        urlToUse.split("/").slice(-2).join("/") || urlToUse;

      const newConv: Conversation = {
        id: crypto.randomUUID(),
        repoUrl: urlToUse,
        title: repoTitle,
        messages: [],
        createdAt: new Date().toISOString(),
      };

      setConversations((prev) => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
      setIsRepoModalOpen(false);
      setNewRepoUrlInput("");
    } catch (error) {
      console.error("Failed to connect new repo:", error);
      const errMsg =
        error instanceof Error ? error.message : "Failed to initialize repository";
      toast.error(errMsg);
    } finally {
      setIsInitializingRepo(false);
    }
  };

  const recentRepos = Array.from(
    new Set(conversations.map((c) => c.repoUrl)),
  ).filter((url) => Boolean(url && url.trim()));

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
      handleLogout();
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
          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsRepoModalOpen(true)}
              title="Connect a new repository"
              className="p-1.5 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-gray-200 transition-colors">
              <FolderGit2 className="h-4 w-4" />
            </button>
            <button
              onClick={handleLogout}
              title="Return to Home"
              className="p-1.5 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-gray-200 transition-colors">
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="p-3 space-y-2">
          <button
            onClick={handleNewChat}
            title="Start a new chat for this repository"
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors shadow-sm">
            <Plus className="h-4 w-4" />
            <span>New Chat</span>
          </button>
          <button
            onClick={() => setIsRepoModalOpen(true)}
            title="Switch repository or connect a new URL"
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white rounded-lg text-sm font-medium transition-colors border border-gray-700">
            <FolderGit2 className="h-4 w-4 text-blue-400" />
            <span>Switch / New Repo</span>
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
              <div className="truncate flex-1 pr-2">
                <div className="truncate font-medium">{conv.title}</div>
                <div className="text-[10px] text-gray-500 font-mono truncate">
                  {conv.repoUrl.replace("https://github.com/", "")}
                </div>
              </div>
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
          <div className="p-3 border-t border-gray-800 bg-gray-900/50 flex items-center justify-between">
            <div
              className="text-xs text-gray-500 truncate min-w-0 flex-1 mr-2"
              title={activeConversation.repoUrl}>
              <div className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">Active Repo</div>
              <div className="text-gray-300 font-mono truncate text-xs">
                {activeConversation.repoUrl.replace("https://github.com/", "")}
              </div>
            </div>
            <button
              onClick={() => setIsRepoModalOpen(true)}
              className="text-xs text-blue-400 hover:text-blue-300 font-medium px-2 py-1 rounded hover:bg-gray-800 transition-colors flex-shrink-0">
              Change
            </button>
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
          <button
            onClick={() => setIsRepoModalOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white rounded-lg text-xs font-medium border border-gray-700 transition-colors shadow-sm">
            <FolderGit2 className="h-3.5 w-3.5 text-blue-400" />
            <span>Switch Repo</span>
          </button>
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
            <div className="flex gap-3 items-center bg-gray-800/80 border border-gray-700/80 rounded-2xl p-3 shadow-xl focus-within:border-blue-500/80 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all backdrop-blur-sm">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about this repository's code, architecture, or flow... (Shift + Enter for new line)"
                rows={2}
                className="flex-1 px-3 py-1.5 bg-transparent text-white placeholder-gray-500 focus:outline-none resize-none font-sans text-sm min-h-[50px] max-h-40 leading-relaxed scrollbar-thin scrollbar-thumb-gray-700"
              />
              <button
                type="button"
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                aria-label="Send message"
                className="h-11 w-11 flex items-center justify-center bg-blue-600 hover:bg-blue-500 text-white rounded-xl disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/40 flex-shrink-0">
                <Send className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Switch / New Repository Modal */}
      {isRepoModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl w-full max-w-lg p-6 shadow-2xl relative space-y-5 animate-in fade-in duration-150">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl text-blue-400">
                  <FolderGit2 className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    Connect New Repository
                  </h3>
                  <p className="text-xs text-gray-400">
                    Input a GitHub URL to index code and start a new chat
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  if (!isInitializingRepo) setIsRepoModalOpen(false);
                }}
                disabled={isInitializingRepo}
                className="p-1 text-gray-400 hover:text-gray-200 rounded-lg hover:bg-gray-800 transition-colors disabled:opacity-50">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div>
              <label
                htmlFor="new-repo-url-input"
                className="block text-xs font-medium text-gray-300 mb-2">
                GitHub Repository URL
              </label>
              <input
                id="new-repo-url-input"
                type="text"
                autoFocus
                disabled={isInitializingRepo}
                value={newRepoUrlInput}
                onChange={(e) => setNewRepoUrlInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isInitializingRepo) {
                    e.preventDefault();
                    handleConnectNewRepo();
                  }
                }}
                placeholder="https://github.com/owner/repository"
                className="w-full px-3.5 py-2.5 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-500 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all font-mono disabled:opacity-50"
              />
            </div>

            {recentRepos.length > 0 && (
              <div>
                <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">
                  Previously Used Repositories
                </div>
                <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
                  {recentRepos.map((url) => (
                    <button
                      key={url}
                      type="button"
                      disabled={isInitializingRepo}
                      onClick={() => setNewRepoUrlInput(url)}
                      className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 font-mono truncate max-w-xs transition-colors text-left disabled:opacity-50">
                      {url.replace("https://github.com/", "")}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                disabled={isInitializingRepo}
                onClick={() => setIsRepoModalOpen(false)}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-xl text-sm font-medium transition-colors border border-gray-700 disabled:opacity-50">
                Cancel
              </button>
              <button
                type="button"
                disabled={!newRepoUrlInput.trim() || isInitializingRepo}
                onClick={() => handleConnectNewRepo()}
                className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-sm font-medium transition-colors shadow-lg shadow-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed">
                {isInitializingRepo ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Indexing Repository...</span>
                  </>
                ) : (
                  <>
                    <FolderGit2 className="h-4 w-4" />
                    <span>Connect & Start Chat</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatPage;
