import React from "react";
import { MessageSquare, Trash2, FolderGit2 } from "lucide-react";

interface Conversation {
  id: string;
  repoUrl: string;
  repoName: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  createdAt: string;
}

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onNewRepo: () => void;
  onDeleteConversation: (id: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onNewRepo,
  onDeleteConversation,
}) => {
  const getRepoName = (url: string) => {
    try {
      const parts = url.split("/");
      return parts[parts.length - 1] || parts[parts.length - 2] || "Repository";
    } catch {
      return "Repository";
    }
  };

  const getPreview = (conv: Conversation) => {
    const firstUserMessage = conv.messages.find((m) => m.role === "user");
    if (firstUserMessage) {
      return firstUserMessage.content.slice(0, 50) + (firstUserMessage.content.length > 50 ? "..." : "");
    }
    return "New conversation";
  };

  return (
    <div className="w-72 h-screen bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Header with buttons */}
      <div className="p-4 border-b border-gray-800 space-y-2">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gray-800 text-white font-medium rounded-lg hover:bg-gray-700 border border-gray-700 transition-all"
        >
          <MessageSquare className="h-4 w-4" />
          <span>New Chat</span>
        </button>
        <button
          onClick={onNewRepo}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-transparent text-gray-400 font-medium rounded-lg hover:bg-gray-800 hover:text-white border border-gray-700 transition-all"
        >
          <FolderGit2 className="h-4 w-4" />
          <span>New Repository</span>
        </button>
      </div>

      {/* Conversations list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {conversations.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No conversations yet</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                activeConversationId === conv.id
                  ? "bg-gray-800 border border-gray-600"
                  : "bg-transparent hover:bg-gray-800/50 border border-transparent"
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium text-gray-200 truncate">
                    {getRepoName(conv.repoUrl)}
                  </h3>
                  <p className="text-xs text-gray-500 mt-1 truncate">
                    {getPreview(conv)}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-500 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-all"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="text-xs text-gray-600 text-center">
          GitHubChat • AI-powered exploration
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
