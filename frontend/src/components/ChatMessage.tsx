import React from "react";
import ReactMarkdown from "react-markdown";
import { User, Bot, ChevronDown, ChevronRight } from "lucide-react";

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

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  rationale?: string;
  contexts?: Document[];
  isLoading?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  rationale,
  contexts,
  isLoading,
}) => {
  const [expandedContexts, setExpandedContexts] = React.useState<{
    [key: number]: boolean;
  }>({});
  const [showRationale, setShowRationale] = React.useState(false);

  const toggleContext = (index: number) => {
    setExpandedContexts((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const isUser = role === "user";

  return (
    <div className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
          isUser
            ? "bg-gray-700"
            : "bg-gray-800 border border-gray-700"
        }`}
      >
        {isUser ? (
          <User className="h-5 w-5 text-gray-300" />
        ) : (
          <Bot className="h-5 w-5 text-gray-400" />
        )}
      </div>

      {/* Message content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block rounded-xl px-5 py-3 ${
            isUser
              ? "bg-gray-700 text-white"
              : "bg-gray-800 text-gray-100 border border-gray-700"
          }`}
        >
          {isLoading ? (
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
              <span className="text-gray-400 text-sm">Thinking...</span>
            </div>
          ) : (
            <div className={`prose prose-sm max-w-none ${isUser ? "prose-invert" : "prose-invert"}`}>
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Rationale section (for assistant messages) */}
        {!isUser && rationale && (
          <div className="mt-3">
            <button
              onClick={() => setShowRationale(!showRationale)}
              className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-400 transition-colors"
            >
              {showRationale ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              <span>View Reasoning</span>
            </button>
            {showRationale && (
              <div className="mt-2 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <div className="prose prose-sm prose-invert max-w-none">
                  <ReactMarkdown>{rationale}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Contexts section (for assistant messages) */}
        {!isUser && contexts && contexts.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-sm text-gray-500">Referenced files:</p>
            {contexts.map((context, index) => (
              <div
                key={index}
                className="bg-gray-800/50 rounded-lg border border-gray-700 overflow-hidden"
              >
                <button
                  onClick={() => toggleContext(index)}
                  className="w-full flex items-center gap-2 p-3 text-left hover:bg-gray-700/50 transition-colors"
                >
                  {expandedContexts[index] ? (
                    <ChevronDown className="h-4 w-4 text-gray-500" />
                  ) : (
                    <ChevronRight className="h-4 w-4 text-gray-500" />
                  )}
                  <span className="text-sm font-medium text-gray-300 truncate">
                    {context.meta_data.file_path}
                  </span>
                </button>
                {expandedContexts[index] && (
                  <div className="px-4 pb-4 border-t border-gray-700">
                    <div className="prose prose-sm prose-invert max-w-none mt-3">
                      <ReactMarkdown>{context.text}</ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
