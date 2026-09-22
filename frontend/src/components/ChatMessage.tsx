import React from 'react';
import ReactMarkdown from 'react-markdown';
import {
  User,
  Bot,
  ChevronDown,
  ChevronRight,
  Brain,
  CheckCircle2,
  Loader2,
  AlertCircle,
  FileCode2,
} from 'lucide-react';
import { Document, AgentStep } from '../services/api';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  rationale?: string;
  contexts?: Document[];
  steps?: AgentStep[];
  isLoading?: boolean;
  statusMessage?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  rationale,
  contexts,
  steps = [],
  isLoading,
  statusMessage,
}) => {
  const [expandedContexts, setExpandedContexts] = React.useState<{
    [key: number]: boolean;
  }>({});
  const [showRationale, setShowRationale] = React.useState(false);
  const [showSteps, setShowSteps] = React.useState(false);

  const toggleContext = (index: number) => {
    setExpandedContexts((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const normalizeMarkdown = (text: string): string => {
    if (!text) return text;
    return text.replace(/\\n/g, '\n');
  };

  const isUser = role === 'user';
  const hasSteps = steps && steps.length > 0;

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
          isUser ? 'bg-gray-700' : 'bg-gray-800 border border-gray-700'
        }`}>
        {isUser ? (
          <User className="h-5 w-5 text-gray-300" />
        ) : (
          <Bot className="h-5 w-5 text-blue-400" />
        )}
      </div>

      {/* Message content */}
      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        {/* Live Step-by-Step Thinking Trace while streaming */}
        {!isUser && isLoading && (
          <div className="mb-3 p-3.5 bg-gray-900/90 border border-blue-500/30 rounded-xl shadow-lg shadow-blue-950/20 text-left">
            <div className="flex items-center justify-between mb-2.5 pb-2 border-b border-gray-800">
              <div className="flex items-center gap-2 text-xs font-semibold text-blue-400">
                <Brain className="h-4 w-4 animate-pulse text-blue-400" />
                <span>Agent Real-Time Execution</span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-blue-950/60 border border-blue-800/40 text-[11px] font-mono text-blue-300">
                <Loader2 className="h-3 w-3 animate-spin text-blue-400" />
                <span>{statusMessage || 'Thinking...'}</span>
              </div>
            </div>

            {hasSteps ? (
              <div className="space-y-2 mt-1">
                {steps.map((step, idx) => (
                  <div
                    key={step.id || idx}
                    className={`flex items-start gap-2.5 text-xs p-2 rounded-lg transition-all ${
                      step.status === 'running'
                        ? 'bg-blue-950/40 border border-blue-500/30'
                        : 'bg-gray-800/40 border border-gray-800'
                    }`}>
                    {step.status === 'completed' ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    ) : step.status === 'running' ? (
                      <Loader2 className="h-4 w-4 text-blue-400 animate-spin mt-0.5 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-rose-400 mt-0.5 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span
                          className={`font-semibold ${
                            step.status === 'completed'
                              ? 'text-gray-200'
                              : 'text-blue-300'
                          }`}>
                          {step.title}
                        </span>
                        <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 bg-gray-800 rounded text-gray-400">
                          {step.status}
                        </span>
                      </div>
                      {step.detail && (
                        <p className="text-[11px] text-gray-400 font-mono mt-1 leading-relaxed">
                          {step.detail}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center gap-2 py-1 text-xs text-gray-400">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-ping" />
                <span>Initializing Agentic RAG workflow...</span>
              </div>
            )}
          </div>
        )}

        {/* Message body */}
        <div
          className={`inline-block rounded-xl px-5 py-3.5 text-left w-full ${
            isUser
              ? 'bg-gray-700 text-white max-w-fit ml-auto'
              : 'bg-gray-800 text-gray-100 border border-gray-700'
          }`}>
          {isLoading && !content ? (
            <div className="flex items-center gap-2 py-1">
              <div className="flex gap-1.5">
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: '0ms' }}
                />
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: '150ms' }}
                />
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: '300ms' }}
                />
              </div>
              <span className="text-gray-300 text-sm font-medium">
                Synthesizing response...
              </span>
            </div>
          ) : (
            <div>
              {isLoading && statusMessage && (
                <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-700/60 text-xs text-blue-400 font-mono">
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-ping" />
                  <span>{statusMessage}</span>
                </div>
              )}
              <div
                className={`prose prose-sm max-w-none ${
                  isUser ? 'prose-invert' : 'prose-invert'
                }`}>
                <ReactMarkdown>{normalizeMarkdown(content)}</ReactMarkdown>
                {isLoading && (
                  <span className="inline-block w-1.5 h-4 ml-1 bg-blue-400 animate-pulse align-middle" />
                )}
              </div>
            </div>
          )}
        </div>

        {/* Completed Step-by-Step Thinking Trace Accordion (Post-Stream) */}
        {!isUser && !isLoading && hasSteps && (
          <div className="mt-2.5 text-left">
            <button
              onClick={() => setShowSteps(!showSteps)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-400 hover:text-blue-300 bg-blue-950/30 hover:bg-blue-950/50 border border-blue-900/40 rounded-lg transition-colors">
              <Brain className="h-3.5 w-3.5 text-blue-400" />
              <span>Thought Process ({steps.length} steps)</span>
              {showSteps ? (
                <ChevronDown className="h-3.5 w-3.5 ml-1" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 ml-1" />
              )}
            </button>
            {showSteps && (
              <div className="mt-2 p-3 bg-gray-900/80 rounded-xl border border-gray-800 space-y-2">
                {steps.map((step, idx) => (
                  <div
                    key={step.id || idx}
                    className="flex items-start gap-2 text-xs p-2 rounded bg-gray-800/40 border border-gray-800/80">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="font-semibold text-gray-200">
                        {step.title}
                      </div>
                      {step.detail && (
                        <p className="text-[11px] text-gray-400 font-mono mt-0.5">
                          {step.detail}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Rationale section */}
        {!isUser && !isLoading && rationale && (
          <div className="mt-2.5 text-left">
            <button
              onClick={() => setShowRationale(!showRationale)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-purple-400 hover:text-purple-300 bg-purple-950/30 hover:bg-purple-950/50 border border-purple-900/40 rounded-lg transition-colors">
              <Brain className="h-3.5 w-3.5 text-purple-400" />
              <span>View Rationale & Verification</span>
              {showRationale ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
            </button>
            {showRationale && (
              <div className="mt-2 p-3.5 bg-gray-800/50 rounded-lg border border-purple-900/30">
                <div className="prose prose-sm prose-invert max-w-none text-xs text-gray-300">
                  <ReactMarkdown>{normalizeMarkdown(rationale)}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Contexts section */}
        {!isUser && !isLoading && contexts && contexts.length > 0 && (
          <div className="mt-3 space-y-2 text-left">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 flex items-center gap-1.5">
              <FileCode2 className="h-3.5 w-3.5 text-blue-400" />
              <span>Referenced Files ({contexts.length})</span>
            </p>
            {contexts.map((context, index) => (
              <div
                key={index}
                className="bg-gray-900/60 rounded-lg border border-gray-800 overflow-hidden">
                <button
                  onClick={() => toggleContext(index)}
                  className="w-full flex items-center justify-between p-2.5 text-left hover:bg-gray-800/60 transition-colors">
                  <div className="flex items-center gap-2 truncate">
                    {expandedContexts[index] ? (
                      <ChevronDown className="h-4 w-4 text-gray-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-gray-400" />
                    )}
                    <span className="text-xs font-mono font-medium text-blue-300 truncate">
                      {context.meta_data.file_path || 'Code snippet'}
                    </span>
                  </div>
                  <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-gray-800 text-gray-400">
                    {context.meta_data.type || 'code'}
                  </span>
                </button>
                {expandedContexts[index] && (
                  <div className="px-4 pb-4 border-t border-gray-800 bg-gray-950/60">
                    <div className="prose prose-sm prose-invert max-w-none mt-3 text-xs">
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
