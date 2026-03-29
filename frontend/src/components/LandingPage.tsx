import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Github, ArrowRight, Code2, Loader2 } from "lucide-react";
import { toast } from "react-hot-toast";

const LandingPage: React.FC = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [isValidating, setIsValidating] = useState(false);
  const navigate = useNavigate();

  const handleStartChat = async () => {
    if (!repoUrl.trim()) return;
    
    setIsValidating(true);
    try {
      const response = await fetch("http://localhost:8000/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_url: repoUrl.trim() }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to initialize repository");
      }

      toast.success("Repository loaded! Starting chat...");
      navigate("/chat", { state: { repoUrl: repoUrl.trim() } });
    } catch (error) {
      console.error("Init error:", error);
      const errorMessage = error instanceof Error ? error.message : "Failed to initialize repository";
      toast.error(errorMessage);
    } finally {
      setIsValidating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && repoUrl.trim()) {
      handleStartChat();
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center p-4">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-to-b from-gray-800/20 to-transparent rounded-full blur-3xl" />
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-xl w-full space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="p-3 bg-gray-800 rounded-xl border border-gray-700">
              <Github className="h-10 w-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            GitHub<span className="text-gray-400">Chat</span>
          </h1>
          <p className="text-lg text-gray-400 flex items-center justify-center gap-2">
            <Code2 className="h-5 w-5 text-gray-500" />
            Chat with any GitHub repository using AI
          </p>
        </div>

        {/* Input Card */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="space-y-5">
            <div>
              <label htmlFor="repo-url" className="block text-sm font-medium text-gray-300 mb-3">
                Repository URL
              </label>
              <div className="relative">
                <input
                  id="repo-url"
                  type="text"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="w-full px-4 py-3.5 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-transparent transition-all text-base"
                  placeholder="https://github.com/username/repository"
                />
              </div>
            </div>

            <button
              onClick={handleStartChat}
              disabled={!repoUrl.trim() || isValidating}
              className="w-full flex items-center justify-center gap-3 px-6 py-3.5 bg-white text-gray-900 font-semibold rounded-lg hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-gray-950 disabled:opacity-50 disabled:cursor-not-allowed transition-all text-base"
            >
              {isValidating ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Initializing repository...</span>
                </>
              ) : (
                <>
                  <span>Start Chat</span>
                  <ArrowRight className="h-5 w-5" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Footer hint */}
        <p className="text-center text-gray-600 text-sm">
          Paste any public GitHub repository URL to start exploring with AI
        </p>
      </div>
    </div>
  );
};

export default LandingPage;
