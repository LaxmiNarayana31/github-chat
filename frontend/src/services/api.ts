/**
 * Centralized API client service for GithubChat frontend.
 * Communicates with the FastAPI backend.
 */

export interface DocumentMetadata {
  file_path: string;
  type: string;
  is_code: boolean;
  is_implementation: boolean;
  title: string;
}

export interface Document {
  text: string;
  meta_data: DocumentMetadata;
}

export interface AgentStep {
  id: string;
  title: string;
  detail?: string;
  status: "running" | "completed" | "error";
}

export interface QueryResponse {
  rationale: string;
  answer: string;
  contexts: Document[];
  steps?: AgentStep[];
}

export interface StreamCallbacks {
  onStatus?: (status: { stage: string; message: string }) => void;
  onStep?: (step: AgentStep) => void;
  onToken?: (token: string) => void;
  onDone?: (data: {
    answer: string;
    rationale: string;
    contexts: Document[];
    steps?: AgentStep[];
  }) => void;
  onError?: (error: string) => void;
}

export interface InitResponse {
  status: string;
  message: string;
  repo_url: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  gemini_configured: boolean;
  groq_configured: boolean;
}

export interface ContextMessage {
  role: "user" | "assistant";
  content: string;
}

const API_BASE_URL: string =
  (import.meta as unknown as { env?: { VITE_API_URL?: string } })?.env
    ?.VITE_API_URL || "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        errorMessage =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail);
      } else if (data && data.error) {
        errorMessage = data.error;
      }
    } catch {
      // Body was not JSON
    }
    throw new Error(errorMessage);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async checkHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      return await handleResponse<HealthResponse>(res);
    } catch {
      throw new Error(
        `Backend unreachable at ${API_BASE_URL}. Ensure the backend server is running.`,
      );
    }
  },

  async initRepo(
    repoUrl: string,
    forceReindex: boolean = false,
    githubToken?: string,
    sessionId?: string,
  ): Promise<InitResponse> {
    const res = await fetch(`${API_BASE_URL}/init`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_url: repoUrl.trim(),
        force_reindex: forceReindex,
        github_token: githubToken || undefined,
        session_id: sessionId || undefined,
      }),
    });
    return handleResponse<InitResponse>(res);
  },

  async queryRepo(
    repoUrl: string,
    query: string,
    sessionId?: string,
  ): Promise<QueryResponse> {
    const res = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_url: repoUrl.trim(),
        query: query.trim(),
        session_id: sessionId || undefined,
      }),
    });
    return handleResponse<QueryResponse>(res);
  },

  async queryRepoStream(
    repoUrl: string,
    query: string,
    callbacks: StreamCallbacks,
    sessionId?: string,
  ): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_url: repoUrl.trim(),
        query: query.trim(),
        session_id: sessionId || undefined,
      }),
    });

    if (!res.ok) {
      let errorMessage = `Request failed with status ${res.status}`;
      try {
        const data = await res.json();
        if (data?.detail) {
          errorMessage =
            typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail);
        } else if (data?.error) {
          errorMessage = data.error;
        }
      } catch {
        // Not JSON
      }
      throw new Error(errorMessage);
    }

    if (!res.body) {
      throw new Error("No response body received from stream endpoint.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          if (!part.trim()) continue;
          const lines = part.split("\n");
          let eventType = "message";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.replace(/^event:\s*/, "").trim();
            } else if (line.startsWith("data:")) {
              dataStr = line.replace(/^data:\s*/, "");
            }
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              if (eventType === "step") {
                const stepObj = data.step || data;
                callbacks.onStep?.(stepObj);
              } else if (eventType === "status") {
                callbacks.onStatus?.(data);
              } else if (eventType === "token") {
                callbacks.onToken?.(data.token);
              } else if (eventType === "done") {
                callbacks.onDone?.(data);
              } else if (eventType === "error") {
                callbacks.onError?.(data.error);
              }
            } catch (e) {
              console.warn("Failed to parse SSE JSON payload:", dataStr, e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  async clearMemory(
    sessionId?: string,
  ): Promise<{ status: string; message: string }> {
    const res = await fetch(`${API_BASE_URL}/clear-memory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId || undefined,
      }),
    });
    return handleResponse<{ status: string; message: string }>(res);
  },

  async setContext(
    messages: ContextMessage[],
    sessionId?: string,
  ): Promise<{ status: string; turns: number }> {
    const url = sessionId
      ? `${API_BASE_URL}/set-context?session_id=${encodeURIComponent(sessionId)}`
      : `${API_BASE_URL}/set-context`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      }),
    });
    return handleResponse<{ status: string; turns: number }>(res);
  },
};

export default api;
