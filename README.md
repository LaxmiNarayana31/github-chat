# GithubChat

Production-grade Codebase Intelligence and Agentic Retrieval-Augmented Generation (RAG) platform. Clone, semantically index, and interact with any GitHub repository or local codebase in real time with exact file citations, autonomous agent routing, hybrid vector retrieval, and persistent conversational memory.

---

## Core Capabilities

- **Autonomous Agentic RAG (LangGraph)**:
  - **Query Intent Router**: Dynamically routes requests to direct conversational chat, repository structural overviews, or deep code retrieval.
  - **Corrective RAG (CRAG) Document Grader**: Evaluates retrieved code snippets for semantic relevance before generating answers.
  - **Query Reformulation Loop**: Iteratively expands and rewrites queries when retrieved context is insufficient.
  - **Self-RAG Grounding Evaluator**: Validates generated answers against retrieved source snippets to prevent hallucinations.
- **Git Commit SHA Verification & Incremental Ingestion**:
  - Validates remote repository commit hashes via `git ls-remote` prior to ingestion.
  - If the commit hash matches the existing database record, cloning and vector indexing are skipped entirely, instantly serving queries.
  - When new commits are detected, the repository is automatically updated and incrementally re-vectorized.
- **Qdrant Cloud Cluster Hybrid Search (RRF)**:
  - Connects to your managed **Qdrant Cloud cluster** via `QDRANT_URL` and `QDRANT_API_KEY`.
  - Combines 768-dimensional dense vectors from Google Gemini (`gemini-embedding-2` / `gemini-embedding-001`) with FastEmbed BM25 sparse lexical vectors.
  - Fuses dense and sparse rankings using Reciprocal Rank Fusion (RRF) for high recall across exact symbol names and semantic concepts.
- **Upstash Redis & Redis Caching Layer**:
  - Supports Upstash Redis REST API or standard Redis for distributed locking, raw Git blob chunk deduplication, and embedding caching.
  - Streaming checkpoint tracking allows large repository ingestions to resume seamlessly without re-embedding uncached files.
  - Automatic graceful fallback to in-memory caching when Redis credentials are not configured.
- **AST-Aware Chunking & PageRank Repository Map**:
  - Recursively parses source files with syntax awareness and constructs a repository-wide call/import dependency graph.
  - Computes structural importance via NetworkX PageRank to guide context assembly.
- **Memori Labs Relational SQL Memory**:
  - Persists conversational entities, processes, sessions, conversations, and dialog turns in a managed PostgreSQL database (such as Aiven) via SQLAlchemy (with transient in-memory storage for test runs).
  - High-level semantic memory recall enriches prompt context with known facts across sessions.
- **Real-Time Token & Thought Streaming**:
  - Server-Sent Events deliver real-time progress steps, reasoning summaries, live token streams, and source citations.
- **Dual User Interfaces**:
  - **React 18 SPA**: Modern TypeScript + Tailwind CSS application (running on port 3000) with real-time progress indicators, collapsible reasoning cards, and syntax-highlighted citations.
  - **Streamlit Dashboard**: Standalone Python web dashboard with model connection indicators, history controls, and transcript exports.

---

## Architecture Overview

```drawio locked
<mxfile>
  <diagram id="qN1j3XWIRPS1F5JPTTXA" name="Page-1">
    <mxGraphModel dx="2906" dy="3988" grid="1" gridSize="10" guides="1" tooltips="0" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="xZI-iIxJQkYUwU4l4Iug-0" />
        <mxCell id="xZI-iIxJQkYUwU4l4Iug-1" parent="xZI-iIxJQkYUwU4l4Iug-0" />
        <UserObject label="" mermaidData="{&#xa;  &quot;data&quot;: &quot;graph TB\r\n    subgraph Client_Layer[\&quot;Client Layer (User Interfaces)\&quot;]\r\n        ReactUI[\&quot;React 18 SPA (Vite + Tailwind CSS + TypeScript)&lt;br/&gt;Port 3000\&quot;]\r\n        StreamlitUI[\&quot;Streamlit Dashboard (Python)&lt;br/&gt;Port 8501\&quot;]\r\n    end\r\n\r\n    subgraph API_Layer[\&quot;API Gateway Layer (FastAPI)\&quot;]\r\n        FastAPIApp[\&quot;FastAPI Application (ASGI Server)&lt;br/&gt;Port 8000\&quot;]\r\n        RAGService[\&quot;RAGService Singleton\&quot;]\r\n    end\r\n\r\n    subgraph Core_Services[\&quot;Service &amp; Orchestration Layer\&quot;]\r\n        RAGOrchestrator[\&quot;RAG Core Engine\&quot;]\r\n        AgenticRAG[\&quot;AgenticRAG (LangGraph StateGraph)\&quot;]\r\n        DataPipeline[\&quot;DataPipeline (Git Ingestion &amp; Indexing)\&quot;]\r\n    end\r\n\r\n    subgraph Storage_Layer[\&quot;Data &amp; Persistence Layer\&quot;]\r\n        QdrantStore[\&quot;Qdrant Cloud Cluster&lt;br/&gt;Managed Vector Database via QDRANT_URL\&quot;]\r\n        MemoriSQL[\&quot;PostgreSQL Database (Aiven / Cloud)&lt;br/&gt;Managed Relational Storage via DB_*\&quot;]\r\n        RedisStore[\&quot;Upstash Redis / Redis&lt;br/&gt;Distributed Locks &amp; Embedding Cache\&quot;]\r\n        LocalFileCache[\&quot;Local Disk Cache&lt;br/&gt;~/.adalflow/databases/*.pkl &amp; repos/\&quot;]\r\n    end\r\n\r\n    subgraph External_Providers[\&quot;External Cloud AI &amp; Local ML Providers\&quot;]\r\n        GeminiAPI[\&quot;Google Gemini API&lt;br/&gt;gemini-embedding-2 / gemini-embedding-001\&quot;]\r\n        GroqAPI[\&quot;Groq Cloud API&lt;br/&gt;openai/gpt-oss-120b / llama-3.3-70b-versatile\&quot;]\r\n        FastEmbedModel[\&quot;FastEmbed QdrantBM25Model&lt;br/&gt;Local In-Memory CPU\&quot;]\r\n    end\r\n\r\n    ReactUI --&gt;|\&quot;HTTP / SSE\&quot;| FastAPIApp\r\n    FastAPIApp --&gt; RAGService\r\n    RAGService --&gt;|\&quot;Calls in-process\&quot;| RAGOrchestrator\r\n\r\n    StreamlitUI --&gt;|\&quot;Direct Python Import &amp; Call\&quot;| RAGOrchestrator\r\n\r\n    RAGOrchestrator --&gt; AgenticRAG\r\n    RAGOrchestrator --&gt; DataPipeline\r\n\r\n    DataPipeline --&gt;|\&quot;Clone &amp; Parse\&quot;| LocalFileCache\r\n    DataPipeline --&gt;|\&quot;Cache Embeddings &amp; Blobs\&quot;| RedisStore\r\n    DataPipeline --&gt;|\&quot;Compute Dense Vectors\&quot;| GeminiAPI\r\n    DataPipeline --&gt;|\&quot;Compute Sparse Vectors\&quot;| FastEmbedModel\r\n    DataPipeline --&gt;|\&quot;Upsert Vectors &amp; Payloads\&quot;| QdrantStore\r\n\r\n    AgenticRAG --&gt;|\&quot;Hybrid Search (RRF)\&quot;| QdrantStore\r\n    AgenticRAG --&gt;|\&quot;Query Embeddings\&quot;| GeminiAPI\r\n    AgenticRAG --&gt;|\&quot;Reasoning, Routing &amp; Generation\&quot;| GroqAPI\r\n    AgenticRAG --&gt;|\&quot;Recall &amp; Commit Dialog Turns\&quot;| MemoriSQL&quot;,&#xa;  &quot;config&quot;: null&#xa;}" id="jSYgUKzs4pwwbk9g5yOO-0">
          <mxCell connectable="0" parent="xZI-iIxJQkYUwU4l4Iug-1" style="group;transparentBounds=1;editIcon=1;lockedGroup=0;groupPadding=10;" vertex="1">
            <mxGeometry as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Client Layer (User Interfaces)" mermaidId="n:Client_Layer" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Client Layer (User Interfaces)" id="xZI-iIxJQkYUwU4l4Iug-2">
          <mxCell parent="jSYgUKzs4pwwbk9g5yOO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="143" width="785" x="1413" y="10" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="React 18 SPA (Vite + Tailwind CSS + TypeScript)&#xa;Port 3000" mermaidId="n:ReactUI" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="React 18 SPA (Vite + Tailwind CSS + TypeScript)&#xa;Port 3000" id="xZI-iIxJQkYUwU4l4Iug-3">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-2" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="394" x="35" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Streamlit Dashboard (Python)&#xa;Port 8501" mermaidId="n:StreamlitUI" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Streamlit Dashboard (Python)&#xa;Port 8501" id="xZI-iIxJQkYUwU4l4Iug-4">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-2" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="271" x="479" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="API Gateway Layer (FastAPI)" mermaidId="n:API_Layer" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="API Gateway Layer (FastAPI)" id="xZI-iIxJQkYUwU4l4Iug-5">
          <mxCell parent="jSYgUKzs4pwwbk9g5yOO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="266" width="367" x="1461" y="202" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="FastAPI Application (ASGI Server)&#xa;Port 8000" mermaidId="n:FastAPIApp" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="FastAPI Application (ASGI Server)&#xa;Port 8000" id="xZI-iIxJQkYUwU4l4Iug-6">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-5" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="297" x="35" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="RAGService Singleton" mermaidId="n:RAGService" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="RAGService Singleton" id="xZI-iIxJQkYUwU4l4Iug-7">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-5" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="212" x="78" y="187" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:FastAPIApp-&gt;RAGService#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="xZI-iIxJQkYUwU4l4Iug-33">
          <mxCell edge="1" parent="xZI-iIxJQkYUwU4l4Iug-5" source="xZI-iIxJQkYUwU4l4Iug-6" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-7">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Service &amp; Orchestration Layer" mermaidId="n:Core_Services" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Service &amp; Orchestration Layer" id="xZI-iIxJQkYUwU4l4Iug-8">
          <mxCell parent="jSYgUKzs4pwwbk9g5yOO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="228" width="1160" x="1252" y="517" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="RAG Core Engine" mermaidId="n:RAGOrchestrator" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="RAG Core Engine" id="xZI-iIxJQkYUwU4l4Iug-9">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-8" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="180" x="494" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="AgenticRAG (LangGraph StateGraph)" mermaidId="n:AgenticRAG" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="AgenticRAG (LangGraph StateGraph)" id="xZI-iIxJQkYUwU4l4Iug-10">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-8" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="322" x="803" y="149" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="DataPipeline (Git Ingestion &amp; Indexing)" mermaidId="n:DataPipeline" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="DataPipeline (Git Ingestion &amp; Indexing)" id="xZI-iIxJQkYUwU4l4Iug-11">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-8" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="339" x="35" y="149" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:RAGOrchestrator-&gt;AgenticRAG#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.72;entryX=0.5;entryY=0;" mermaidBaseValue="" id="xZI-iIxJQkYUwU4l4Iug-34">
          <mxCell edge="1" parent="xZI-iIxJQkYUwU4l4Iug-8" source="xZI-iIxJQkYUwU4l4Iug-9" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.72;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-10">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="964" y="124" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:RAGOrchestrator-&gt;DataPipeline#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.72;entryX=0.5;entryY=0;" mermaidBaseValue="" id="xZI-iIxJQkYUwU4l4Iug-35">
          <mxCell edge="1" parent="xZI-iIxJQkYUwU4l4Iug-8" source="xZI-iIxJQkYUwU4l4Iug-9" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.72;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-11">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="204" y="124" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Data &amp; Persistence Layer" mermaidId="n:Storage_Layer" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Data &amp; Persistence Layer" id="xZI-iIxJQkYUwU4l4Iug-12">
          <mxCell parent="jSYgUKzs4pwwbk9g5yOO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="143" width="1585" x="1254" y="794" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Qdrant Cloud Cluster&#xa;Managed Vector Database via QDRANT_URL" mermaidId="n:QdrantStore" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Qdrant Cloud Cluster&#xa;Managed Vector Database via QDRANT_URL" id="xZI-iIxJQkYUwU4l4Iug-13">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-12" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="369" x="1181" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="PostgreSQL Database (Aiven / Cloud)&#xa;Managed Relational Storage via DB_*" mermaidId="n:MemoriSQL" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="PostgreSQL Database (Aiven / Cloud)&#xa;Managed Relational Storage via DB_*" id="xZI-iIxJQkYUwU4l4Iug-14">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-12" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="323" x="800" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Upstash Redis / Redis&#xa;Distributed Locks &amp; Embedding Cache" mermaidId="n:RedisStore" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Upstash Redis / Redis&#xa;Distributed Locks &amp; Embedding Cache" id="xZI-iIxJQkYUwU4l4Iug-15">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-12" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="331" x="419" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Local Disk Cache&#xa;~/.adalflow/databases/*.pkl &amp; repos/" mermaidId="n:LocalFileCache" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Local Disk Cache&#xa;~/.adalflow/databases/*.pkl &amp; repos/" id="xZI-iIxJQkYUwU4l4Iug-16">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-12" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="334" x="35" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="External Cloud AI &amp; Local ML Providers" mermaidId="n:External_Providers" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="External Cloud AI &amp; Local ML Providers" id="xZI-iIxJQkYUwU4l4Iug-17">
          <mxCell parent="jSYgUKzs4pwwbk9g5yOO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;verticalAlign=top;fillColor=light-dark(#ffffde,#1f2020);strokeColor=light-dark(#aaaa33,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="143" width="1224" x="10" y="794" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Google Gemini API&#xa;gemini-embedding-2 / gemini-embedding-001" mermaidId="n:GeminiAPI" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Google Gemini API&#xa;gemini-embedding-2 / gemini-embedding-001" id="xZI-iIxJQkYUwU4l4Iug-18">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-17" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="389" x="800" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Groq Cloud API&#xa;openai/gpt-oss-120b / llama-3.3-70b-versatile" mermaidId="n:GroqAPI" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Groq Cloud API&#xa;openai/gpt-oss-120b / llama-3.3-70b-versatile" id="xZI-iIxJQkYUwU4l4Iug-19">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-17" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="393" x="357" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="FastEmbed QdrantBM25Model&#xa;Local In-Memory CPU" mermaidId="n:FastEmbedModel" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="FastEmbed QdrantBM25Model&#xa;Local In-Memory CPU" id="xZI-iIxJQkYUwU4l4Iug-20">
          <mxCell parent="xZI-iIxJQkYUwU4l4Iug-17" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="73" width="272" x="35" y="45" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="HTTP / SSE" mermaidId="e:ReactUI-&gt;FastAPIApp#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="HTTP / SSE" id="xZI-iIxJQkYUwU4l4Iug-21">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-3" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-6">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Calls in-process" mermaidId="e:RAGService-&gt;RAGOrchestrator#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0;entryY=0.06;" mermaidBaseValue="Calls in-process" id="xZI-iIxJQkYUwU4l4Iug-22">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-7" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0;entryY=0.06;" target="xZI-iIxJQkYUwU4l4Iug-9">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="1645" y="468" />
                <mxPoint x="1645" y="503" />
                <mxPoint x="1645" y="537" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Direct Python Import &amp; Call" mermaidId="e:StreamlitUI-&gt;RAGOrchestrator#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=1;entryY=0.06;" mermaidBaseValue="Direct Python Import &amp; Call" id="xZI-iIxJQkYUwU4l4Iug-23">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-4" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=1;entryY=0.06;" target="xZI-iIxJQkYUwU4l4Iug-9">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="2027" y="153" />
                <mxPoint x="2027" y="188" />
                <mxPoint x="2027" y="222" />
                <mxPoint x="2027" y="284" />
                <mxPoint x="2027" y="355" />
                <mxPoint x="2027" y="416" />
                <mxPoint x="2027" y="468" />
                <mxPoint x="2027" y="503" />
                <mxPoint x="2027" y="537" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Clone &amp; Parse" mermaidId="e:DataPipeline-&gt;LocalFileCache#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="Clone &amp; Parse" id="xZI-iIxJQkYUwU4l4Iug-24">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-16">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Cache Embeddings &amp; Blobs" mermaidId="e:DataPipeline-&gt;RedisStore#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.93;entryX=0.5;entryY=0;" mermaidBaseValue="Cache Embeddings &amp; Blobs" id="xZI-iIxJQkYUwU4l4Iug-25">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.93;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-15">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="1839" y="745" />
                <mxPoint x="1839" y="780" />
                <mxPoint x="1839" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Compute Dense Vectors" mermaidId="e:DataPipeline-&gt;GeminiAPI#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.8;entryX=0.33;entryY=0;" mermaidBaseValue="Compute Dense Vectors" id="xZI-iIxJQkYUwU4l4Iug-26">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.8;entryX=0.33;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-18">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="895" y="745" />
                <mxPoint x="895" y="780" />
                <mxPoint x="895" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Compute Sparse Vectors" mermaidId="e:DataPipeline-&gt;FastEmbedModel#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.63;entryX=0.5;entryY=0;" mermaidBaseValue="Compute Sparse Vectors" id="xZI-iIxJQkYUwU4l4Iug-27">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.63;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-20">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="181" y="745" />
                <mxPoint x="181" y="780" />
                <mxPoint x="181" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Upsert Vectors &amp; Payloads" mermaidId="e:DataPipeline-&gt;QdrantStore#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.65;entryX=0.34;entryY=0;" mermaidBaseValue="Upsert Vectors &amp; Payloads" id="xZI-iIxJQkYUwU4l4Iug-28">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.65;entryX=0.34;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-13">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="2523" y="745" />
                <mxPoint x="2523" y="780" />
                <mxPoint x="2523" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Hybrid Search (RRF)" mermaidId="e:AgenticRAG-&gt;QdrantStore#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.81;entryX=0.64;entryY=0;" mermaidBaseValue="Hybrid Search (RRF)" id="xZI-iIxJQkYUwU4l4Iug-29">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.81;entryX=0.64;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-13">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="2708" y="745" />
                <mxPoint x="2708" y="780" />
                <mxPoint x="2708" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Query Embeddings" mermaidId="e:AgenticRAG-&gt;GeminiAPI#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.65;entryX=0.65;entryY=0;" mermaidBaseValue="Query Embeddings" id="xZI-iIxJQkYUwU4l4Iug-30">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.65;entryX=0.65;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-18">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="1099" y="745" />
                <mxPoint x="1099" y="780" />
                <mxPoint x="1099" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Reasoning, Routing &amp; Generation" mermaidId="e:AgenticRAG-&gt;GroqAPI#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.59;entryX=0.5;entryY=0;" mermaidBaseValue="Reasoning, Routing &amp; Generation" id="xZI-iIxJQkYUwU4l4Iug-31">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.59;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-19">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="564" y="745" />
                <mxPoint x="564" y="780" />
                <mxPoint x="564" y="814" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Recall &amp; Commit Dialog Turns" mermaidId="e:AgenticRAG-&gt;MemoriSQL#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="Recall &amp; Commit Dialog Turns" id="xZI-iIxJQkYUwU4l4Iug-32">
          <mxCell edge="1" parent="jSYgUKzs4pwwbk9g5yOO-0" source="xZI-iIxJQkYUwU4l4Iug-10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="xZI-iIxJQkYUwU4l4Iug-14">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

For complete technical specifications, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Agentic RAG Pipeline

```drawio locked
<mxfile>
  <diagram id="WZWjQqguIFp1qsPSXmQ1" name="Page-1">
    <mxGraphModel dx="1544" dy="2119" grid="1" gridSize="10" guides="1" tooltips="0" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <UserObject label="" mermaidData="{&#xa;  &quot;data&quot;: &quot;flowchart TD\r\n    StartNode([Start: User Query]) --&gt; RouteNode[\&quot;Query Intent Router (LLM Evaluator)\&quot;]\r\n\r\n    RouteNode --&gt;|\&quot;direct_chat\&quot;| DirectChatNode[\&quot;Direct Conversational Chat (LLM)\&quot;]\r\n    RouteNode --&gt;|\&quot;repo_overview\&quot;| OverviewNode[\&quot;Repository Overview (File Tree &amp; Summary)\&quot;]\r\n    RouteNode --&gt;|\&quot;code_search\&quot;| RetrieveNode[\&quot;Retrieve Node (Gemini Dense + FastEmbed BM25 + Qdrant Cloud RRF)\&quot;]\r\n\r\n    RetrieveNode --&gt; GradeNode[\&quot;CRAG Document Grader (LLM)\&quot;]\r\n\r\n    GradeNode --&gt;|\&quot;Documents Relevant\&quot;| GenerateNode[\&quot;Generate Grounded Answer &amp; Rationale (Groq LLM)\&quot;]\r\n    GradeNode --&gt;|\&quot;Not Relevant &amp; Retries Left\&quot;| RewriteNode[\&quot;Query Rewriter (Technical Expansion)\&quot;]\r\n\r\n    RewriteNode --&gt;|\&quot;Rewritten Query\&quot;| RetrieveNode\r\n\r\n    GenerateNode --&gt; HallucinationNode[\&quot;Self-RAG Grounding Evaluator (LLM)\&quot;]\r\n\r\n    HallucinationNode --&gt;|\&quot;Hallucinated &amp; Retries Left\&quot;| GenerateNode\r\n    HallucinationNode --&gt;|\&quot;Grounded / Pass\&quot;| SaveMemoryNode[\&quot;Save Memory Node (Memori Labs SQL Storage)\&quot;]\r\n\r\n    DirectChatNode --&gt; SaveMemoryNode\r\n    OverviewNode --&gt; SaveMemoryNode\r\n    SaveMemoryNode --&gt; EndNode([End: Yield Answer, Rationale, Citations])&quot;,&#xa;  &quot;config&quot;: null&#xa;}" id="27">
          <mxCell connectable="0" parent="1" style="group;transparentBounds=1;editIcon=1;lockedGroup=0;groupPadding=10;" vertex="1">
            <mxGeometry as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Start: User Query" mermaidId="n:StartNode" mermaidBaseStyle="html=1;rounded=1;whiteSpace=wrap;arcSize=50;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Start: User Query" id="2">
          <mxCell parent="27" style="html=1;rounded=1;whiteSpace=wrap;arcSize=50;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="39" width="149" x="855" y="10" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Query Intent Router (LLM Evaluator)" mermaidId="n:RouteNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Query Intent Router (LLM Evaluator)" id="3">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="319" x="770" y="99" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Direct Conversational Chat (LLM)" mermaidId="n:DirectChatNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Direct Conversational Chat (LLM)" id="4">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="296" x="781" y="572" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Repository Overview (File Tree &amp; Summary)" mermaidId="n:OverviewNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Repository Overview (File Tree &amp; Summary)" id="5">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="372" x="1127" y="572" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Retrieve Node (Gemini Dense + FastEmbed BM25 + Qdrant Cloud RRF)" mermaidId="n:RetrieveNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Retrieve Node (Gemini Dense + FastEmbed BM25 + Qdrant Cloud RRF)" id="6">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="558" x="160" y="222" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="CRAG Document Grader (LLM)" mermaidId="n:GradeNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="CRAG Document Grader (LLM)" id="7">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="274" x="183" y="326" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Generate Grounded Answer &amp; Rationale (Groq LLM)" mermaidId="n:GenerateNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Generate Grounded Answer &amp; Rationale (Groq LLM)" id="8">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="429" x="10" y="449" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Query Rewriter (Technical Expansion)" mermaidId="n:RewriteNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Query Rewriter (Technical Expansion)" id="9">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="328" x="489" y="449" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Self-RAG Grounding Evaluator (LLM)" mermaidId="n:HallucinationNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Self-RAG Grounding Evaluator (LLM)" id="10">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="317" x="66" y="572" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Save Memory Node (Memori Labs SQL Storage)" mermaidId="n:SaveMemoryNode" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Save Memory Node (Memori Labs SQL Storage)" id="11">
          <mxCell parent="27" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="390" x="734" y="695" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="End: Yield Answer, Rationale, Citations" mermaidId="n:EndNode" mermaidBaseStyle="html=1;rounded=1;whiteSpace=wrap;arcSize=50;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="End: Yield Answer, Rationale, Citations" id="12">
          <mxCell parent="27" style="html=1;rounded=1;whiteSpace=wrap;arcSize=50;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="39" width="302" x="778" y="799" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:StartNode-&gt;RouteNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="13">
          <mxCell edge="1" parent="27" source="2" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="3">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="direct_chat" mermaidId="e:RouteNode-&gt;DirectChatNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="direct_chat" id="14">
          <mxCell edge="1" parent="27" source="3" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="4">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="repo_overview" mermaidId="e:RouteNode-&gt;OverviewNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.98;entryX=0.5;entryY=0;" mermaidBaseValue="repo_overview" id="15">
          <mxCell edge="1" parent="27" source="3" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.98;entryX=0.5;entryY=0;" target="5">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="1313" y="188" />
                <mxPoint x="1313" y="249" />
                <mxPoint x="1313" y="301" />
                <mxPoint x="1313" y="353" />
                <mxPoint x="1313" y="415" />
                <mxPoint x="1313" y="476" />
                <mxPoint x="1313" y="538" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="code_search" mermaidId="e:RouteNode-&gt;RetrieveNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.87;entryX=0.5;entryY=0;" mermaidBaseValue="code_search" id="16">
          <mxCell edge="1" parent="27" source="3" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.87;entryX=0.5;entryY=0;" target="6">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="439" y="188" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:RetrieveNode-&gt;GradeNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.39;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="17">
          <mxCell edge="1" parent="27" source="6" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.39;exitY=1;entryX=0.5;entryY=0;" target="7">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="320" y="301" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Documents Relevant" mermaidId="e:GradeNode-&gt;GenerateNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.35;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="Documents Relevant" id="18">
          <mxCell edge="1" parent="27" source="7" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.35;exitY=1;entryX=0.5;entryY=0;" target="8">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="225" y="415" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Not Relevant &amp; Retries Left" mermaidId="e:GradeNode-&gt;RewriteNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.84;exitY=1;entryX=0.34;entryY=0;" mermaidBaseValue="Not Relevant &amp; Retries Left" id="19">
          <mxCell edge="1" parent="27" source="7" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.84;exitY=1;entryX=0.34;entryY=0;" target="9">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="535" y="415" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Rewritten Query" mermaidId="e:RewriteNode-&gt;RetrieveNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.58;exitY=0;entryX=0.75;entryY=1;" mermaidBaseValue="Rewritten Query" id="20">
          <mxCell edge="1" parent="27" source="9" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.58;exitY=0;entryX=0.75;entryY=1;" target="6">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="712" y="415" />
                <mxPoint x="712" y="353" />
                <mxPoint x="712" y="301" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:GenerateNode-&gt;HallucinationNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.513986013986014;exitY=1;entryX=0.5189274447949527;entryY=0;curved=1;exitX=0.51;exitY=1;entryX=0.52;entryY=0;" mermaidBaseValue="" id="21">
          <mxCell edge="1" parent="27" source="8" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.51;exitY=1;entryX=0.52;entryY=0;" target="10">
            <mxGeometry relative="1" as="geometry">
              <mxPoint as="offset" />
              <Array as="points">
                <mxPoint x="237" y="538" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Hallucinated &amp; Retries Left" mermaidId="e:HallucinationNode-&gt;GenerateNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.4589905362776025;exitY=0;entryX=0.4696969696969697;entryY=1;curved=1;exitX=0.46;exitY=0;entryX=0.47;entryY=1;" mermaidBaseValue="Hallucinated &amp; Retries Left" id="22">
          <mxCell edge="1" parent="27" source="10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.46;exitY=0;entryX=0.47;entryY=1;" target="8">
            <mxGeometry relative="1" as="geometry">
              <mxPoint as="offset" />
              <Array as="points">
                <mxPoint x="115" y="538" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="Grounded / Pass" mermaidId="e:HallucinationNode-&gt;SaveMemoryNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0;entryY=0.19;" mermaidBaseValue="Grounded / Pass" id="23">
          <mxCell edge="1" parent="27" source="10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0;entryY=0.19;" target="11">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="225" y="661" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:DirectChatNode-&gt;SaveMemoryNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="24">
          <mxCell edge="1" parent="27" source="4" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="11">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:OverviewNode-&gt;SaveMemoryNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.93;entryY=0;" mermaidBaseValue="" id="25">
          <mxCell edge="1" parent="27" source="5" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.93;entryY=0;" target="11">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="1313" y="661" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:SaveMemoryNode-&gt;EndNode#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="26">
          <mxCell edge="1" parent="27" source="11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="12">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

## Prerequisites

- **Python**: 3.12 or higher
- **Node.js**: 18 or higher (for the React frontend)
- **Git**: Installed and available in PATH
- **Cloud Service Credentials**:
  - `GEMINI_API_KEY`: Google AI Studio API key (for dense embeddings)
  - `GROQ_API_KEY`: Groq Cloud API key (for fast LLM inference)
  - `QDRANT_URL`: Managed Qdrant Cloud cluster endpoint
  - `QDRANT_API_KEY`: Managed Qdrant Cloud API key
  - Managed PostgreSQL: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `SSL_MODE` (or `DATABASE_URL`)
  - `REDIS_URL` or `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`: (Optional) Upstash Redis / standard Redis for distributed locks, deduplication, and embedding caching
  - `GITHUB_TOKEN`: (Optional) GitHub Personal Access Token for private repositories or elevated clone limits

---

## Quickstart & Setup Guide

### Clone the Repository

```bash
git clone https://github.com/LaxmiNarayana31/github-chat.git
cd github-chat
```

---

### Configure Environment Variables

Create your `.env` configuration file from the template:

```bash
# On Linux / macOS
cp backend/.env.example backend/.env

# On Windows PowerShell
Copy-Item backend\.env.example backend\.env
```

Open `backend/.env` and insert your credentials:

```env
# Google AI Studio
GEMINI_API_KEY=your_gemini_api_key_here

# Groq Cloud
GROQ_API_KEY=your_groq_api_key_here

# Qdrant Cloud
QDRANT_URL=https://your-cluster-id.eu-west-2-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# Managed PostgreSQL (e.g. Aiven)
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host.aivencloud.com
DB_PORT=5432
DB_NAME=defaultdb
SSL_MODE=require

# Optional: Upstash Redis REST or standard Redis (falls back to local memory if omitted)
# UPSTASH_REDIS_REST_URL=https://your-database.upstash.io
# UPSTASH_REDIS_REST_TOKEN=your_upstash_rest_token
# REDIS_URL=redis://localhost:6379/0

# Optional: GitHub Personal Access Token for private repositories
# GITHUB_TOKEN=your_github_token_here
```

| Key | Where to Get | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) | Dense semantic vector embeddings |
| `GROQ_API_KEY` | [Groq Cloud Console](https://console.groq.com/keys) | Agent routing, CRAG grading, & generation |
| `QDRANT_URL` & `QDRANT_API_KEY` | [Qdrant Cloud Console](https://cloud.qdrant.io) | Managed hybrid vector retrieval cluster |
| `DB_*` / `DATABASE_URL` | [Aiven Console](https://console.aiven.io) | Managed PostgreSQL relational memory storage |
| `REDIS_URL` / `UPSTASH_*` | [Upstash Console](https://console.upstash.com) | (Optional) Distributed locks & embedding cache |
| `GITHUB_TOKEN` | [GitHub Settings](https://github.com/settings/tokens) | (Optional) Access to private repos & higher rate limits |

---

### Backend Setup & Startup (FastAPI)

Navigate to the `backend` directory:

```bash
cd backend
```

Choose your preferred package manager to install dependencies:

#### Using uv (Fastest)

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Using standard pip & virtualenv

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend server is now running at `http://localhost:8000`.
- Health Check: `http://localhost:8000/health`
- Interactive Swagger Docs: `http://localhost:8000/docs`

---

### Client Setup & Startup

You can use either the modern React SPA or the interactive Streamlit dashboard:

#### Option A: React 18 SPA (Recommended)

In a new terminal, navigate to the `frontend` folder:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. Enter any public GitHub repository URL (such as `https://github.com/fastapi/fastapi`), click **Index**, and begin chatting with full citations and streaming thoughts.

#### Option B: Streamlit Web Dashboard

In a new terminal, launch the Streamlit interface from the repository root:

```bash
# With uv
uv run --project backend streamlit run streamlit_app/main.py

# Or with activated virtual environment
streamlit run streamlit_app/main.py
```

Open `http://localhost:8501` in your browser.

---

### Verification & Testing

Verify that all services, pipelines, agents, and security controls are functioning properly by running the automated test suite:

```bash
# Run all unit, integration, and security tests
uv run --project backend pytest backend/tests
```

All 76 tests validate the document ingestion pipeline, Gemini embedder batching, Qdrant hybrid retrieval, Memori SQL turn saving, API routes, LangGraph agent execution, SSRF guards, prompt injection isolation, and citation fidelity verification.

---

## Enterprise Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): Detailed architectural components, state graphs, and data pipelines.
- [SECURITY.md](SECURITY.md): Threat matrix, SSRF defenses, prompt injection delimiters, and credential policies.
- [EVALUATION.md](EVALUATION.md): Research quality frameworks, citation fidelity metrics, and latency benchmarks.
