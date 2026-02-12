import re
import json
from typing import Any
from uuid import uuid4
from dataclasses import dataclass, field

import numpy as np
import adalflow as adal
from adalflow.core.types import ModelType
from adalflow.core.types import Conversation, DialogTurn, UserQuery, AssistantResponse
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.utils import printc

from app.config import config
from app.data_pipeline import DatabaseManager
from app.system_prompt import SYSTEM_PROMPT, RAG_TEMPLATE

# Memory component
class Memory(adal.DataComponent):
    def __init__(self):
        super().__init__()
        self.current_conversation = Conversation()
    def call(self): return self.current_conversation.dialog_turns
    def add_dialog_turn(self, uq, ar):
        self.current_conversation.append_dialog_turn(DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=uq),
            assistant_response=AssistantResponse(response_str=ar)
        ))

@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc":"Rationale."})
    answer: str = field(default="", metadata={"desc":"Answer."})
    __output_fields__ = ["rationale","answer"]

# RAG component
class RAG(adal.Component):
    def __init__(self):
        super().__init__()
        self.memory = Memory()
        self.embedder = adal.Embedder(
            model_client=config["embedder"]["model_client"](),
            model_kwargs=config["embedder"]["model_kwargs"],
        )
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)
        
        # Create the model client
        model_client = config["generator"]["model_client"]()
        
        self._configure_message_parser(model_client)
        
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": data_parser.get_output_format_str(),
                "conversation_history": self.memory(),
                "system_prompt": SYSTEM_PROMPT,
                "contexts": None,
            },
            model_client=model_client,
            model_kwargs=config["generator"]["model_kwargs"],
            output_processors=data_parser,
        )
    
    # Configure the model client to parse template tags into chat messages
    def _configure_message_parser(self, model_client):
        """Configure the model client to parse template tags into chat messages."""
        
        original_convert = model_client.convert_inputs_to_api_kwargs
        
        def patched_convert(input=None, model_kwargs={}, model_type=ModelType.UNDEFINED):
            final_model_kwargs = model_kwargs.copy()
            if model_type == ModelType.LLM and input:
                messages = []
                
                # Extract system message from <SYS>...</SYS>
                sys_match = re.search(r'<SYS>(.*?)</SYS>', input, re.DOTALL)
                if sys_match:
                    system_content = sys_match.group(1).strip()
                    messages.append({"role": "system", "content": system_content})
                
                # Extract middle content (context, history) between </SYS> and <USER>
                middle_match = re.search(r'</SYS>(.*?)<USER>', input, re.DOTALL)
                if middle_match:
                    middle_content = middle_match.group(1).strip()
                    if middle_content and messages:
                        messages[0]["content"] += "\n\n" + middle_content
                    elif middle_content:
                        messages.append({"role": "system", "content": middle_content})
                
                # Extract user message from <USER>...</USER> - MUST be last
                user_match = re.search(r'<USER>(.*?)</USER>', input, re.DOTALL)
                if user_match:
                    user_content = user_match.group(1).strip()
                    messages.append({"role": "user", "content": user_content})
                else:
                    # Fallback: add the whole input as user message
                    messages.append({"role": "user", "content": input})
                
                final_model_kwargs["messages"] = messages
                return final_model_kwargs
            else:
                return original_convert(input, model_kwargs, model_type)
        
        model_client.convert_inputs_to_api_kwargs = patched_convert

    def prepare_retriever(self, repo_url_or_path):
        self.transformed_docs = self.db_manager.prepare_database(repo_url_or_path)
        self.retriever = FAISSRetriever(
            **config["retriever"],
            embedder=self.embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def call(self, query: str) -> Any:
        # Embed query and extract vectors
        embed_output = self.embedder(query)
        if not embed_output.data:
            return {"rationale":"", "answer":""}, []
        vectors = [emb.embedding for emb in embed_output.data]
        query_vec = np.asarray(vectors, dtype="float32")

        # Search (Semantic Search)
        retrieved = self.retriever(query_vec)
        retrieved[0].documents = [self.transformed_docs[i] for i in retrieved[0].doc_indices]

        # Generate
        prompt_kwargs = {
            "input_str": query,
            "contexts": retrieved[0].documents,
            "conversation_history": self.memory(),
        }
        
        response = self.generator(prompt_kwargs=prompt_kwargs)
        printc(f"Raw response: {response.raw_response}", color="yellow")
        printc(f"Parsed data: {response.data}", color="yellow")
        printc(f"Error: {response.error}", color="red")
        
        final = response.data
        
        # Fallback: if parsing failed (empty fields), try to extract from raw response
        if final and hasattr(final, 'rationale') and hasattr(final, 'answer'):
            if not final.rationale and not final.answer:
                # Try to parse JSON from raw response
                try:
                    # Extract JSON from markdown code block if present
                    raw = response.raw_response
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
                    if json_match:
                        raw = json_match.group(1)
                    parsed = json.loads(raw)
                    final = RAGAnswer(
                        rationale=parsed.get('rationale', ''),
                        answer=parsed.get('answer', '')
                    )
                except (json.JSONDecodeError, AttributeError) as e:
                    # Use raw response as answer if all else fails
                    final = RAGAnswer(rationale="", answer=response.raw_response)
        
        self.memory.add_dialog_turn(uq=query, ar=str(final))
        return final, retrieved
