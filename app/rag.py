from uuid import uuid4
from dataclasses import dataclass, field

import adalflow as adal
from adalflow.core.types import Conversation, DialogTurn, UserQuery, AssistantResponse


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
