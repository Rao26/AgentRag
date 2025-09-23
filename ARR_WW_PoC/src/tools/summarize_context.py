from typing import List
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import config

class SummarizeInput(BaseModel):
    documents: List[Document] = Field(description="List of documents to summarize")
    max_length: int = Field(default=500, description="Maximum summary length")

class SummarizeContextTool(BaseTool):
    name: str = "summarize_context"
    description: str = "Summarize a set of documents to extract key information"
    args_schema: type[BaseModel] = SummarizeInput
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=config.CEREBRAS_API_KEY,
            openai_api_base=config.CEREBRAS_API_BASE,
            max_tokens=1000
        )
    
    def _run(self, documents: List[Document], max_length: int = 500) -> str:
        """Summarize the provided documents"""
        try:
            # Combine document content
            context = "\n\n".join([doc.page_content for doc in documents])
            
            if len(context) > config.MAX_CONTEXT_LENGTH:
                context = context[:config.MAX_CONTEXT_LENGTH]
            
            system_prompt = """You are a helpful assistant that creates concise summaries. 
            Extract the key information from the provided text and create a well-structured summary.
            Focus on the most important facts, concepts, and insights."""
            
            human_message = f"""Please summarize the following text in about {max_length} words:
            
            {context}
            
            Summary:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    async def _arun(self, documents: List[Document], max_length: int = 500) -> str:
        """Async summarize the provided documents"""
        return self._run(documents, max_length)