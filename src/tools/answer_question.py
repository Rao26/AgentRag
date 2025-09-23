from typing import List
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import config

class AnswerInput(BaseModel):
    question: str = Field(description="The question to answer")
    context: str = Field(description="The context to use for answering")
    documents: List[Document] = Field(description="Source documents for citation")

class AnswerQuestionTool(BaseTool):
    name: str = "answer_question"
    description: str = "Generate a final answer based on the context and source documents"
    args_schema: type[BaseModel] = AnswerInput
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=config.CEREBRAS_API_KEY,
            openai_api_base=config.CEREBRAS_API_BASE,
            max_tokens=1500
        )
    
    def _run(self, question: str, context: str, documents: List[Document]) -> dict:
        """Generate answer with citations"""
        try:
            system_prompt = """You are a helpful assistant that provides accurate, grounded answers.
            Always base your answers strictly on the provided context.
            Cite your sources using the document metadata when referencing specific information.
            If the context doesn't contain enough information to answer the question, say so clearly."""
            
            # Prepare source information
            sources_info = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'N/A')
                sources_info.append(f"[{i+1}] Source: {source}, Page: {page}")
            
            sources_text = "\n".join(sources_info)
            
            human_message = f"""Question: {question}

            Context to use for answering:
            {context}

            Available sources for citation:
            {sources_text}

            Please provide a comprehensive answer based on the context above. 
            Include citations in your answer using the source numbers like [1], [2], etc.
            If you need to reference multiple sources for different parts of your answer, use multiple citations.

            Answer:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "answer": response.content,
                "sources": sources_info,
                "documents": documents
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "documents": []
            }
    
    async def _arun(self, question: str, context: str, documents: List[Document]) -> dict:
        """Async generate answer with citations"""
        return self._run(question, context, documents)