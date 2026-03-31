from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from zod import *
from langchain import createAgent, providerStrategy

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source to be used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    url: str = Field(description="The agent's answer to the query")
    sources:List = Field(default_factory=list,description="List of sources used to generate the answer.")

llm = ChatOpenAI(model="gpt-4.1-mini")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
# agent = create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-course!")
    content = "search for 3 job postings for an Senior Sdet Playwright with 11 years experience using langchain in India with Remote work mode on linkedin and list their details"
    # content = "what is the weather prediction for pune for the month of April 2026"
    result = agent.invoke({"messages":HumanMessage(content=content)})
    print(result)
    print(result['structured_response'].answer)

if __name__ == "__main__":
    main()