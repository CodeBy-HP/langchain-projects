from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools=[TavilySearch()]
agent=create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-projects!")
    result=agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Bhopal on linkedin and list their details")})
    print(result)
    

if __name__ == "__main__":
    main()
