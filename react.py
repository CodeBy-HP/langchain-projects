import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    Docstring for triple

    :param num: a number to triple
    :type num: float
    :return: the triple of the input number
    :rtype: float
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = AzureChatOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),temperature=0).bind_tools(tools)



