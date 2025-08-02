# main.py

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

# 🛠 Import tools
from tools import search_tool, wiki_tool, save_tool

# 🔑 Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 📐 Define structured response format
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# 🧠 Initialize GPT-4 model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# 🎯 Create output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# 💬 Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant that generates insightful reports.\nFormat your output like this:\n{format_instructions}"),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

# 🔧 Register tools
tools = [search_tool, wiki_tool, save_tool]

# 🕹 Create agent with tool-calling ability
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# 🚀 Initialize executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 📝 Prompt user for research input
query = input("What can I help you research today? ")

# 🎯 Run agent executor with input
raw_response = agent_executor.invoke({"query": query})

# 🧪 Try parsing structured response
try:
    structured_response = parser.parse(raw_response.get("output"))
    print("✅ Parsed Structured Output:")
    print(structured_response)
except Exception as e:
    print("❌ Error parsing response:", e)
    print("📄 Raw Agent Output:")
    print(raw_response.get("output", raw_response))

# 💾 Save output to file (optional fallback)
if raw_response.get("output"):
    save_result = save_tool.run(raw_response["output"])
    print("📁", save_result)