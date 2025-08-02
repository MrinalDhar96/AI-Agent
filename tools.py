# tools.py

from langchain_community.tools import WikipediaQueryRun, ddg_search
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import os

# ---------- Utility Function ----------
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Confirming path and saving file
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"✅ Data successfully saved to: {os.path.abspath(filename)}"
    except Exception as e:
        return f"❌ Error saving file: {str(e)}"

# ---------- Save Tool ----------
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a local text file with timestamp."
)

# ---------- Search Tool ----------
duckduckgo_search = ddg_search()
search_tool = Tool(
    name="search",
    func=duckduckgo_search.run,
    description="Use DuckDuckGo to search the web for up-to-date information."
)

# ---------- Wikipedia Tool ----------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = Tool(
    name="wikipedia_query",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description="Pull summarized content from Wikipedia using LangChain API wrapper."
)

# ---------- Tool Registry ----------
TOOLS = [search_tool, wiki_tool, save_tool]
