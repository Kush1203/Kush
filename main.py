from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

#inherit from base model to able to pass on the llm
class BlogResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt -4o-mini")
parser = PydanticOutputParser(pydantic_object=BlogResponse)
#learning checkpoint 1-response = llm.invoke("what is this")
#print(response)

#now final prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a blog writing assistant that will help generate a trending blog.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text involve structuring content for online readability and engagement. Key elements include a clear title, introduction, body with subheadings, conclusion, and call to action.\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
 #now test the agent,import from langchain.agent and import the agent executer
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
#verbose is the thought process
query = input("What can i help you blog? ")
raw_response = agent_executor.invoke({"query": query})
 

structured_response=parser.parse(raw_response.get("output")[0]["text"])
print(structured_response)
