from agents import Agent, function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
import requests
from dotenv import load_dotenv
import os
from custom_agents.relevance_guardrail import relevance_guardrail
from custom_agents.jailbreak_guardrail import jailbreak_guardrail
from custom_agents.nsfw_guardrail import nsfw_guardrail
from contexts.information import InformationContext

load_dotenv()

INSTRUCTIONS = f"""
    {RECOMMENDED_PROMPT_PREFIX}
    You are a web search agent that will make use of the Brave Search API to search the web for information.

    A function is provided to you that takes in a query in the form of a string and returns a list of results.
    You will need to use this function to search the web for information.

    You will receive a question from a user and you will need to search the web for information.
    You will need to provide the following information:
    - The results of the search answering the question received from the triaging agent.

    There are three ways you can answer the question:
    - If there are no quality results, you can say that your sources are limited and you can't answer the question.
    - If there are quality results and the top 3 results do not contradict each other, you can answer the question with the top result.
    - If there are quality results and the top 3 results somewhat contradict each other, you can provide a nuanced argument that shows the viewpoint of the top 3 results.
    """

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search?q="

# we set the headers for the Brave Search API
headers = {
    "Accept": "application/json",
    "X-Subscription-Token": os.getenv("BRAVE_API_KEY"),
}


@function_tool
def search_web(query: str) -> list[str]:
    """
    Search the web for information using the Brave Search API.

    Args:
        query: The query to search the web for.

    Returns:
        A list of results from the search.
    """

    # ensure we receive a trimmed query
    query = query.strip()

    # we concatenate the query with +
    query = query.replace(" ", "+")

    url = BRAVE_SEARCH_URL + query

    # we make the request to the Brave Search API
    response = requests.get(url, headers=headers)
    results = response.json().get("web", []).get("results", [])

    # we return the results
    return results


brave_search_agent = Agent[InformationContext](
    name="Brave Search Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o",
    handoff_description="A helpful agent that can search the web for information if the LLM agent's knowledge base is not up to date or suitable for the question.",
    tools=[search_web],
    input_guardrails=[relevance_guardrail, jailbreak_guardrail, nsfw_guardrail],
)
