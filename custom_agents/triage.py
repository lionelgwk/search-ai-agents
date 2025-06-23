from agents import Agent, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from contexts.information import InformationContext
from pydantic import BaseModel
from custom_agents.relevance_guardrail import relevance_guardrail
from custom_agents.jailbreak_guardrail import jailbreak_guardrail
from custom_agents.nsfw_guardrail import nsfw_guardrail
from custom_agents.brave_search import brave_search_agent
from custom_agents.llm import llm_agent


class HandoffInfo(BaseModel):
    """
    Reason on why the handoff is made and how it is decided.
    """

    handoff_reason: str


INSTRUCTIONS = f"""
    {RECOMMENDED_PROMPT_PREFIX}
    You are a helpful triaging agent that receives questions from users on anything and decides which appropriate agent to handoff to.

    You have the following agents available to you:
    - brave_search_agent: A search agent that can search the web for information.
    - llm_agent: A LLM agent that can answer questions based on its knowledge base.

    Your main distinction to make is to determine if the question is sensitive to recency, 
    meaning if the information needed is after the llm_agent's cut off date, you should handoff to the brave_search_agent.

    If the question is not sensitive to recency, you should handoff to the llm_agent.

    The llm_agent's knowledge base is as of 2024-10-01 in YYYY-MM-DD format.

    Important: You will ONLY be handing off to ONE agent; you will not be handing off to multiple agents. 
    
    Choose the most appropriate agent based on the question and its recency to ensure data freshness.

    You will receive a question from a user and you will need to decide which agent to handoff to.
    You will need to provide the following information:
    - The agent to handoff to
    - The reason for the handoff
"""


async def on_handoff(
    ctx: RunContextWrapper[InformationContext], input_data: HandoffInfo
) -> None:
    ctx.context.handoff_reason = input_data.handoff_reason


triage_agent = Agent[InformationContext](
    name="Triage Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o",
    handoffs=[
        handoff(
            agent=brave_search_agent, input_type=HandoffInfo, on_handoff=on_handoff
        ),
        handoff(agent=llm_agent, input_type=HandoffInfo, on_handoff=on_handoff),
    ],
    input_guardrails=[relevance_guardrail, jailbreak_guardrail, nsfw_guardrail],
)
