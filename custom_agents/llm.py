from agents import Agent
from custom_agents.relevance_guardrail import relevance_guardrail
from custom_agents.jailbreak_guardrail import jailbreak_guardrail
from custom_agents.nsfw_guardrail import nsfw_guardrail
from contexts.information import InformationContext

INSTRUCTIONS = """
You are a helpful LLM agent that can answer questions based on its knowledge base.

You will receive a question from a user and you will need to answer the question based on your knowledge base.

Your knowledge base is as of 2024-10-01 in YYYY-MM-DD format.

You will need to provide the following information:
- The answer to the question
"""

llm_agent = Agent[InformationContext](
    name="LLM Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o",
    handoff_description="A helpful LLM agent that can answer questions based on its knowledge base, cut off date 2024-10-01.",
    input_guardrails=[relevance_guardrail, jailbreak_guardrail, nsfw_guardrail],
)
