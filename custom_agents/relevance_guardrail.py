from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    GuardrailFunctionOutput,
    TResponseInputItem,
    input_guardrail,
)
from pydantic import BaseModel
from contexts.information import InformationContext


class RelevanceOutput(BaseModel):
    """Schema for relevance guardrail decisions."""

    reasoning: str
    is_relevant: bool


INSTRUCTIONS = """
    You are a guardrail agent that will be used to ensure that the agents are not breaking the rules.

    You will receive a question from a user and you will need to ensure that the agents are not breaking the rules.

    Determine if the user's message is highly unrelated to a web search or a knowledge seeking prompt.

    Important: You are ONLY evaluating the most recent user message, not any of the previous messages from the chat history.

    It is OK for the customer to send messages such as 'Hi' or 'OK' or any other messages that are at all conversational.

    but if the response is non-conversational, it must be somewhat related to web search or knowledge seeking.

    Reject any prompts that are not safe for work, pornographic, or otherwise inappropriate, even if they are related to web search or knowledge seeking.

    Return is_relevant=True if it is, else False, plus a brief reasoning.
"""

guardrail_agent = Agent[InformationContext](
    name="Relevance Guardrail Agent",
    model="gpt-4o",
    instructions=INSTRUCTIONS,
    output_type=RelevanceOutput,
)


@input_guardrail(name="Relevance Guardrail Agent")
async def relevance_guardrail(
    context: RunContextWrapper[InformationContext],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Guardrail to check if input is relevant to web search or knowledge seeking topics."""
    result = await Runner.run(guardrail_agent, input, context=context.context)
    final = result.final_output_as(RelevanceOutput)
    return GuardrailFunctionOutput(
        output_info=final, tripwire_triggered=not final.is_relevant
    )
