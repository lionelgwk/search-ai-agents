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


class NSFWOutput(BaseModel):
    """Schema for NSFW guardrail decisions."""

    reasoning: str
    is_nsfw: bool


INSTRUCTIONS = """
    You are a guardrail agent that will be used to ensure that the agents are not breaking the rules.

    You will receive a question from a user and you will need to ensure that the agents are not breaking the rules.

    Determine if the user's message is highly pornographic or otherwise NSFW.

    Important: You are ONLY evaluating the most recent user message, not any of the previous messages from the chat history.

    Reject any prompts that are not safe for work, pornographic, or otherwise inappropriate.

    Return is_nsfw=True if it is, else False, plus a brief reasoning.
"""

guardrail_agent = Agent[InformationContext](
    name="NSFW Guardrail Agent",
    model="gpt-4o",
    instructions=INSTRUCTIONS,
    output_type=NSFWOutput,
)


@input_guardrail(name="NSFW Guardrail Agent")
async def nsfw_guardrail(
    context: RunContextWrapper[InformationContext],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Guardrail to check if input is NSFW."""
    result = await Runner.run(guardrail_agent, input, context=context.context)
    final = result.final_output_as(NSFWOutput)
    return GuardrailFunctionOutput(output_info=final, tripwire_triggered=final.is_nsfw)
