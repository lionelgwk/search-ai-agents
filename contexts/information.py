# NOTE: this information is not fed to the agent, but it can be used to pass information along to the next agent

from dataclasses import dataclass


@dataclass
class InformationContext:
    """
    Context for the search agents.
    """

    question: str | None = None
    answer: str | None = None
    previous_agent: str | None = None
    handoff_reason: str | None = None


def create_initial_context() -> InformationContext:
    """
    Create an initial context for the search agents.
    """
    return InformationContext(
        question=None, answer=None, previous_agent=None, handoff_reason=None
    )
