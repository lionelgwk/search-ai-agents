from agents import Runner
import asyncio
from custom_agents.triage import triage_agent
from contexts.information import create_initial_context


async def main():
    print("Starting...")
    context = create_initial_context()
    result = await Runner.run(
        starting_agent=triage_agent,
        input="Can you tell me who is the current president of the United States in 2025?",
        context=context,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
