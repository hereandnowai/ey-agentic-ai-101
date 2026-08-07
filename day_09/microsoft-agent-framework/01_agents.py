# What is an AI Agent?
# Model Client + instructions + memory + tools to accomplish a task

from _maf import get_client, run, banner, POLICY

banner("File 1 - Microsoft Agent Framework - Agents - client + instructions")

async def main():
    client = get_client()
    agent = client.as_agent(
        name="support",
        instructions=f"You are Sheldon Retail Support. {POLICY} Answer in one sentence.")
    result = await agent.run("My headphones arrived 12 days ago and crackle. Can I return them?")

    print("The Agent said")
    print(f" {result.text}")
    print()
    print("What came back")
    print(f" type: {type(result).__name__}")
    print(f" messages : {len(result.messages)}")

run(main())