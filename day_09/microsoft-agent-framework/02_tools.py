# what is a tool?
# tool = python function

from typing import Annotated
from _maf import get_client, run, banner, POLICY, ORDERS

banner("File 2 - MAF - function the agent may call to get order info")

CALLS: list[str] = []

def lookup_order(order_id: Annotated[str, "an order reference like SC-90455"]) -> str:
    """Look up a Sheldon Retail order and return its delivery status."""
    CALLS.append(order_id)
    record = ORDERS.get(order_id)
    if record is None:
        return f"No order {order_id} exists."
    return f"{order_id}: delivered {record['days']} days ago, faulty={record['faulty']}"

async def main():
    agent = get_client().as_agent(
        name="support",
        tools=[lookup_order],
        instructions=f"You are Sheldon Retail Support. {POLICY} Always call lookup_order first.")
    result = await agent.run("Is order SC-90455 returnable?")

    print(f" tool was called with: {CALLS}")
    print(f" agent answered      : {result.text}")

run(main())