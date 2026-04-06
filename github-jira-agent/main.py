import asyncio
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config.logger import setup_logging, generate_trace_id, logger
from config.mcp_client_settings import validate_env
from core.graph import graph

# INITIALIZE LOGGING & ENV
setup_logging()

async def run_interactive_agent():
    """
    Main interactive agent loop with multi-stage flow:
    Suggest -> PR -> Merge/Update
    Exits automatically upon graph completion (END).
    """

    if not validate_env():
        logger.error("❌ Environment validation failed")
        return

    session_id = generate_trace_id()

    with logger.contextualize(trace_id=session_id):
        logger.info(f"🚀 Initializing Autonomous DevOps Agent [ID: {session_id}]")

        config = {"configurable": {"thread_id": f"session_{session_id}"}}

        system_prompt = SystemMessage(content=(
            "You are an expert Java Spring Boot Engineer.\n"
            "1. TICKET ANALYSIS: Fetch the assigned Jira issue.\n"
            "2. DYNAMIC DISCOVERY: Use 'search_code' to find file paths.\n"
            "3. IMPLEMENTATION: Propose changes with FULL file content.\n\n"
            "GITHUB RULES:\n"
            "- Repository: ankit-devwork/springboot-demo\n"
            "- Always use feature branches (never main).\n"
        ))

        initial_msg = (
            "Fetch Jira issue SCRUM-6, find the Controller file in 'springboot-demo', "
            "and propose the DELETE implementation."
        )

        # Initial input state
        current_state = {
            "messages": [system_prompt, HumanMessage(content=initial_msg)],
            "correlation_id": session_id
        }

        logger.info("✅ Initial state created. Starting graph...")

        while True:
            try:
                # 1. RUN GRAPH (Streams until an interrupt or END is reached)
                async for event in graph.astream(current_state, config, stream_mode="values"):
                    if not event or "messages" not in event:
                        continue
                    
                    last_msg = event["messages"][-1]

                    # Only print actual AI content (avoiding tool noise)
                    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        print(f"\n🤖 AGENT:\n{last_msg.content}")

                # ✅ CRITICAL: Check if the graph reached the END state
                # This prevents the loop from asking for "Action" after a successful MERGE
                state_snapshot = await graph.aget_state(config)
                if not state_snapshot.next:
                    logger.info("🏁 Graph reached END state. Workflow finalized.")
                    print("\n" + "═" * 60)
                    print("✅ WORKFLOW COMPLETE: Session ended gracefully.")
                    print("═" * 60 + "\n")
                    break 

                # If we are here, the graph is paused at an interrupt (suggestor or pr_generator)
                current_state = None 

                # 2. PROMPT USER
                print("\n" + "─" * 60)
                print("COMMANDS: [Feedback] | 'APPROVE' (to PR) | 'MERGE' (to finish) | 'ABORT'")
                user_input = input("👉 Action: ").strip()

                if user_input.upper() == "ABORT":
                    print("\n👋 Session Aborted. Goodbye!")
                    break

                # 3. HANDLE COMMANDS & RESUME
                logger.info(f"📝 User Input: {user_input}")
                
                # Update the checkpoint with the user's message
                await graph.aupdate_state(
                    config, 
                    {"messages": [HumanMessage(content=user_input)]}
                )
                
                # Loop continues, calling astream(None, ...) to resume from checkpoint
                continue 

            except KeyboardInterrupt:
                logger.warning("⚠️ Process interrupted by user.")
                break
            except Exception as e:
                logger.error(f"❌ Runtime Error: {e}", exc_info=True)
                break

async def main():
    try:
        await run_interactive_agent()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())