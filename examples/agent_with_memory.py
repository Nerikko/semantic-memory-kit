"""
Example: AI agent that uses semantic memory for context retrieval
"""
from semantic_memory import SemanticMemory
import anthropic

mem = SemanticMemory("./memory")
mem.index()

client = anthropic.Anthropic()

def answer_with_memory(user_question: str) -> str:
    # Get relevant context from memory
    context = mem.query_and_format(user_question, top_k=4)
    
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=f"""You are a helpful assistant with access to your memory.
        
Relevant memory context:
{context}

Use this context when relevant. If the context doesn't help, answer from your general knowledge.""",
        messages=[{"role": "user", "content": user_question}]
    )
    return response.content[0].text

# Test
print(answer_with_memory("What Stripe integration work have we done?"))
