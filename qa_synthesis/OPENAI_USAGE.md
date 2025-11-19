# Using OpenAI API for ASearcher Data Generation

This guide shows how to use OpenAI's GPT models instead of local Qwen models for generating QA data in ASearcher.

## Quick Start

### 1. Set Your API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or pass it directly in code:

```python
from openai_client import OpenAIAPIClient

client = OpenAIAPIClient(model="gpt-4-turbo-preview", api_key="your-key")
```

### 2. Replace SGLangAPIClient

In your QA generation script, simply replace:

```python
# Old (local Qwen models)
from llm_utils import SGLangAPIClient

reasoning_client = SGLangAPIClient(model_name="QwQ-32B-Preview")
instruct_client = SGLangAPIClient(model_name="Qwen2.5-72B-Instruct")
```

With:

```python
# New (OpenAI API)
from openai_client import OpenAIAPIClient as SGLangAPIClient

reasoning_client = SGLangAPIClient(model="gpt-4-turbo-preview")
instruct_client = SGLangAPIClient(model="gpt-3.5-turbo")
```

**That's it!** The interface is 100% compatible - no other code changes needed.

## Model Recommendations

### Model Mapping

| Original (Qwen)          | Recommended (OpenAI)     | Purpose           | Cost (per 1M tokens) |
|-------------------------|--------------------------|-------------------|---------------------|
| QwQ-32B-Preview         | gpt-4-turbo-preview      | Reasoning tasks   | $10 input, $30 output |
| Qwen2.5-72B-Instruct    | gpt-3.5-turbo            | Instruction tasks | $0.50 input, $1.50 output |

### Quality vs Cost Trade-offs

```python
# High quality, expensive
reasoning_client = OpenAIAPIClient(model="gpt-4-turbo-preview")
instruct_client = OpenAIAPIClient(model="gpt-4")

# Good quality, 10× cheaper
reasoning_client = OpenAIAPIClient(model="gpt-4-turbo-preview")
instruct_client = OpenAIAPIClient(model="gpt-3.5-turbo")  # Recommended

# Fast and cheap (not recommended for reasoning)
reasoning_client = OpenAIAPIClient(model="gpt-3.5-turbo")
instruct_client = OpenAIAPIClient(model="gpt-3.5-turbo")
```

## Cost Estimates

### Per 1000 Questions Generated

Based on typical usage patterns (2-3 LLM calls per question, average 2K tokens per call):

| Configuration                          | Estimated Cost |
|---------------------------------------|---------------|
| gpt-4-turbo + gpt-4                   | $80-120       |
| gpt-4-turbo + gpt-3.5-turbo (recommended) | $50-80    |
| gpt-3.5-turbo + gpt-3.5-turbo         | $8-15         |

### Cost Comparison: Cloud vs Local

**OpenAI API (gpt-4-turbo + gpt-3.5-turbo)**:
- Cost: ~$60 per 1000 questions
- Time: ~2-4 hours (parallel processing)
- Infrastructure: None needed
- Total: $60, 2-4 hours

**Local Hosting (QwQ-32B + Qwen2.5-72B)**:
- Cost: GPU rental ~$2-5/hour × 8-12 hours = $16-60
- Time: 8-12 hours (slower inference)
- Infrastructure: SGLang server setup, 80GB+ VRAM needed
- Total: $16-60, 8-12 hours + setup time

**Verdict**: OpenAI API is comparable in cost but much simpler and faster for small-to-medium batches (< 10K questions).

## Advanced Configuration

### Custom API Endpoint

Use with Azure OpenAI or compatible endpoints:

```python
client = OpenAIAPIClient(
    model="gpt-4",
    base_url="https://your-azure-endpoint.openai.azure.com/",
    api_key="your-azure-key",
    organization="your-org-id"  # Optional
)
```

### Adjust Generation Parameters

All SGLang sampling parameters are supported:

```python
result = await client.async_generate(
    prompt="Your prompt here",
    sampling_kwargs={
        "temperature": 0.8,      # Randomness (0-2)
        "top_p": 0.95,           # Nucleus sampling (0-1)
        "max_new_tokens": 4096,  # Max output length
        "n": 3,                  # Generate 3 completions
        "stop": ["</answer>"]    # Stop sequences
    }
)
```

### Monitor Usage and Costs

The client tracks usage automatically:

```python
# After generation
stats = client.get_stats()
print(f"Total tokens used: {stats['total_tokens']}")
print(f"Total requests: {stats['total_requests']}")

# Estimate cost
cost = client.estimate_cost()
print(f"Estimated cost: ${cost:.2f}")
```

## Complete Example

```python
import asyncio
from openai_client import OpenAIAPIClient
from qa_synthesis_agent import ConstructQAAgent

async def main():
    # Initialize clients
    reasoning_client = OpenAIAPIClient(
        model="gpt-4-turbo-preview",
        timeout=600.0  # 10 min timeout for long reasoning
    )
    instruct_client = OpenAIAPIClient(
        model="gpt-3.5-turbo",
        timeout=300.0  # 5 min timeout
    )

    async with reasoning_client, instruct_client:
        # Load your data
        all_links = load_links("data/links.json")
        pages = load_pages("data/pages.json")
        search_client = SearchClient()

        # Create agent
        agent = ConstructQAAgent(all_links, pages, search_client)

        # Generate QA pairs
        questions = await agent.generate_questions(
            reasoning_client=reasoning_client,
            instruct_client=instruct_client,
            num_questions=100
        )

        # Save results
        save_questions(questions, "output/qa_pairs.json")

        # Print stats
        print("\n=== Generation Stats ===")
        print(f"Reasoning: {reasoning_client.get_stats()}")
        print(f"Instruct: {instruct_client.get_stats()}")
        print(f"\nEstimated costs:")
        print(f"  Reasoning: ${reasoning_client.estimate_cost():.2f}")
        print(f"  Instruct: ${instruct_client.estimate_cost():.2f}")
        print(f"  Total: ${reasoning_client.estimate_cost() + instruct_client.estimate_cost():.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

Test the client with a simple example:

```bash
# Set API key
export OPENAI_API_KEY='your-key'

# Run built-in test
python openai_client.py
```

This will:
1. Test simple generation
2. Test multiple completions (n=3)
3. Test Qwen chat template parsing
4. Print usage statistics and cost estimates

## Troubleshooting

### Rate Limits

If you hit rate limits:

```python
# Slow down requests
import asyncio

async def generate_with_backoff(client, prompt, kwargs):
    for attempt in range(3):
        try:
            return await client.async_generate(prompt, kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### API Key Not Found

```
ValueError: OpenAI API key not found
```

**Solution**: Set the environment variable:
```bash
export OPENAI_API_KEY='sk-...'
```

Or pass directly:
```python
client = OpenAIAPIClient(api_key="sk-...")
```

### Timeout Errors

For long reasoning tasks, increase timeout:

```python
client = OpenAIAPIClient(
    model="gpt-4-turbo-preview",
    timeout=1200.0  # 20 minutes
)
```

### Chat Template Parsing Issues

The client automatically handles Qwen chat templates. If you encounter parsing issues:

```python
# Check what messages are being sent
import logging
logging.basicConfig(level=logging.DEBUG)

# The client will log the parsed messages
```

## Migration Checklist

- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Replace `SGLangAPIClient` import with `OpenAIAPIClient`
- [ ] Update model names (QwQ-32B → gpt-4-turbo, Qwen2.5-72B → gpt-3.5-turbo)
- [ ] Test with small batch (10-20 questions)
- [ ] Monitor costs with `estimate_cost()`
- [ ] Adjust `timeout` parameter if needed for long tasks
- [ ] Run full generation pipeline
- [ ] Compare output quality with local Qwen results

## FAQ

**Q: Can I use gpt-3.5-turbo for everything to save money?**

A: You can, but reasoning quality may suffer. We recommend:
- Use gpt-4-turbo for complex reasoning tasks
- Use gpt-3.5-turbo for simple instruction following

**Q: How does this compare to local Qwen models in quality?**

A:
- gpt-4-turbo is generally comparable or better than QwQ-32B for reasoning
- gpt-3.5-turbo is faster but slightly lower quality than Qwen2.5-72B-Instruct
- For production use, test with your specific use case

**Q: Can I use this with other OpenAI-compatible APIs?**

A: Yes! Set `base_url` to your endpoint:
```python
client = OpenAIAPIClient(
    model="your-model",
    base_url="https://your-endpoint.com/v1"
)
```

Works with: Azure OpenAI, Together AI, Anyscale, Replicate, and others.

**Q: What about data privacy?**

A: OpenAI's API has data retention policies. For sensitive data:
- Use Azure OpenAI (enterprise SLA)
- Or stick with local hosting
- Review OpenAI's data usage policy: https://openai.com/policies/api-data-usage-policies

**Q: How can I reduce costs further?**

A:
1. Use gpt-3.5-turbo instead of gpt-4
2. Reduce `max_new_tokens` to minimum needed
3. Batch similar requests together
4. Cache results to avoid re-generation
5. Use local models for development, OpenAI for production

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the built-in test: `python openai_client.py`
3. Review OpenAI API docs: https://platform.openai.com/docs/
4. File an issue in the ASearcher repository
