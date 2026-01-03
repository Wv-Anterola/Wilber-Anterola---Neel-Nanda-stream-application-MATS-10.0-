# Manual Generation Check (Baseline vs Intervened)

This is a spot check to confirm why intervened outputs omit the "Final answer"
tag in the generation-phase probe. Outputs are truncated. Long runs of blank
lines are replaced by [blank lines omitted] for readability.

## Prompt 1
Prompt: What is 7 + 5?

Baseline (first 400 chars, truncated):
"""
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
Think step-by-step to solve this problem. Show your reasoning process.
End with: Final answer: <number>.

Question: What is 7 + 5?

Let me think through this step by step:
assistant
Sure, let's break it down step by step:

1. **Identify the numbers involved**: We have two numbers, 7 and 5.
2. **Understand the operatio
"""

Intervened (first 400 chars, truncated):
"""
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
Think step-by-step to solve this problem. Show your reasoning process.
End with: Final answer: <number>.

Question: What is 7 + 5?

Let me think through this step by step:
assistant
Sure

[blank lines omitted]
"""

Intervened (last 200 chars, truncated):
"""
e

[blank lines omitted]
"""

## Prompt 2
Prompt: If Sarah has 20 dollars and spends 8, how much does she have left?

Baseline (first 400 chars, truncated):
"""
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
Think step-by-step to solve this problem. Show your reasoning process.
End with: Final answer: <number>.

Question: If Sarah has 20 dollars and spends 8, how much does she have left?

Let me think through this step by step:
assistant
Sure, let's break it down step by step:

1. **Initial Amount**: Sarah starts with 20 
"""

Intervened (first 400 chars, truncated):
"""
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
Think step-by-step to solve this problem. Show your reasoning process.
End with: Final answer: <number>.

Question: If Sarah has 20 dollars and spends 8, how much does she have left?

Let me think through this step by step:
assistant
To

[blank lines omitted]
"""

Intervened (last 200 chars, truncated):
"""
o

[blank lines omitted]
"""

Summary: patched runs often emit a short stub and then blank lines. This
explains why the generation-phase probe finds zero "Final answer" tags in
intervened outputs.
