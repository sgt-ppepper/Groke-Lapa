#!/usr/bin/env python3
"""Test LLM API connectivity."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up from src/llm to Groke-Lapa
sys.path.insert(0, str(project_root))

# Load env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Now imports work
from src.llm.mamay import MamayLLM

llm = MamayLLM()

print("Calling LLM...")
response = llm.generate(
    prompt="Привіт! Скажи 'Так'.",
    system="Ти досвідчений вчитель математики.",
    temperature=0.7,
    max_tokens=50
)

print(f"Response: {response}")