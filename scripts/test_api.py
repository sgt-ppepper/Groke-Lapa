#!/usr/bin/env python3
"""Simple test to verify LLM API connectivity."""
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import time
from openai import OpenAI

def test_api():
    """Test raw API call without our wrappers."""
    api_key = os.getenv("LAPATHON_API_KEY")
    base_url = "http://146.59.127.106:4000"
    
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: NOT SET")
    print(f"Base URL: {base_url}")
    print()
    
    # Test 1: Simple ping with tiny prompt
    print("=" * 50)
    print("TEST 1: Minimal prompt (10 tokens)")
    print("=" * 50)
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0
    )
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="lapa",
            messages=[{"role": "user", "content": "Привіт! Скажи 'Так'."}],
            temperature=0.5,
            max_tokens=50
        )
        elapsed = time.time() - start
        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED after {elapsed:.1f}s: {e}")
        return
    
    print()
    
    # Test 2: Check mamay model
    print("=" * 50)
    print("TEST 2: Mamay model")
    print("=" * 50)
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="mamay",
            messages=[{"role": "user", "content": "Скажи 'Привіт' одним словом."}],
            temperature=0.5,
            max_tokens=50
        )
        elapsed = time.time() - start
        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED after {elapsed:.1f}s: {e}")
    
    print()
    
    # Test 3: Medium prompt like content generator uses
    print("=" * 50)
    print("TEST 3: Medium prompt (~500 chars)")
    print("=" * 50)
    
    prompt = """Ти вчитель алгебри для 9 класу.

Контекст: Теорема Віета встановлює зв'язок між коренями квадратного рівняння та його коефіцієнтами.

Поясни теорему Віета у 2-3 реченнях."""
    
    print(f"Prompt length: {len(prompt)} chars")
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="lapa",
            messages=[
                {"role": "system", "content": "Ти досвідчений вчитель алгебри."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        elapsed = time.time() - start
        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Response ({len(response.choices[0].message.content)} chars):")
        print(response.choices[0].message.content[:300] + "..." if len(response.choices[0].message.content) > 300 else response.choices[0].message.content)
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED after {elapsed:.1f}s: {e}")
    
    print()
    print("=" * 50)
    print("TESTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    test_api()
