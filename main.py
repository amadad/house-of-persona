#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    template = "math"  # Can be "instruction", "npc" or "knowledge"
    sample_size = 10   # Set to 0 for full 200k personas
    out_path = f"gpt4o_{template}_synthesis_output.jsonl"
    
    # Add the current directory to Python path
    os.environ["PYTHONPATH"] = "."
    
    # Build the command arguments
    cmd_args = [
        "python", 
        "code/openai_synthesize.py",
        "--template", template,
        "--sample_size", str(sample_size),
        "--output_path", out_path
    ]
    
    # Execute the command
    os.execvp(cmd_args[0], cmd_args)

if __name__ == "__main__":
    main() 