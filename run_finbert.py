import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Set environment variables and run the finetune_finbert.py script.
    """
    # Get script path
    script_path = Path(__file__).parent / "scripts" / "finetune_finbert.py"
    
    # Set environment variables to completely disable caching
    os.environ["HF_DATASETS_CACHE"] = str(Path.cwd() / "data" / "no_cache")
    os.environ["TRANSFORMERS_CACHE"] = str(Path.cwd() / "data" / "no_cache") 
    os.environ["HF_HOME"] = str(Path.cwd() / "data" / "no_cache")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    # Create no_cache directory
    (Path.cwd() / "data" / "no_cache").mkdir(exist_ok=True, parents=True)
    
    # Get command line arguments
    args = sys.argv[1:]
    
    # Build command
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"Running FinBERT fine-tuning script with cache disabled")
    print(f"Cache directory set to: {os.environ['HF_DATASETS_CACHE']}")
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 