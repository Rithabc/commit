# CommitGen LoRA: AI Git Hook

This project implements an AI-powered Git Hook that automatically writes highly accurate, conventional Git commit messages based on your staged files. It uses a **fine-tuned Qwen 2.5 Coder 1.5B model** with a custom LoRA adapter.

## Included Files
- `commitgen_adapter.gguf`: The fine-tuned LoRA weights trained specifically on commit message formatting and logic.
- `Modelfile`: The configuration blueprint that instructs Ollama how to load the base model and fuse the adapter.
- `prepare-commit-msg.py`: The actual Git hook script that intercepts your commits, reads the diff, and talks to the AI.
- `README.md`: This instruction file.

---

## How to Run & Grade This Project

You do not need to download massive multi-gigabyte models manually to grade this. The project is designed to be plug-and-play using Ollama!

### 1. Prerequisites
- Download and install [Ollama](https://ollama.com/) (Available for Windows, Mac, and Linux).
- Ensure Python 3 is installed.

### 2. Build the Model
Open your terminal inside this extracted folder and run:
```bash
ollama create commitgen -f Modelfile
```
*Note: Ollama will automatically download the ~1GB `qwen2.5-coder:1.5b` base model from the internet and instantly fuse the included `commitgen_adapter.gguf` file to it!*

### 3. Test the Model

#### Option A: Quick Manual Test
Make a quick change to any text file in this folder, and stage it:
```bash
git add .
```
Then run the python script directly to see the AI generate a message based on your change:
```bash
python prepare-commit-msg.py test_msg.txt
```
*(The AI-generated commit message will be printed to your terminal and saved to `test_msg.txt`).*

#### Option B: Use the Git Hook (Intended Use)
To see it work exactly as designed:
1. Copy the `prepare-commit-msg.py` script into the `.git/hooks/` folder of this repository (or any local Git repository).
2. Rename it to exactly `prepare-commit-msg` (remove the `.py` extension).
3. *(If on Mac/Linux, make it executable: `chmod +x .git/hooks/prepare-commit-msg`)*
4. Make a change, run `git add .`, and then simply run `git commit`. 

The hook will automatically intercept the commit process, ping the fine-tuned model, and pre-fill your commit editor with the AI's generated message!
