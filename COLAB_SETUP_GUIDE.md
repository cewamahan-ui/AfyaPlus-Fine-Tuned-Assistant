# Google Colab Setup Guide - AfyaPlus Fine-Tuning

## Step 1: Open the Notebook in Colab

### Option A: Upload directly
1. Go to https://colab.research.google.com
2. Click "File" > "Upload notebook"
3. Upload `AfyaPlus_Colab_Training.ipynb`

### Option B: Clone from GitHub
1. Go to https://colab.research.google.com
2. Click "File" > "Open notebook"
3. Click "GitHub" tab
4. Paste: `https://github.com/cewamahan-ui/AfyaPlus-Fine-Tuned-Assistant`
5. Select `AfyaPlus_Colab_Training.ipynb`

---

## Step 2: Mount Google Drive (Cell 1)
- Click the play button (▶️) on the first code cell
- Click "Connect to Google Drive" when prompted
- Select your Google account

---

## Step 3: Install Dependencies (Cell 2)
- Click play on the pip install cell
- Wait for "✓" to appear

---

## Step 4: Navigate to Project (Cell 3)
- Click play to see your files listed
- If error: adjust path to match your Drive folder name

---

## Step 5: Data Preparation (Cell 4-5)
- Click play on the data prep cell
- You should see: "Loaded 101 examples, Errors: 0"

---

## Step 6: Set GPU Runtime (IMPORTANT!)
1. Click "Runtime" menu at top
2. Select "Change runtime type"
3. Select "T4 GPU" from dropdown
4. Click "Save"

---

## Step 7: Get HuggingFace Token (Required for LLaMA)
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "colab-access"
4. Role: "read"
5. Click "Generate token"
6. COPY the token (starts with "hf_...")

---

## Step 8: Add Token to Notebook
1. Find Cell 6 (fine-tuning section)
2. Uncomment this line by removing the #:
   ```
   login(token="hf_your_token_here")
   ```
3. Replace `hf_your_token_here` with your actual token
4. Run the cell

---

## Step 9: Run Fine-Tuning (Cell 6)
- Click play on the fine-tuning cell
- **Training takes 2-3 hours!** 
- Keep the browser tab open
- Don't close your laptop/sleep

### While waiting:
- You can see progress in the output
- Look for "Step 10/..., loss: X.XX" messages
- When complete, you'll see "Done. Adapter saved..."

---

## Step 10: Merge Model (Cell 7)
- After training finishes, click play
- Takes ~5-10 minutes
- You'll see "Merged model saved..."

---

## Step 11: Test Inference (Cell 8)
- Click play to test sample queries
- See how the model responds to AfyaPlus questions
- Each query shows: Query, Gate status, Response

---

## Step 12: Get Anthropic API Key (For Evaluation)
1. Go to: https://console.anthropic.com/
2. Sign up / Log in
3. Click "API Keys" in left menu
4. Click "Create Key"
5. COPY the key (starts with "sk-ant-...")

---

## Step 13: Run Evaluation (Cell 9)
1. In Cell 9, replace `your_anthropic_api_key` with your actual key
2. Click play
3. Takes ~10-15 minutes for 11 test examples
4. Creates `comparison_results.csv`

---

## Step 14: Download Results (Cell 10)
- Click play to save files to Drive
- Files saved:
  - `comparison_results.csv`
  - `trainer_state.json` (for loss curve)

---

## Step 15: Update Your Memo
1. Download `comparison_results.csv` from Drive
2. Open `memo.md` in the project
3. Replace placeholder percentages with actual numbers from the CSV
4. Commit to GitHub

---

## Troubleshooting

### "No GPU available"
- Runtime > Change runtime type > T4 GPU
- If T4 not available, try: Runtime > Disconnect and delete runtime > Reconnect

### "Out of memory"
- Normal on free tier
- Try again during off-peak hours

### "Login failed"
- Make sure HuggingFace token is correct
- Token must start with "hf_"

### Colab disconnected
- Training auto-saves checkpoints every 10 steps
- Reconnect and run fine_tune.py again - it will resume from last checkpoint
