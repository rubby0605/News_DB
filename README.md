#!/usr/bin/env bash
set -euo pipefail

# readme.sh
# Generate/refresh README.md for this repo.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROJECT_NAME="$(basename "$ROOT_DIR")"

# Try to get remote URL (optional)
REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"

# Collect python files (top-level + subfolders, excluding venv/hidden)
PY_FILES="$(find . \
  -type f -name "*.py" \
  -not -path "./.git/*" \
  -not -path "./.venv/*" \
  -not -path "./venv/*" \
  -not -path "./__pycache__/*" \
  -not -path "*/__pycache__/*" \
  | sed 's|^\./||' \
  | sort || true)"

# Collect some common folders/files
HAS_DATA_DIR="no"
[ -d "./Data" ] && HAS_DATA_DIR="yes"

# A small tree-like view (no external `tree` dependency)
STRUCTURE="$(find . -maxdepth 2 \
  -not -path "./.git*" \
  -not -path "./.venv*" \
  -not -path "./venv*" \
  -not -path "./__pycache__*" \
  -not -path "*/__pycache__*" \
  -print \
  | sed 's|^\./||' \
  | awk 'NF' \
  | sort)"

TODAY="$(date '+%Y-%m-%d')"

cat > README.md <<EOF
# ${PROJECT_NAME}

用 LLM 分析新聞情緒預測漲跌（News sentiment \u2192 price movement / signal）

> 本 README 由 \`./readme.sh\` 於 ${TODAY} 自動生成/更新。

## Repo
${REMOTE_URL:-（尚未設定 git remote）}

## 你可以用它做什麼
- 彙整/清洗新聞資料（例如 CSV）
- 用 LLM/情緒模型把新聞轉成特徵
- 產生交易訊號或預測（請務必注意回測偏誤與資料延遲）

> ⚠️ 免責：本專案僅供研究/學習，不構成投資建議。

## 專案結構（max depth=2）
\`\`\`
${STRUCTURE}
\`\`\`

## 快速開始
### 1) 下載
\`\`\`bash
git clone ${REMOTE_URL:-<your_repo_url>}
cd ${PROJECT_NAME}
\`\`\`

### 2) 安裝依賴
本 repo 以 Python 為主。你可以直接用系統 Python + pip（不一定要 venv）。

\`\`\`bash
python3 -V
python3 -m pip install -r requirements.txt
\`\`\`

> 如果你還沒有 \`requirements.txt\`，可以先用：
\`\`\`bash
python3 -m pip install -U pip
python3 -m pip freeze > requirements.txt
\`\`\`

### 3) 執行（依你的主程式調整）
\`\`\`bash
python3 your_main.py
\`\`\`

## 資料夾說明
- \`Data/\`: ${HAS_DATA_DIR}
  - 如果你有放新聞/股價資料（CSV 等），建議在這裡集中管理
- \`.gitignore\`: 已存在（建議把大型資料、API keys 排除）

## Python 檔案清單
EOF

if [ -n "$PY_FILES" ]; then
  echo "" >> README.md
  echo "$PY_FILES" | while IFS= read -r f; do
    echo "- \`$f\`" >> README.md
  done
else
  echo "" >> README.md
  echo "- （目前未偵測到 .py 檔案）" >> README.md
fi

cat >> README.md <<'EOF'

## 建議你下一步補上（很值得）
- [ ] requirements.txt（固定環境，避免跑不起來）
- [ ] .env.example（如果有 API key，別直接寫進程式）
- [ ] 一個清楚的 entry point：例如 `main.py` 或 `run.py`
- [ ] 回測注意事項（避免 look-ahead bias、避免 shuffle 時序資料）

---

### 生成 README
```bash
chmod +x readme.sh
./readme.sh
