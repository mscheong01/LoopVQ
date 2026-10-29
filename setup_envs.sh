#!/usr/bin/env bash
# Create the three conda environments this repo uses, and install into them.
#
#   ./setup_envs.sh                # all three
#   ./setup_envs.sh serve          # the serving engine (python/, sglang fork)
#   ./setup_envs.sh eval           # the lm_eval / EvalPlus client (no GPU)
#   ./setup_envs.sh build          # calibration: codebook fitting and fine-tuning (calibration/)
#
# Names default to config.sh's ENV_SERVE / ENV_EVAL and ENV_BUILD=loopvq-build:
#   ENV_SERVE=my-serve ./setup_envs.sh serve
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVE="${ENV_SERVE:-loopvq-serve}"
ENV_EVAL="${ENV_EVAL:-loopvq-eval}"
ENV_BUILD="${ENV_BUILD:-loopvq-build}"
WHICH=("$@"); [[ $# -eq 0 ]] && WHICH=(serve eval build)
command -v conda >/dev/null || { echo "conda not on PATH"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

has() { printf '%s\n' "${WHICH[@]}" | grep -qx "$1"; }

if has serve; then
  echo "== $ENV_SERVE : serving engine"
  conda env list | grep -q "^$ENV_SERVE " || conda create -n "$ENV_SERVE" python=3.10 -y
  conda activate "$ENV_SERVE"
  # The fork brings its own torch (2.9.1) and transformers (5.3.0). It installs as loopvq-sglang
  # and imports as sglang, so it cannot be confused with an upstream SGLang.
  pip install -e "$REPO/python"
  python -c "import torch, sglang" && echo "   ok: sglang imports"
  conda deactivate
fi

if has eval; then
  echo "== $ENV_EVAL : the lm_eval / EvalPlus client (no GPU)"
  # Separate env because lm_eval 0.4.9 does not import under transformers 5.x, which the fork
  # requires.
  conda env list | grep -q "^$ENV_EVAL " || conda create -n "$ENV_EVAL" python=3.10 -y
  conda activate "$ENV_EVAL"
  pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu \
                               --extra-index-url https://pypi.org/simple
  pip install lm_eval==0.4.9 transformers==4.53.2 evalplus==0.3.1 math-verify==0.9.0 tenacity "numpy<2"
  python -c "import lm_eval, evalplus, math_verify" && echo "   ok: lm_eval, evalplus, math_verify import"
  conda deactivate
fi

if has build; then
  echo "== $ENV_BUILD : calibration"
  # transformers 4.56.x: Ouro's remote code does not load under 5.x, and the tools pass dtype=,
  # which needs >= 4.56.
  conda env list | grep -q "^$ENV_BUILD " || conda create -n "$ENV_BUILD" python=3.11 -y
  conda activate "$ENV_BUILD"
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install transformers==4.56.2 accelerate datasets pandas pyarrow networkx safetensors
  python -c "import torch, transformers, networkx" && echo "   ok: calibration deps import"
  conda deactivate
fi
