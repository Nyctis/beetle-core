#!/usr/bin/env bash
# Set up an ordinary GPU server as a Beetle seller. Such a machine has no
# confidential hardware, so it sells at tier T3-C: the owner can read what runs
# on it, and buyers are told so.
#
#   node/install.sh                 # checks, installs dependencies, writes the unit
#   sudo systemctl enable --now beetle-host
#
# Needs Ubuntu 22.04 or 24.04, an NVIDIA driver and Docker. It installs uv and
# the NVIDIA Container Toolkit if missing, then builds the sandbox image.
# BEETLE_HOME picks the state directory.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
USER_NAME=$(id -un)
export BEETLE_HOME=${BEETLE_HOME:-$HOME/.beetle}
export PATH="$HOME/.local/bin:$PATH"

command -v docker >/dev/null || { echo "Docker is required: https://docs.docker.com/engine/install/"; exit 1; }
docker info >/dev/null 2>&1 || { echo "this user can't reach Docker: add it to the docker group"; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU=1
  if ! docker info 2>/dev/null | grep -q nvidia; then
    echo "== NVIDIA Container Toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  fi
fi

echo "== Python environment"
uv sync --quiet
BIN=$REPO/.venv/bin/beetle

echo "== sandbox image (GPU=$GPU)"
ARGS=()
[ "$GPU" = 1 ] && ARGS=(--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128)
docker image inspect beetle-sandbox:dev >/dev/null 2>&1 || \
  docker build -q -t beetle-sandbox:dev ${ARGS[@]+"${ARGS[@]}"} -f sandbox/Dockerfile .

echo "== systemd unit"
mkdir -p "$BEETLE_HOME"
sed -e "s#__BEETLE_BIN__#$BIN#" -e "s#__BEETLE_HOME__#$BEETLE_HOME#g" \
    -e "s#__BEETLE_REPO__#$REPO#" -e "s#__BEETLE_USER__#$USER_NAME#" node/beetle-host.service \
  | sudo tee /etc/systemd/system/beetle-host.service >/dev/null
sudo systemctl daemon-reload

cat <<NEXT

Installed. Next, as $USER_NAME:
  $BIN host init --chain chain.json --public-addr IP:41710 --rendezvous-dir DIR
  $BIN host doctor
  $BIN host list --models NAME --train-usd-per-mtok 1 --infer-in 0.1 --infer-out 0.3 --vllm-serve NAME
  sudo systemctl enable --now beetle-host
NEXT
