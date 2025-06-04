#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="/tmp/repo"
export VFB_PROJECT_PATH="$REPO_PATH/$PROJECT_LIBRARY_NAME"

# check if we only execute a command on running
if [ -n "${1:-}" ]; then
  export COMMAND_ID="$1"
  python -m mainsequence.virtualfundbuilder run_resource "$EXECUTION_TYPE"
  exit 0
fi


# Source our new utils
source "$(dirname "$0")/utils.sh"

# -------------------------------------------------------------
# Decide whether to clone via SSH or API token
# -------------------------------------------------------------


if [ "${AUTHENTICATION_METHOD:-ssh}" = "api" ] && [ -n "${GIT_API_TOKEN:-}" ] && [ -n "${GIT_REPO_URL:-}" ]; then
  clone_via_api_token
else
  clone_via_ssh_key
fi

# -------------------------------------------------------------
# Set necessary variables
# (Note: the $REPO_PATH above is where we cloned the code)
# -------------------------------------------------------------
export TDAG_CONFIG_PATH=~/tdag/default_config.yml
export TDAG_RAY_CLUSTER_ADDRESS="ray://localhost:10001"
export TDAG_RAY_API_ADDRESS="http://localhost:8265"
export TDAG_RAY_SERVE_HOST="0.0.0.0"
export TDAG_RAY_SERVE_PORT="8003"
export MLFLOW_ENDPOINT="http://localhost:5000"

cd "$REPO_PATH"
export GIT_HASH="$(git rev-parse HEAD)"

# Install requirements
install_requirements_if_present "${REPO_PATH}/requirements.txt"

# -------------------------------------------------------------
# Execute your script
# -------------------------------------------------------------
if [ -z "${EXECUTION_OBJECT:-}" ]; then
  echo "EXECUTION_OBJECT is not set. Running only with EXECUTION_TYPE."
  python -m mainsequence.virtualfundbuilder run_resource "$EXECUTION_TYPE"
else
  python -m mainsequence.virtualfundbuilder run_resource "$EXECUTION_TYPE" "$EXECUTION_OBJECT"
fi

# -------------------------------------------------------------
# Cleanup: remove the temporary SSH key file if it exists
# -------------------------------------------------------------
if [ "${AUTHENTICATION_METHOD:-ssh}" != "api" ] && [ -n "${SSH_KEY_FILE:-}" ] && [ -f "$SSH_KEY_FILE" ]; then
  rm -f "$SSH_KEY_FILE"
fi