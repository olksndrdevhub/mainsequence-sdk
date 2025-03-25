#!/usr/bin/env bash
set -euo pipefail

# Source our new utils
source "$(dirname "$0")/utils.sh"

# -------------------------------------------------------------
# Decide whether to clone via SSH or API token
# -------------------------------------------------------------
REPO_PATH="/tmp/repo"

# Configure Git identity
git config --global user.email "ms_pod@example.com" || true
git config --global user.name "ms_pod" || true
git config --global --add safe.directory "$REPO_PATH" || true

if [ "${AUTHENTICATION_METHOD:-ssh}" = "api" ] && [ -n "${GIT_API_TOKEN:-}" ] && [ -n "${GIT_REPO_URL:-}" ]; then
  clone_via_api_token
else
  clone_via_ssh_key
fi

# -------------------------------------------------------------
# Set necessary variables
# (Note: the $REPO_PATH above is where we cloned the code)
# -------------------------------------------------------------
export VFB_PROJECT_PATH="$REPO_PATH/$PROJECT_LIBRARY_NAME"
export TDAG_CONFIG_PATH=~/tdag/default_config.yml
export TDAG_RAY_CLUSTER_ADDRESS="ray://localhost:10001"
export TDAG_RAY_API_ADDRESS="http://localhost:8265"
export TDAG_RAY_SERVE_HOST="0.0.0.0"
export TDAG_RAY_SERVE_PORT="8003"
export MLFLOW_ENDPOINT="http://localhost:5000"

cd "$REPO_PATH"
export GIT_HASH="$(git rev-parse HEAD)"

# Copy examples
cp -a "/opt/code/mainsequence-sdk/examples/time_series" "$VFB_PROJECT_PATH/time_series" || echo "WARNING: Copy TimeSeries step failed!"
cp -a "/opt/code/mainsequence-sdk/tests/system_tests" "$VFB_PROJECT_PATH/scripts" || echo "WARNING: Copy System Tests step failed!"

echo "Copy examples commit"
cd "$REPO_PATH"
git add "$REPO_PATH"
git commit -am "copy examples commit"
git push

# Update the backend that project is setup
update_backend_job_status "SUCCEEDED"

echo ">> Copy files script completed."
echo "======================================================"
