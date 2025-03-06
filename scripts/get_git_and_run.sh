#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# Decide whether to clone via SSH or API token
# If AUTHENTICATION_METHOD=api (and we have GIT_API_TOKEN + GIT_REPO_URL),
# we clone with HTTPS + embedded token. Otherwise, we use the old SSH approach.
# -------------------------------------------------------------
if [ "${AUTHENTICATION_METHOD:-ssh}" = "api" ] && [ -n "${GIT_API_TOKEN:-}" ] && [ -n "${GIT_REPO_URL:-}" ]; then
  echo "Using API token approach..."

  # Make sure we don't get prompted for host key verification
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh

  cat <<EOF > ~/.ssh/config
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF

  chmod 600 ~/.ssh/config

  # Ensure we don't do SSH
  export GIT_SSH_COMMAND="echo skipping_ssh"

  # Clone the Git repository into /tmp/repo
  REPO_PATH="/tmp/repo"
  rm -rf "$REPO_PATH"
  REMOTE_URL=$(echo "$GIT_REPO_URL" | sed "s|https://|https://${GIT_API_TOKEN}@|")
  git clone "$REMOTE_URL" "$REPO_PATH"

else
  echo "Using SSH key approach..."

  # Write the private key to a secure file
  SSH_KEY_FILE="$(mktemp)"
  echo "$GIT_PRIVATE_KEY" > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"

  # Start the SSH agent and add our private key
  eval "$(ssh-agent -s)"
  ssh-add "$SSH_KEY_FILE"

  # Make sure we don't get prompted for host key verification
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh

  cat <<EOF > ~/.ssh/config
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF

  chmod 600 ~/.ssh/config

  # Ensure git uses the same SSH options
  export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

  # Clone the Git repository into /tmp/repo
  REPO_PATH="/tmp/repo"
  rm -rf "$REPO_PATH"
  git clone "$GIT_SSH_URL" "$REPO_PATH"
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

if [ -f "${REPO_PATH}/requirements.txt" ]; then
    python -m pip install -r "${REPO_PATH}/requirements.txt"
else
    echo "requirements.txt not found at ${REPO_PATH}, skipping installation."
fi

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
