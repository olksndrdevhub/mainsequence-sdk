#!/usr/bin/env bash

########################################
# Ensure ~/.ssh/config is set for no host key checks
########################################
ensure_no_host_key_check() {
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  cat <<EOF > ~/.ssh/config
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
  chmod 600 ~/.ssh/config
}

########################################
# Write an SSH private key to a secure file and load via ssh-agent
# Requires environment variable:
#    GIT_PRIVATE_KEY
########################################
setup_ssh_key() {
  SSH_KEY_FILE="$(mktemp)"
  echo "$GIT_PRIVATE_KEY" > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"

  # Start the SSH agent and add our private key
  eval "$(ssh-agent -s)"
  ssh-add "$SSH_KEY_FILE"

  # In case another script needs to remove it
  export SSH_KEY_FILE
}

########################################
# Clone via API token
# Requires environment variables:
#    GIT_API_TOKEN
#    GIT_REPO_URL
#    REPO_PATH
########################################
clone_via_api_token() {
  echo "Using API token approach..."
  ensure_no_host_key_check

  # Ensure we don't do SSH
  export GIT_SSH_COMMAND="echo skipping_ssh"

  rm -rf "$REPO_PATH"
  REMOTE_URL=$(echo "$GIT_REPO_URL" | sed "s|https://|https://${GIT_API_TOKEN}@|")
  git clone "$REMOTE_URL" "$REPO_PATH"
}

########################################
# Clone via SSH.
# If GIT_PRIVATE_KEY is provided, it will be used for authentication.
# Otherwise, it will attempt an anonymous clone using the SSH URL.
# Requires environment variables:
#    GIT_PRIVATE_KEY (optional)
#    GIT_SSH_URL
#    REPO_PATH
########################################
clone_via_ssh_key() {
  # If a private key is provided, set it up with the ssh-agent.
  if [ -n "${GIT_PRIVATE_KEY:-}" ]; then
    echo "Using SSH key approach..."
    setup_ssh_key
  else
    echo "No SSH key provided. Attempting anonymous clone via SSH URL."
  fi

  ensure_no_host_key_check

  export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

  rm -rf "$REPO_PATH"
  git clone "$GIT_SSH_URL" "$REPO_PATH"
}

########################################
# If a requirements.txt exists, install it
#   $1: path to requirements.txt
########################################
install_requirements_if_present() {
  local req_file="$1"
  if [ -f "$req_file" ]; then
    python -m pip install uv
    uv pip install -r "$req_file"
  else
    echo "requirements.txt not found at ${req_file}, skipping installation."
  fi
}

########################################
# Pull changes from remote:
#   if AUTHENTICATION_METHOD=api: uses API token
#   else: uses SSH approach
########################################
pull_changes() {
  if [ "$AUTHENTICATION_METHOD" = "api" ] && [ -n "${GIT_API_TOKEN:-}" ] && [ -n "${GIT_REPO_URL:-}" ]; then
    echo "Using API token for Git pull..."
    REMOTE_URL=$(echo "$GIT_REPO_URL" | sed "s|https://|https://$GIT_API_TOKEN@|")
    git remote set-url origin "$REMOTE_URL" 2>/dev/null || true
    git pull origin main || true
  else
    echo "Using SSH-based approach..."
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git fetch origin || true
    git merge origin/main || true
  fi
}

########################################
# Post job status to the backend
# Uses environment variables:
#   TDAG_ENDPOINT, MAINSEQUENCE_TOKEN
#   status is passed as the 1st argument
########################################
update_backend_job_status() {
  local status="$1"
  local url="${TDAG_ENDPOINT}/orm/api/pods/job/job_run_status/"

  http_code=$(curl -s -o /tmp/update_response.txt -w "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Token ${MAINSEQUENCE_TOKEN}" \
    -d "{\"status\": \"${status}\"}" \
    "${url}")

  response_body=$(cat /tmp/update_response.txt)
  if [ "${http_code}" -eq 200 ]; then
    echo "Update success: ${response_body}"
  else
    echo "Error updating job (HTTP code ${http_code}): ${response_body}"
  fi
}