echo "======================================================"
echo "PostStart script invoked at: $(date)"
echo "Running as user: $(whoami)"
set +e  # Don't abort on error
mkdir -p /tmp

if [ -z "$HOME_DIR" ]; then
  HOME_DIR="/tmp/repo"
fi
ROOT_PROJECT_PATH="$HOME_DIR/$PROJECT_NAME"

if [ -z "$VFB_PROJECT_PATH" ]; then
  VFB_PROJECT_PATH=$ROOT_PROJECT_PATH/$PROJECT_LIBRARY_NAME
fi

echo ">> HOME_DIR: $HOME_DIR"
echo ">> PROJECT_NAME: $PROJECT_NAME"
echo ">> VFB_PROJECT_PATH: $VFB_PROJECT_PATH"
echo ">> AUTHENTICATION_METHOD: $AUTHENTICATION_METHOD"

if [ ! -d "$VFB_PROJECT_PATH" ]; then
  echo "Folder $VFB_PROJECT_PATH does not exist. Cloning repo..."

  # Basic SSH config if we’re using SSH-based Git
  mkdir -p "$HOME_DIR/.ssh" || true
  chmod 700 "$HOME_DIR/.ssh" || true

  echo "Host *" > "$HOME_DIR/.ssh/config" 2>/dev/null || true
  echo "  StrictHostKeyChecking no" >> "$HOME_DIR/.ssh/config" 2>/dev/null || true
  chmod 600 "$HOME_DIR/.ssh/config" || true

  # Write private key (only relevant if using SSH)
  if [ -n "$GIT_PRIVATE_KEY" ]; then
    echo "$GIT_PRIVATE_KEY" > "$HOME_DIR/.ssh/id_rsa"
    chmod 600 "$HOME_DIR/.ssh/id_rsa"
    unset GIT_PRIVATE_KEY
  fi

  # GitHub known hosts
  if ! grep -q "github.com" "$HOME_DIR/.ssh/known_hosts" 2>/dev/null; then
    ssh-keyscan -t ed25519 github.com >> "$HOME_DIR/.ssh/known_hosts" 2>/dev/null || true
    chmod 600 "$HOME_DIR/.ssh/known_hosts" || true
  fi

  # Start SSH agent
  eval "$(ssh-agent -s)" 2>/dev/null || true
  ssh-add "$HOME_DIR/.ssh/id_rsa" 2>/dev/null || true

  # Configure Git identity
  git config --global user.email "ms_pod@example.com" || true
  git config --global user.name "ms_pod" || true
  git config --global --add safe.directory "$ROOT_PROJECT_PATH" || true

  # ---------------------------------------------------------------------
  # CLONE the repo using either API or SSH
  # ---------------------------------------------------------------------
  if [ "$AUTHENTICATION_METHOD" = "api" ] && [ -n "$GIT_API_TOKEN" ] && [ -n "$GIT_REPO_URL" ]; then
    echo "Using API token for Git clone..."
    REMOTE_URL=$(echo "$GIT_REPO_URL" | sed "s|https://|https://$GIT_API_TOKEN@|")
    git clone "$REMOTE_URL" "$ROOT_PROJECT_PATH" || true
  else
    echo "Using SSH-based approach..."
    if [ -n "$GIT_SSH_URL" ]; then
      git clone "$GIT_SSH_URL" "$ROOT_PROJECT_PATH" || true
    else
      echo "WARNING: GIT_SSH_URL is not set or empty. Skipping clone."
    fi
  fi

  # create default folders if not exist
  mkdir -p "$VFB_PROJECT_PATH"
  mkdir -p "$VFB_PROJECT_PATH/notebooks"
  mkdir -p "$VFB_PROJECT_PATH/configurations"
  mkdir -p "$VFB_PROJECT_PATH/scripts"
  mkdir -p "$VFB_PROJECT_PATH/signals"
  mkdir -p "$VFB_PROJECT_PATH/rebalance_strategies"

  # Write some boilerplate files if needed
  touch "$VFB_PROJECT_PATH/__init__.py"
  touch "$VFB_PROJECT_PATH/signals/__init__.py"
  touch "$VFB_PROJECT_PATH/rebalance_strategies/__init__.py"

  # Copy a sample notebook (optional)
  cp -a "/opt/code/mainsequence-sdk/notebooks/Getting Started.ipynb" "$VFB_PROJECT_PATH/notebooks" || echo "WARNING: Copy step failed!"

  chown -R 1000:100 "$HOME_DIR" 2>/dev/null || true
  python -m pip freeze > ${ROOT_PROJECT_PATH}/requirements.txt

  echo "Create initial commit"
  cd $ROOT_PROJECT_PATH
  git add $ROOT_PROJECT_PATH
  git commit -am "initial commit"
  git push
else
  echo "Folder $VFB_PROJECT_PATH already exists. Updating repo..."

  # Fix SSH key perms if re-using SSH
  chmod 600 "$HOME_DIR/.ssh/id_rsa" 2>/dev/null || true

  cd "$ROOT_PROJECT_PATH" || true

  # Pull from remote
  if [ "$AUTHENTICATION_METHOD" = "api" ] && [ -n "$GIT_API_TOKEN" ] && [ -n "$GIT_REPO_URL" ]; then
    echo "Using API token for Git pull..."
    REMOTE_URL=$(echo "$GIT_REPO_URL" | sed "s|https://|https://$GIT_API_TOKEN@|")
    git remote set-url origin "$REMOTE_URL" 2>/dev/null || true
    git pull origin main || true
  else
    echo "Using SSH-based approach..."
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git fetch origin || true
    git merge origin/main || true
  fi
fi

if [ -f "${ROOT_PROJECT_PATH}/requirements.txt" ]; then
    python -m pip install -r "${ROOT_PROJECT_PATH}/requirements.txt"
else
    echo "requirements.txt not found at ${ROOT_PROJECT_PATH}, skipping installation."
fi

echo ">> PostStart script completed."
echo "======================================================"