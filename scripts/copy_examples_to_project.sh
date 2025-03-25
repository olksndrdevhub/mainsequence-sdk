#!/usr/bin/env bash
set +e  # Don't abort on error

# Source our new utils
source "$(dirname "$0")/utils.sh"

echo "======================================================"
echo "Copy files script invoked at: $(date)"
echo "Running as user: $(whoami)"

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

# Fix SSH key perms if re-using SSH
chmod 600 "$HOME_DIR/.ssh/id_rsa" 2>/dev/null || true

cd "$ROOT_PROJECT_PATH" || true

# Pull from remote
pull_changes

# Copy examples
cp -a "/opt/code/mainsequence-sdk/examples/time_series" "$VFB_PROJECT_PATH/scripts" || echo "WARNING: Copy TimeSeries step failed!"

echo "Copy examples commit"
cd "$ROOT_PROJECT_PATH"
git add "$ROOT_PROJECT_PATH"
git commit -am "copy examples commit"
git push

# Update the backend that project is setup
update_backend_job_status "SUCCEEDED"

echo ">> Copy files script completed."
echo "======================================================"
