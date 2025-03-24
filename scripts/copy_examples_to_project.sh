echo "======================================================"
echo "Copy files script invoked at: $(date)"
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


cp -a "/opt/code/mainsequence-sdk/examples/time_series" "$VFB_PROJECT_PATH/time_series" || echo "WARNING: Copy Notebooks step failed!"
cp -a "/opt/code/mainsequence-sdk/examples/" "$VFB_PROJECT_PATH/time_series" || echo "WARNING: Copy Notebooks step failed!"




# update the backend that project is setup
url="${TDAG_ENDPOINT}/orm/api/pods/job/job_run_status/"

http_code=$(curl -s -o /tmp/update_response.txt -w "%{http_code}" \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Token ${MAINSEQUENCE_TOKEN}" \
  -d "{\"status\": \"SUCCEEDED\"}" \
  "${url}")

response_body=$(cat /tmp/update_response.txt)
if [ "${http_code}" -eq 200 ]; then
  echo "Update success: ${response_body}"
else
  echo "Error updating job (HTTP code ${http_code}): ${response_body}"
fi

echo ">> Copy files script completed."
echo "======================================================"

