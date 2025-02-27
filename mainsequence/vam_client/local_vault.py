import os
import sqlite3
import json
from pathlib import Path

from cryptography.fernet import Fernet

VAULT_PATH=os.environ.get('VAULT_PATH')
DATABASE_FILE,ENCRYPTION_KEY_FILE=None,None
if VAULT_PATH is not None:
    if VAULT_PATH =="LOCAL":
        root_path = Path(__file__).resolve().parent.parent
        DATABASE_FILE = f'{root_path}/local_vault.db'
        ENCRYPTION_KEY_FILE = f'{root_path}/VAULT_ENCRYPTION_KEY'



def get_secrets_for_account_id(account_id):
    """
    Gets the secrets from the vault from an specific account_id
    Parameters
    ----------
    account_id

    Returns
    -------

    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
            SELECT account_id, secrets FROM api_keys
            WHERE account_id = ?
        ''', (account_id,))
    results = cursor.fetchall()
    conn.close()

    entries = []
    for account_id, encrypted_secrets in results:
        decrypted_secrets = decrypt_secrets(encrypted_secrets)
        entries.append({
            'account_id': account_id,
            'secrets': decrypted_secrets
        })
        return entries[0]


def generate_key():
    """Generate an encryption key and save it to a file."""
    key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as key_file:
        key_file.write(key)

def load_key():
    """Load the encryption key from the file."""
    with open(ENCRYPTION_KEY_FILE, 'rb') as key_file:
        return key_file.read()

def encrypt_secrets(secrets):
    """Encrypt the secrets JSON."""
    key = load_key()
    f = Fernet(key)
    return f.encrypt(json.dumps(secrets).encode())

def decrypt_secrets(encrypted_secrets):
    """Decrypt the secrets JSON."""
    key = load_key()
    f = Fernet(key)
    return json.loads(f.decrypt(encrypted_secrets).decode())

def validate_secrets(secrets, execution_venue_symbol):
    """Validate that secrets conform to the schema for the given execution venue."""
    schema = {
        'api_key': str,
        'secret_key': str
    }

    for key, expected_type in schema.items():
        if key not in secrets:
            raise ValueError(f"Missing required key for {execution_venue_symbol}: {key}")
        if not isinstance(secrets[key], expected_type):
            raise ValueError(f"Invalid type for {key} in {execution_venue_symbol}. Expected {expected_type}")
        if isinstance(secrets[key], str) and not secrets[key].strip():
            raise ValueError(f"'{key}' must be a non-empty string for {execution_venue_symbol}")


def get_all_entries_in_vault_for_venue(execution_venue_symbol):
    """Retrieve all entries from the database for a specific execution venue."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT account_id, secrets FROM api_keys
        WHERE execution_venue_symbol = ?
    ''', (execution_venue_symbol,))
    results = cursor.fetchall()
    conn.close()

    entries = []
    for account_id, encrypted_secrets in results:
        decrypted_secrets = decrypt_secrets(encrypted_secrets)
        entries.append({
            'account_id': account_id,
            'secrets': decrypted_secrets
        })

    return entries