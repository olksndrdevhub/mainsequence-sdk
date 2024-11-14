import fire
import os
import sqlite3
import json

from mainsequence.vam_client.models import ExecutionVenue
from mainsequence.vam_client.utils import build_assets_for_venue, get_venue_from_symbol, get_mainsequence.vam_client_logger
# Constants
from .local_vault import (DATABASE_FILE,load_key,generate_key,
validate_secrets,encrypt_secrets,decrypt_secrets
                          )

logger = get_mainsequence.vam_client_logger()

class ClientApp:

    def create_local_vault(self):
        """Create an empty database with the required table."""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                account_id TEXT,
                execution_venue_symbol TEXT,
                secrets BLOB,
                PRIMARY KEY (account_id, execution_venue_symbol)
            )
        ''')
        conn.commit()
        conn.close()

        # Generate encryption key if it doesn't exist
        try:
            load_key()
        except FileNotFoundError:
            generate_key()

    def upsert_key_to_vault(self, account_id, execution_venue_symbol, secrets):
        """Insert or update a key in the database."""
        assert execution_venue_symbol in CONSTANTS.EXECUTION_VENUES_NAMES.keys(), "execution_venue_symbol not in tracked venues"

        ev,_ = ExecutionVenue.get(symbol=execution_venue_symbol)

        if isinstance(secrets, str):
            secrets = json.loads(secrets)

        validate_secrets(secrets, execution_venue_symbol)

        encrypted_secrets = encrypt_secrets(secrets)

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO api_keys (account_id, execution_venue_symbol, secrets)
            VALUES (?, ?, ?)
        ''', (account_id, execution_venue_symbol, encrypted_secrets))
        conn.commit()
        logger.info(f"Rows affected: {cursor.rowcount}", DATABASE_FILE)
        conn.close()

    def read_key_from_vault(self, account_id, execution_venue_symbol):
        """Read a key from the database."""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
                SELECT secrets FROM api_keys
                WHERE account_id = ? AND execution_venue_symbol = ?
            ''', (account_id, execution_venue_symbol))
        result = cursor.fetchone()
        conn.close()

        if result:
            encrypted_secrets = result[0]
            return decrypt_secrets(encrypted_secrets)
        else:
            return None


    def build_assets(self, execution_venue_symbol, api_key, secret_key):
        """ Build the assets for an execution venue """
        ev, _ = ExecutionVenue.filter(symbol=execution_venue_symbol)
        execution_venue = get_venue_from_symbol(execution_venue_symbol)
        build_assets_for_venue(
            execution_venue=execution_venue,
            api_key=api_key,
            api_secret=secret_key
        )


if __name__ == "__main__":
    fire.Fire(ClientApp)