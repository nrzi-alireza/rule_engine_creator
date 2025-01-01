import sqlite3
from typing import Any, List, Tuple


class Database:
    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            raise Exception(f"Error connecting to database: {e}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def execute_query(self, query: str, params: Tuple = ()) -> None:
        """Execute a query without returning results."""
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
        except sqlite3.Error as e:
            raise Exception(f"Error executing query: {e}")

    def fetch_all(self, query: str, params: Tuple = ()) -> List[Tuple[Any, ...]]:
        """Execute a query and return all results."""
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            raise Exception(f"Error fetching data: {e}")

    def create_table(self, table_name: str, columns: list[str]) -> None:
        """Create a new table."""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        self.execute_query(query)
