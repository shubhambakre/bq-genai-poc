"""BigQuery Client module for handling connections to Google BigQuery."""

import os
from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryClient:
    """A client class for managing BigQuery connections and operations."""

    def __init__(self, project_id: str, credentials_path: str | None = None):
        """
        Initialize the BigQuery client.

        Args:
            project_id: The Google Cloud project ID.
            credentials_path: Path to the service account JSON file.
                              If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        self.project_id = project_id
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._client: bigquery.Client | None = None
        self._credentials: service_account.Credentials | None = None

    @property
    def credentials(self) -> service_account.Credentials:
        """Load and return service account credentials."""
        if self._credentials is None:
            if not self.credentials_path:
                raise ValueError(
                    "No credentials path provided. Set GOOGLE_APPLICATION_CREDENTIALS "
                    "environment variable or pass credentials_path to constructor."
                )
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(
                    f"Service account file not found: {self.credentials_path}"
                )
            self._credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
        return self._credentials

    @property
    def client(self) -> bigquery.Client:
        """Get or create the BigQuery client instance."""
        if self._client is None:
            self._client = bigquery.Client(
                project=self.project_id,
                credentials=self.credentials,
            )
        return self._client

    def get_connection_uri(self, dataset_id: str) -> str:
        """
        Generate the SQLAlchemy connection URI for BigQuery.

        Args:
            dataset_id: The BigQuery dataset ID.

        Returns:
            A SQLAlchemy-compatible BigQuery connection URI.
        """
        return f"bigquery://{self.project_id}/{dataset_id}?credentials_path={self.credentials_path}"

    def test_connection(self) -> bool:
        """
        Test the BigQuery connection by running a simple query.

        Returns:
            True if connection is successful, raises exception otherwise.
        """
        query = "SELECT 1"
        query_job = self.client.query(query)
        _ = list(query_job.result())
        return True

    def list_datasets(self) -> list[str]:
        """
        List all datasets in the project.

        Returns:
            A list of dataset IDs.
        """
        datasets = list(self.client.list_datasets())
        return [dataset.dataset_id for dataset in datasets]

    def list_tables(self, dataset_id: str) -> list[str]:
        """
        List all tables in a dataset.

        Args:
            dataset_id: The BigQuery dataset ID.

        Returns:
            A list of table IDs.
        """
        tables = list(self.client.list_tables(dataset_id))
        return [table.table_id for table in tables]

    def get_table_schema(self, dataset_id: str, table_id: str) -> list[dict]:
        """
        Get the schema of a specific table.

        Args:
            dataset_id: The BigQuery dataset ID.
            table_id: The table ID.

        Returns:
            A list of dictionaries containing field information.
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return [
            {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description,
            }
            for field in table.schema
        ]

    def execute_query(self, query: str) -> list[dict]:
        """
        Execute a SQL query and return results.

        Args:
            query: The SQL query to execute.

        Returns:
            A list of dictionaries representing the query results.
        """
        query_job = self.client.query(query)
        results = query_job.result()
        return [dict(row) for row in results]
