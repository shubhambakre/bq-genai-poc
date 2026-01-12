"""LLM Logic module for Text-to-SQL using LangChain and Google Gemini."""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class GenAIChain:
    """A class that sets up and manages the LangChain SQL chain with Gemini Pro."""

    def __init__(
        self,
        connection_uri: str,
        google_api_key: str | None = None,
        model_name: str = "gemini-pro",
        temperature: float = 0.0,
    ):
        """
        Initialize the GenAI chain for Text-to-SQL.

        Args:
            connection_uri: SQLAlchemy connection URI for BigQuery.
            google_api_key: Google API key for Gemini. If None, uses GOOGLE_API_KEY env var.
            model_name: The Gemini model to use.
            temperature: Model temperature for response generation.
        """
        self.connection_uri = connection_uri
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.temperature = temperature

        self._db: SQLDatabase | None = None
        self._llm: ChatGoogleGenerativeAI | None = None
        self._sql_chain = None
        self._answer_chain = None

    @property
    def db(self) -> SQLDatabase:
        """Get or create the SQLDatabase instance."""
        if self._db is None:
            self._db = SQLDatabase.from_uri(self.connection_uri)
        return self._db

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            if not self.google_api_key:
                raise ValueError(
                    "No Google API key provided. Set GOOGLE_API_KEY environment variable "
                    "or pass google_api_key to constructor."
                )
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                temperature=self.temperature,
                convert_system_message_to_human=True,
            )
        return self._llm

    @property
    def sql_chain(self):
        """Get or create the SQL query generation chain."""
        if self._sql_chain is None:
            self._sql_chain = create_sql_query_chain(self.llm, self.db)
        return self._sql_chain

    def get_table_info(self) -> str:
        """
        Get information about available tables in the database.

        Returns:
            A string containing table schemas and sample data.
        """
        return self.db.get_table_info()

    def get_table_names(self) -> list[str]:
        """
        Get the names of all available tables.

        Returns:
            A list of table names.
        """
        return self.db.get_usable_table_names()

    def generate_sql(self, question: str) -> str:
        """
        Generate a SQL query from a natural language question.

        Args:
            question: The natural language question.

        Returns:
            The generated SQL query.
        """
        raw_response = self.sql_chain.invoke({"question": question})
        sql_query = self._clean_sql_response(raw_response)
        return sql_query

    def _clean_sql_response(self, response: str) -> str:
        """
        Clean the LLM response to extract pure SQL.

        Args:
            response: The raw LLM response.

        Returns:
            Cleaned SQL query.
        """
        sql = response.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        if sql.lower().startswith("sqlquery:"):
            sql = sql[9:].strip()
        return sql

    def execute_sql(self, sql_query: str) -> str:
        """
        Execute a SQL query against the database.

        Args:
            sql_query: The SQL query to execute.

        Returns:
            The query results as a string.

        Raises:
            Exception: If the query execution fails.
        """
        return self.db.run(sql_query)

    def generate_answer(self, question: str, sql_query: str, sql_result: str) -> str:
        """
        Generate a natural language answer from the SQL results.

        Args:
            question: The original user question.
            sql_query: The SQL query that was executed.
            sql_result: The results from executing the SQL.

        Returns:
            A natural language answer.
        """
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, 
answer the user question in a clear and concise manner.

Question: {question}
SQL Query: {sql_query}
SQL Result: {sql_result}

Answer:"""
        )

        answer_chain = answer_prompt | self.llm | StrOutputParser()

        return answer_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "sql_result": sql_result,
        })

    def ask(self, question: str) -> dict:
        """
        Process a natural language question end-to-end.

        Args:
            question: The natural language question.

        Returns:
            A dictionary containing the SQL query, results, and natural language answer.

        Raises:
            Exception: If any step in the pipeline fails.
        """
        sql_query = self.generate_sql(question)
        sql_result = self.execute_sql(sql_query)
        answer = self.generate_answer(question, sql_query, sql_result)

        return {
            "question": question,
            "sql_query": sql_query,
            "sql_result": sql_result,
            "answer": answer,
        }
