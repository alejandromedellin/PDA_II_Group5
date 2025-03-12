import requests
import pandas as pd
import logging

class PySimFin:
    PRICES_ENDPOINT = "https://backend.simfin.com/api/v3/companies/prices/compact"
    FIN_STATEMENTS_ENDPOINT = "https://backend.simfin.com/api/v3/companies/statements/compact"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "accept": "application/json",
            "Authorization": f"api-key {self.api_key}"
        }
        
        self.logger = logging.getLogger("PySimFin")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.logger.debug("PySimFin initialized with provided API key.")

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieves share price data for a given ticker and date range.
        Returns a DataFrame with columns from the API plus a 'ticker' column.
        """
        url = f"{self.PRICES_ENDPOINT}?ticker={ticker}&start={start}&end={end}"
        self.logger.info(f"Requesting share prices for {ticker} from {start} to {end}.")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        # Check if data exists and is not empty
        if not data or not data[0].get("data"):
            self.logger.error(f"Unexpected data structure in share prices response: {data}")
            return pd.DataFrame()  # Return an empty DataFrame
        columns = data[0]["columns"]
        rows = data[0]["data"]
        df = pd.DataFrame(rows, columns=columns)
        # Attach the ticker column since it's available in the parent JSON but not in the DataFrame.
        df["ticker"] = ticker
        return df

    def get_financial_statement(self, ticker: str, fyear: int, period: str, 
                                statements: str = "PL,BS,CF,DERIVED") -> dict:
        """
        Retrieves financial statements for the given ticker, fiscal year, and period.
        Returns a dictionary mapping each statement type to its DataFrame.
        """
        url = (f"{self.FIN_STATEMENTS_ENDPOINT}?ticker={ticker}&statements={statements}"
               f"&period={period}&fyear={fyear}")
        self.logger.info(f"Requesting financial statements for {ticker} for fiscal year {fyear} ({period}).")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        dfs = {}
        if data and isinstance(data, list) and "statements" in data[0]:
            for stmt in data[0]["statements"]:
                if "statement" in stmt and "columns" in stmt and "data" in stmt:
                    stmt_type = stmt["statement"]
                    columns = stmt["columns"]
                    rows = stmt["data"]
                    df = pd.DataFrame(rows, columns=columns)
                    # Optionally attach the ticker (if desired)
                    df["ticker"] = ticker
                    dfs[stmt_type] = df
                    self.logger.info(f"Financial statement '{stmt_type}' DataFrame created successfully.")
            return dfs
        else:
            self.logger.error(f"Unexpected data structure in financial statements response: {data}")
            raise ValueError(f"Unexpected data structure in financial statements response: {data}")
