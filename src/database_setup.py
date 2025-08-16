import sqlite3

def setup_database():
    """
    Creates the SQLite database and the optimization_results table.
    This function only needs to be run once.
    """
    conn = sqlite3.connect('strategy_parameters.db')
    cursor = conn.cursor()

    # Create table to store the results
    # The combination of ticker and strategy_name will be our unique key
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_results (
            ticker TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            parameters TEXT NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            PRIMARY KEY (ticker, strategy_name)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database 'strategy_parameters.db' and table 'optimization_results' created successfully.")

if __name__ == '__main__':
    setup_database()
