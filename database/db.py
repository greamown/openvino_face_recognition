import sqlite3, logging, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/"common"))
from common import read_json

DB_PATH  = read_json("./database/db_config.json")["DB_PATH"]
INIT_DATA  =  {
                "features": "name TEXT, \
                            feature TEXT, \
                            create_time TEXT"
             }

# Connect
def connect_db():
    try:
        connector = sqlite3.connect(DB_PATH)
    except:
        logging.error("Can't connect to the database") 
    return connector

# Execute db
def execute_db(command:str, update:bool):
    connector = connect_db()
    db_data = None
    try:
        cur = connector.cursor()
        cur.execute(command)
        if not (update):
            db_data = cur.fetchall()
        cur.close()
        if (update):
            connector.commit()
        return db_data

    except (Exception, sqlite3.DatabaseError) as error:
        logging.error(error)
        return ["error", error]

    finally:
        if connector is not None:
            connector.close()

# Initial DB
def init_db():
    connector = connect_db()
    try:
        logging.info("Create table in the database...")
        cur = connector.cursor()
        for key in INIT_DATA:
            if key != "iteration":
                cur.execute(create_table_cmd(key,INIT_DATA[key]))

        cur.close()
        connector.commit()
    except (Exception, sqlite3.DatabaseError) as error:
        logging.error(error)
        return ["error", error]
    finally:
        if connector is not None:
            connector.close()

# Create table
def create_table_cmd(table_name:str, content:str):
    """ Create tables in the database"""
    commands =  """
                CREATE TABLE IF NOT EXISTS {} (
                {}
                )
                """.format(table_name, content)
    return commands

# Add new data in table
def insert_table_cmd(table_name:str, keys:str, values:str):
    """ Insert data in tables of the database"""
    commands =  """
                INSERT INTO {} ({})
                VALUES ({});
                """.format(table_name, keys, values)
    info_db = execute_db(commands, True)
    return info_db

# Delete data from table
def delete_data_table_cmd(table_name:str, content:str):
    """ Delete data from tables of the database"""
    commands =  """
                DELETE FROM {} WHERE {};
                """.format(table_name, content)
    info_db = execute_db(commands, True)
    return info_db

# Update data from table
def update_data_table_cmd(table_name:str, values:str, select:str):
    """ Update data in table """
    commands =  """
                UPDATE {} SET {} WHERE {};
                """.format(table_name, values, select)
    info_db = execute_db(commands, True)
    return info_db