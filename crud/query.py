import sqlite3
import pandas as pd

def run(cursor, query):
    '''
    Return pandas dataframe
    '''
    cursor.execute(query)
    # Fetch all results
    rows = cursor.fetchall()

    # Get column names from cursor.description
    columns = [description[0] for description in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=columns)

    return df

if __name__ == '__main__':
    with sqlite3.connect("database.sqlite") as conn:
        cursor = conn.cursor()
        query = '''
            SELECT DISTINCT a.concept_id, c.name, sg.group_abbr
            FROM Annotations AS a
            INNER JOIN Concepts AS c
            ON a.concept_id = c.concept_id
            INNER JOIN Semantic_Groups AS sg
            ON c.semantic_type = sg.semantic_type
        '''
        df = run(cursor, query)
        df.to_csv("prepare/semantic_group_map.csv", index=None)
