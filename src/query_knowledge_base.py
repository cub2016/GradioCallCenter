from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage

import re
import os
import sqlite3

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from constants import DB_PATH

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.2)


# === Query Interface ===
def query_knowledge_base(user_question):

    context = query_from_user_input(user_question)
    context = str(', '.join(['{}']*len(context)).format(*context))
    print("context is "+ context)
    template = """
        You are an assistant that answers questions based on the context provided.
        If you don't know the answer, say you don't know. 
        keep the answer concise. 
        Context:
        {context}
    
        Question:
        {question}
    
        Answer in a clear and concise manner. And if there is a single file name
        for the response, include it in the response.
    """
    prompt = PromptTemplate.from_template(template)

    filled_prompt = prompt.format(context=context, question=user_question)

    response = llm.invoke([HumanMessage(content=filled_prompt)])

    # Regular expression pattern to match .wav or .mp3 file names
    pattern = r'([a-zA-Z0-9_\\]+\.wav|[a-zA-Z0-9_\\]+\.mp3)'

    # Search for the pattern in the input string
    match = re.search(pattern, response.content)

    # Check if a match was found and extract the file name
    file_name_base=""
    if match:
        file_name = match.group(0)
        index = file_name.rfind(os.sep, 0, len(file_name))
        if file_name.rfind(os.sep, 0) > 0:
            file_name_base = file_name[file_name.rfind(os.sep) + 1:]
    else:
        print("No file name found.")
        file_name=""

    transcript = ""
    sentiment_analysis = ""
    summary = ""
    score = ""

    if file_name!="":
        [id, transcript, summary, sentiment_analysis, score, file] = query_db_by_file_name(file_name)
    print("=== Answer ===")
    print(response.content)
    return response.content, transcript, summary, sentiment_analysis, score, file_name

def generate_sql_from_question(user_query):
    template = f"""
    You are a SQL assistant. Given a user question and the schema:

    Table: transcript_data
    Columns:
        - id (INTEGER)
        - transcript (TEXT)
        - summary_output (TEXT)
        - sentiment_output (TEXT)
        - sentiment_score (REAL)
        - file_name (TEXT)
  
     Convert this user query to a valid SQLite SELECT query.  If the answer is 
     based upon a single record, include the whole record in the expected reply 
     from the query.

    User query: "{user_query}"
    SQL:
    """

    prompt = PromptTemplate.from_template(template)

    filled_prompt = prompt.format(user_query=user_query)

    query = llm.invoke([HumanMessage(content=filled_prompt)])
    return query.content


def query_from_user_input(user_query):
    sql = generate_sql_from_question(user_query)
    print(f"Executing SQL: {sql}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results

def query_db_by_file_name(file_name):
    sql = f"""SELECT * 
        FROM transcript_data where file_name LIKE '%{file_name}%'"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results[0]

