import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
from langchain.schema import Document
from utils import load_json_data,get_columns, generate_sql, run_sql  # Make sure this function is defined in your utils.py
from langchain_google_vertexai import VertexAIEmbeddings
from opensearchpy import OpenSearch
from langchain_community.vectorstores import OpenSearchVectorSearch
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import vertexai
import json

PROJECT_ID = "festive-nova-433913-u5"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
MODEL = "textembedding-gecko@latest"
FILE_PATH = "customer_shopping_data_columns.jsonl"
JQ_SCHEMA = '.'  # Make sure this jq schema is correctly formatted
VM_IP = 'localhost'  # VM's public IP address
PORT = '9200'
table = 'projectdb.customer_shopping_data'
# Initialize the embedding model
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
vertexai.init(project=PROJECT_ID, location=LOCATION)
generation_config = {
"max_output_tokens": 8192,  
"temperature": 0.5,  
"top_p": 0.95,
}
multimodal_model = GenerativeModel(
    "gemini-pro", generation_config=generation_config
)
embedding_model = VertexAIEmbeddings(model_name=MODEL, project=PROJECT_ID)

def create_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="manish15",
            database="projectdb"
        )
        if connection.is_connected():
            print("Successfully connected to the database.")
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def main():
    st.set_page_config(page_title="Statspeak", layout="wide")
    st.header("Statspeak")

    # Load data from JSONL file using custom load_json_data function
    documents = load_json_data(FILE_PATH, JQ_SCHEMA)
    docsearch = OpenSearchVectorSearch.from_documents(
            documents,
            embedding_model,
            opensearch_url=f'https://{VM_IP}:{PORT}',
            index_name="project",
            engine="faiss",
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False,
        )
    upload = st.file_uploader("Upload your data...")
    query = st.text_input("Enter your Query...")
    if query:
        connection = create_mysql_connection()
        columns = get_columns(query,docsearch)
        # st.write(columns)
        with open('sample_sql_query.json', 'r') as f:
            question_sql_list = json.load(f)
        sql = generate_sql(query,columns, question_sql_list, multimodal_model,table )
        st.write(sql)
        df = run_sql(sql,connection)
        # st.write(df)
        # df_string = df.to_string()
        input_for_model = f"""
        Task:
        Analyze the data to answer the user's query.
        provide answer from data only.
        Data:
        {df}
        """
        response = multimodal_model.generate_content([query,input_for_model])
        st.write(response.text)
if __name__ == "__main__":
    main()
