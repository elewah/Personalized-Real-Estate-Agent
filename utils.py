import os
import ast
import pandas as pd
import numpy as np
import lancedb
from lancedb.pydantic import vector, LanceModel
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Model for LanceDB ---
class RealEstateListing(LanceModel):
    Property_ID: str
    Title: str
    Address: str
    Price_CAD: float
    Bedrooms: int
    Bathrooms: int
    Area_sqft: int
    Type: str
    Status: str
    Description: str
    Image_URL: str
    Agent_Name: str
    Agent_Phone: str
    Agent_Email: str
    Preferences_Embeddings: vector(384)
    Natural_Description_Embeddings: vector(384)

# --- Questionnaire Fields ---
fields = [
    {"key": "property_type", "question": "What type of property are you looking for?", "type": str},
    {"key": "city", "question": "Which city or area do you prefer?", "type": str},
    {"key": "budget", "question": "What is your budget in CAD?", "type": float},
    {"key": "bedrooms", "question": "How many bedrooms do you need?", "type": int},
    {"key": "bathrooms", "question": "How many bathrooms do you need?", "type": int},
    {"key": "area", "question": "What is the minimum area (in square feet)?", "type": float},
    {"key": "status", "question": "Do you want a property that is for sale or rent?", "type": str},
    {"key": "features", "question": "Do you have any specific features or descriptions you're looking for (e.g., modern kitchen, city view, private dock)?", "type": str},
]

# --- Embedding Generation ---
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def generate_embeddings(input_data):
    return model.encode(input_data)

def parse_embedding(x):
    if isinstance(x, str):
        if x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except:
                return [float(val.strip()) for val in x.strip('[]').split() if val.strip()]
        else:
            return [float(val.strip()) for val in x.split() if val.strip()]
    return x

# --- LanceDB Setup ---
def load_data_and_setup_lancedb():
    df = pd.read_csv("real_estate_with_embeddings.csv")
    df = df.astype({
        'Property_ID': 'str', 'Title': 'str', 'Address': 'str', 'Price_CAD': 'float',
        'Bedrooms': 'int', 'Bathrooms': 'int', 'Area_sqft': 'int', 'Type': 'str',
        'Status': 'str', 'Description': 'str', 'Image_URL': 'str', 'Agent_Name': 'str',
        'Agent_Phone': 'str', 'Agent_Email': 'str'
    })

    df['Preferences_Embeddings'] = df['Preferences_Embeddings'].apply(parse_embedding)
    df['Natural_Description_Embeddings'] = df['Natural_Description_Embeddings'].apply(parse_embedding)

    db = lancedb.connect("~/.lancedb")
    table_name = "RealEstateListing"
    try:
        db.drop_table(table_name)
    except Exception:
        pass
    table = db.create_table(table_name, schema=RealEstateListing)
    table.add(df.to_dict(orient='records'))
    return table

# --- Recommendation Functions ---
def get_recommendations_from_preferences(table, preferences: str, top_k: int = 5):
    query_vector = generate_embeddings(preferences)
    results = table.search(query_vector, vector_column_name="Preferences_Embeddings").limit(top_k).to_pydantic(RealEstateListing)
    return [(c.Property_ID, c.Title, c.Address, c.Price_CAD, c.Description, c.Image_URL) for c in results]

def select_best_listing_and_describe(preferences: str, listings):
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.1,
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    listing_descriptions = "\n".join(
        [f"{i+1}. Title: {title}, Address: {address}, Price: {price} CAD"
         for i, (pid, title, address, price, _, _) in enumerate(listings)]
    )

    prompt = f"""
You are a helpful real estate assistant.

The user's preferences are:
{preferences}

Here are the top recommended listings:
{listing_descriptions}

Based on the user's preferences, select the ONE listing that best matches their needs.
Then write a personalized and factual description of this property, highlighting why it fits the user's needs.
Do NOT change or add any factual information.
Respond in this format:
Here is the best listing for you:
- Title: <title>
- Description: <personalized description>
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# --- LLM for asking questions ---
llm_chat = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.1,
    openai_api_base=os.environ.get("OPENAI_API_BASE"),
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant helping collect user preferences for buying a property. Ask the user the following question in your own words."),
    ("human", "{question}")
])

def get_llm_question(question):
    messages = chat_template.format_messages(question=question)
    response = llm_chat.invoke(messages)
    return response.content 