# Personalized Real Estate Agent

## 1. Project Introduction & Description

This project showcases a sophisticated, AI-powered real estate agent designed to help users find their ideal properties. The application leverages Large Language Models (LLMs) and semantic search to deliver a personalized and interactive experience.

The core functionalities include:
-   **Synthetic Data Generation:** Creating a realistic and diverse dataset of real estate listings using an LLM.
-   **Vector-Based Semantic Search:** Finding properties that match a user's preferences not just by keywords, but by semantic meaning.
-   **Retrieval-Augmented Generation (RAG):** Providing users with personalized, context-aware property descriptions based on their unique needs, retrieved from our database.

The project is structured into two main parts, demonstrated through Jupyter notebooks:
-   `data_preparation_vector_database_creation.ipynb`: Focuses on generating the data and setting up the vector database.
-   `augmented_response_generation.ipynb`: Implements the end-to-end user-facing application, from conversational preference gathering to generating augmented, personalized responses.

## 2. Technologies Used

### LLM Concepts
-   **Synthetic Data Generation:** An LLM is prompted with specific instructions to generate structured, realistic datasets of Canadian real estate listings.
-   **Embeddings:** The `sentence-transformers` library is used to convert textual data (listings and user preferences) into high-dimensional vectors. This allows for the numerical representation of semantic meaning, which is crucial for similarity search.
-   **Vector Database:** `LanceDB` is used as a lightweight, serverless vector database for efficient storage, indexing, and retrieval of property embeddings.
-   **Retrieval-Augmented Generation (RAG):** The core of the personalization engine. The system first *retrieves* relevant listings from the LanceDB using semantic search based on user preferences. Then, it *augments* this retrieved information by feeding it into an LLM to generate a rich, personalized, and human-like description.
-   **Conversational AI:** An LLM is used to guide the user through a series of questions in a natural, conversational manner to collect their property preferences.

### Python Packages
-   `langchain`: The primary framework for building the LLM-powered application, orchestrating prompts, models, and chains.
-   `openai`: The official Python client for interacting with OpenAI's LLM APIs (e.g., GPT-4o).
-   `lancedb`: The vector database for storing and searching listings.
-   `sentence-transformers`: For generating high-quality text embeddings locally.
-   `pandas`: For robust data manipulation and handling of the real estate listings.
-   `python-dotenv`: For managing environment variables securely.
-   `numpy`: For numerical operations, especially with embeddings.
-   `ipykernel`: To run the Jupyter notebooks.

## 3. Fulfillment of Requirements

This project successfully meets all the specified assignment criteria.

### Synthetic Data Generation
-   **Requirement:** Generate at least 10 diverse and realistic real estate listings.
-   **Implementation (`data_preparation_vector_database_creation.ipynb`):** An LLM (`gpt-4o`) is prompted to generate 10 unique real estate listings across Canada. The prompt explicitly defines the required fields (e.g., Price, Bedrooms, Type, Status), ensuring the data is structured and realistic for the Canadian market.

### Semantic Search
-   **Requirement:** Create a vector database and store listing embeddings.
-   **Implementation (`data_preparation_vector_database_creation.ipynb`):**
    1.  The `sentence-transformers` model `paraphrase-MiniLM-L6-v2` is used to generate two distinct vector embeddings for each listing: one from structured key-value data (`Preferences_Embeddings`) and another from the full natural description (`Natural_Description_Embeddings`).
    2.  A `LanceDB` table is created with a schema that accommodates all listing fields and both vector embeddings.
    3.  The generated listings and their corresponding embeddings are successfully stored in the LanceDB table.

-   **Requirement:** Semantically search listings based on buyer preferences.
-   **Implementation (`augmented_response_generation.ipynb`):** The `get_recommendations_from_preferences` function takes a string of user preferences, generates a query embedding from it, and performs a semantic similarity search against the `Preferences_Embeddings` vector column in LanceDB to find the most relevant listings.

### Augmented Response Generation
-   **Requirement:** Logic for searching and augmenting listing descriptions based on buyer preferences.
-   **Implementation (`augmented_response_generation.ipynb`):** The project demonstrates a clear and logical RAG flow:
    1.  **Collect:** User preferences are collected via a dynamic, LLM-driven conversational questionnaire.
    2.  **Search (Retrieve):** The collected preferences are used to perform a semantic search on the vector database, retrieving the top K most relevant listings.
    3.  **Augment (Generate):** The user's original preferences and the retrieved listings are passed as context to a second LLM call. This LLM is tasked with selecting the single best match and generating a personalized description.

-   **Requirement:** Use an LLM for generating personalized descriptions.
-   **Implementation (`augmented_response_generation.ipynb`):** The `select_best_listing_and_describe` function uses an LLM to analyze the search results in the context of the user's specific needs. It then crafts a unique, appealing, and personalized summary for the best-matched property, highlighting features relevant to the user's query without fabricating factual information.

## 4. Environment Setup

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/elewah/Personalized-Real-Estate-Agent.git
    cd Personalized-Real-Estate-Agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    You need to provide your OpenAI API credentials. Create a file named `.env` in the root of the project directory and add the following content.
    ```
    OPENAI_API_KEY="your_openai_api_key"
    OPENAI_API_BASE="your_openai_api_base_url" # Only if using a custom base URL
    ```
    Replace `"your_openai_api_key"` with your actual key.

## 5. Testing Demos

The project's functionality can be demonstrated in two ways: through Jupyter notebooks for a step-by-step walkthrough, or via an interactive Streamlit web application.

First, regardless of the demo you choose, you must prepare the data and vector database by running the `data_preparation_vector_database_creation.ipynb` notebook.

1.  **Start Jupyter Notebook**
    Ensure your virtual environment is activated and run:
    ```bash
    jupyter notebook
    ```

2.  **Run the Data Preparation Notebook**
    -   Open and run **`data_preparation_vector_database_creation.ipynb`** cell by cell. This handles the initial data generation and populates the LanceDB vector database. This only needs to be run once to set up the data.

Once the data is prepared, you can choose one of the following methods to run the demo.

### Option 1: Jupyter Notebook Demo

-   Open and run **`augmented_response_generation.ipynb`**. This is the main demonstration notebook. The final cells will launch an interactive questionnaire where you can input your housing preferences. The notebook will then process your preferences, perform the search, and display the final personalized recommendation.

### Option 2: Streamlit Web Application Demo

For a more user-friendly and interactive experience, you can run the Streamlit application.

-  **Run the Streamlit App**
    In your terminal (with the virtual environment activated), run the following command:
    ```bash
    streamlit run app.py
    ```
    This will launch the web application in your browser, where you can interact with the real estate recommender chatbot. 
