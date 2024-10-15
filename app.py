import streamlit as st
import openai
import os
import json
import re
from datetime import datetime, timedelta

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_query(content):
    # First, try to extract the query directly from LAST SUMMARY RESPONSE
    last_summary_match = re.search(r'LAST SUMMARY RESPONSE:(.*?)$', content, re.DOTALL)
    if last_summary_match:
        last_summary = last_summary_match.group(1).strip()
        name_query_match = re.match(r'(.*?),\s*(.*?)\s*GMT\s*(.*)', last_summary, re.DOTALL)
        if name_query_match:
            name = name_query_match.group(1).strip()
            query = name_query_match.group(3).strip()
            return query, name

    # If direct extraction fails, use LLM to infer the query
    extraction_prompt = f"""
    Analyze the following sitrep content and extract:
    1. The most relevant user query or request
    2. The name of the person making the query (if available)

    Content:
    {content}

    Provide the output as a JSON object with keys: "query" and "name".
    If there's no clear query, infer one based on the context of the sitrep.
    """

    extraction_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with extracting the most relevant user query from sitrep content."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    
    try:
        extracted_info = json.loads(extraction_response.choices[0].message['content'])
        return extracted_info['query'], extracted_info['name']
    except json.JSONDecodeError:
        return "What are the key concerns in this sitrep?", None

def generate_response(query, sitrep_title, name):
    current_time = datetime.utcnow() + timedelta(hours=1)  # Assuming GMT+1
    response_time = current_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed response following this structure:
    1. Address the person by name (if provided) and acknowledge their query.
    2. Provide specific information about the alert mentioned in the SITREP TITLE.
    3. Explain the implications of the observed behavior.
    4. Suggest actionable steps for investigation or resolution.
    5. If applicable, provide information about thresholds or statistics related to the issue.
    6. Offer guidance on interpreting the information.
    7. Ask for any necessary confirmations or further information.

    Use the following format:
    {name if name else "Analyst"}, {response_time}
    [Detailed response following the structure above]

    Do not include any closing remarks, "Best regards," signatures, or cybersecurity team mentions at the end.
    Ensure the response is comprehensive, tailored to the specific sitrep context, and provides valuable insights and recommendations.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed, contextual responses to sitrep queries. Your responses should be comprehensive and tailored to the specific situation, without any closing remarks or signatures."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        sitrep_title_match = re.search(r'SITREP TITLE:(.*?)$', content, re.MULTILINE)
        sitrep_title = sitrep_title_match.group(1).strip() if sitrep_title_match else "Unknown Title"
        
        query, name = extract_query(content)
        if query:
            response = generate_response(query, sitrep_title, name)
            return query, response
        else:
            return "No specific query identified. Analyzing overall sitrep content.", generate_response("Provide an analysis of this sitrep", sitrep_title, "Analyst")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "Error in processing", "Unable to generate a response due to an error. Please check the sitrep content and try again."

def main():
    st.title("Sitrep Processor")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Sitrep content here:", height=300)
    
    if st.button("Process Sitrep"):
        if not content:
            st.error("Please provide the Sitrep content.")
        else:
            query, response = process_sitrep(content)
            st.subheader("Identified Query or Context")
            st.markdown(query)
            st.subheader("Generated Response")
            st.markdown(response)

if __name__ == "__main__":
    main()
