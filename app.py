import streamlit as st
import openai
import os
import json
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_json_string(json_string):
    return re.sub(r'^```json\n|```$', '', json_string, flags=re.MULTILINE).strip()

def extract_query(content):
    extraction_prompt = f"""
    Analyze the following content and extract:
    1. SITREP TITLE
    2. CLIENT QUERY (Look for any client question or concern in the SITREP TITLE or the most recent response)

    Content:
    {content}

    Provide the output in JSON format with keys: title, client_query
    The client_query should be the most relevant client question or concern, summarized in 1-2 sentences.
    If no explicit query is found, infer the main concern from the context.
    """

    extraction_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract key information accurately from the given content."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    
    cleaned_json = clean_json_string(extraction_response.choices[0].message['content'])
    return json.loads(cleaned_json)

def is_general_query(query):
    general_keywords = ["how", "what", "best practice", "recommend", "mitigate", "prevent", "improve"]
    return any(keyword in query.lower() for keyword in general_keywords)

def generate_response(query):
    if not is_general_query(query):
        return "This query requires specific analysis. A Cybersecurity Analyst will review and respond shortly."

    response_prompt = f"""
    Provide a concise 2-3 line response to the following cybersecurity question:
    "{query}"

    Focus on providing practical, general advice based on industry-standard guidelines and best practices. 
    Do not include any customer-specific information or recommendations that would require analysis of specific logs or systems.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity assistant providing brief, focused responses based on industry standards. Limit your response to 2-3 lines."},
            {"role": "user", "content": response_prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        extracted_info = extract_query(content)
        response = generate_response(extracted_info['client_query'])
        return extracted_info, response
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def main():
    st.title("Sitrep Processor - Phase 1")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Slack message content here:", height=200)
    
    if st.button("Process Sitrep"):
        if not content:
            st.error("Please provide the Slack message content.")
        else:
            extracted_info, response = process_sitrep(content)
            
            if extracted_info:
                st.subheader("Extracted Information")
                st.write(f"Title: {extracted_info['title']}")
                st.write(f"Client Query: {extracted_info['client_query']}")
            
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main()
