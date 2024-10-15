import streamlit as st
import openai
import os
import json
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_json_string(json_string):
    return re.sub(r'^```json\n|```$', '', json_string, flags=re.MULTILINE).strip()

def extract_and_verify(content):
    extraction_prompt = f"""
    Carefully analyze the following content and extract:
    1. SITREP TITLE
    2. CLIENT QUERY (Look for any client question or concern throughout the entire content, including the title)

    Content:
    {content}

    Provide the output in JSON format with keys: title, client_query
    The client_query should be the most relevant client question or concern, summarized in 1-2 sentences.
    If no explicit query is found, infer the main concern from the context.
    """

    extraction_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise extractor of key information. Analyze thoroughly and provide accurate results."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    
    cleaned_json = clean_json_string(extraction_response.choices[0].message['content'])
    extracted_info = json.loads(cleaned_json)

    # Double-check mechanism
    verification_prompt = f"""
    Re-analyze the following content and verify if the extracted information is accurate:

    Content:
    {content}

    Extracted Information:
    Title: {extracted_info['title']}
    Client Query: {extracted_info['client_query']}

    If any information is incorrect or incomplete, provide the corrected version.
    If everything is correct, simply respond with "Verified".

    Provide your response in JSON format with keys: verified, title, client_query
    """

    verification_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a meticulous verifier of extracted information. Double-check thoroughly and provide accurate results."},
            {"role": "user", "content": verification_prompt}
        ]
    )

    verified_info = json.loads(clean_json_string(verification_response.choices[0].message['content']))
    
    if verified_info['verified'] != "Verified":
        extracted_info = {
            'title': verified_info['title'],
            'client_query': verified_info['client_query']
        }

    return extracted_info

def generate_response(extracted_info):
    query = extracted_info['client_query'].lower()
    
    # Check if the query is general in nature
    general_keywords = ["how", "what", "best practice", "recommend", "mitigate", "prevent", "improve"]
    is_general_query = any(keyword in query for keyword in general_keywords)
    
    if not is_general_query:
        return "This query requires specific analysis. A Cybersecurity Analyst will review and respond shortly."

    response_prompt = f"""
    Provide a concise response to the following cybersecurity query:
    "{extracted_info['client_query']}"

    Focus on one or more of the following aspects, as relevant to the query:
    1. Mitigation strategies for potential security threats
    2. Best practices for securing a network
    3. General recommendations for improving cybersecurity hygiene
    4. Steps to prevent the issue from occurring again

    Base your response on industry-standard guidelines and practices. 
    Do not include any customer-specific information or recommendations that would require analysis of specific logs or systems.
    Limit your response to 2-3 sentences.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity assistant providing general advice based on industry standards. Avoid customer-specific details."},
            {"role": "user", "content": response_prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        extracted_info = extract_and_verify(content)
        response = generate_response(extracted_info)
        return extracted_info, response
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def main():
    st.title("Sitrep Processor - Thorough Analysis")
    
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
