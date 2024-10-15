import streamlit as st
import openai
import os
import re
from datetime import datetime, timedelta

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_sitrep_info(content):
    sitrep_info = {}
    sitrep_info['title'] = re.search(r'SITREP TITLE:(.*?)$', content, re.MULTILINE).group(1).strip()
    sitrep_info['status'] = re.search(r'SITREP STATUS:(.*?)$', content, re.MULTILINE).group(1).strip()
    sitrep_info['organization'] = re.search(r'ORGANIZATION:(.*?)$', content, re.MULTILINE).group(1).strip()
    
    last_summary_match = re.search(r'LAST SUMMARY RESPONSE:(.*?)$', content, re.DOTALL)
    if last_summary_match:
        last_summary = last_summary_match.group(1).strip()
        name_query_match = re.match(r'(.*?),\s*(.*?)\s*GMT\s*(.*)', last_summary, re.DOTALL)
        if name_query_match:
            sitrep_info['name'] = name_query_match.group(1).strip()
            sitrep_info['query'] = name_query_match.group(3).strip()
        else:
            sitrep_info['name'] = 'Analyst'
            sitrep_info['query'] = last_summary
    else:
        sitrep_info['name'] = 'Analyst'
        sitrep_info['query'] = f"Please provide an analysis of the issue: {sitrep_info['title']}"

    return sitrep_info

def classify_query(query):
    classification_prompt = f"""
    Determine if the following query falls under Phase 1 or requires more specific analysis (Another Phase).

    Phase 1 queries are general in nature and typically ask for:
    - More information
    - Guidance
    - Best practices
    - General recommendations
    - Mitigation strategies for potential security threats
    - Steps to prevent issues from occurring again

    Phase 1 queries do not require analysis of specific customer logs or systems.

    Query: {query}

    Respond with only "Phase 1" or "Another Phase" without any explanation.
    """

    classification_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that classifies cybersecurity queries into appropriate phases based on their content and specificity."},
            {"role": "user", "content": classification_prompt}
        ]
    )

    return classification_response.choices[0].message['content'].strip()

def generate_response(sitrep_info):
    current_time = datetime.utcnow() + timedelta(hours=1)  # Assuming GMT+1
    response_time = current_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_info['title']}
    QUERY: {sitrep_info['query']}

    Provide a general response that focuses on:
    1. Best practices related to the issue
    2. General mitigation strategies
    3. Industry-standard recommendations
    4. Steps to improve overall cybersecurity hygiene

    Do not include any specific customer information or log analysis.

    Use the following format:
    {sitrep_info['name']}, {response_time}
    [General response following the guidelines above]

    Ensure the response is informative but not specific to any particular system or logs.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing responses to general sitrep queries, focusing on best practices and industry-standard recommendations."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        sitrep_info = extract_sitrep_info(content)
        phase = classify_query(sitrep_info['query'])
        if phase == "Phase 1":
            response = generate_response(sitrep_info)
            return sitrep_info['query'], response, phase
        else:
            return sitrep_info['query'], None, phase
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "Error in processing", None, "Error"

def main():
    st.title("Sitrep Processor for Phase 1 Queries")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the complete Sitrep content here:", height=300)
    
    if st.button("Process Sitrep"):
        if not content:
            st.error("Please provide the Sitrep content.")
        else:
            query, response, phase = process_sitrep(content)
            st.subheader("Identified Query")
            st.markdown(query)
            
            if phase == "Phase 1":
                st.subheader("Generated Response")
                st.markdown(response)
            else:
                st.warning("Another Phase")

if __name__ == "__main__":
    main()
