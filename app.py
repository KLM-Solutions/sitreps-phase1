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

def generate_response(sitrep_info, full_content):
    current_time = datetime.utcnow() + timedelta(hours=1)  # Assuming GMT+1
    response_time = current_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

    prompt = f"""
    Analyze the following sitrep content and provide a balanced, informative response:

    SITREP TITLE: {sitrep_info['title']}
    QUERY: {sitrep_info['query']}

    Generate a response that covers the following points concisely:
    1. Acknowledge the query and provide context about the alert or issue.
    2. Explain the implications or potential risks of the observed behavior.
    3. Suggest 2-3 actionable steps for investigation or resolution.
    4. If relevant, briefly mention any thresholds or statistics related to the issue.
    5. Offer a concise guidance on interpreting the information.
    6. If necessary, ask for any critical additional information needed.

    Use the following format:
    {sitrep_info['name']}, {response_time}
    [Balanced response covering the points above]

    Aim for a response of about 150-200 words. Focus on providing valuable insights and practical recommendations without excessive detail.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing informative yet concise responses to sitrep queries. Your responses should cover key points while remaining focused and practical."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        sitrep_info = extract_sitrep_info(content)
        response = generate_response(sitrep_info, content)
        return sitrep_info['query'], response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "Error in processing", "Unable to generate a response due to an error. Please check the sitrep content and try again."

def main():
    st.title("Balanced Sitrep Processor")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the complete Sitrep content here:", height=300)
    
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
