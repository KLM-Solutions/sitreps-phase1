import streamlit as st
import openai
import os
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_query(content):
    # Extract the last response or comment from the content
    last_response = re.findall(r'(?:^|\n).*?(?:GMT|UTC).*?\n(.*)', content, re.DOTALL)
    if last_response:
        return last_response[-1].strip()
    return None

def is_general_query(query):
    general_keywords = [
        "how", "what", "best practice", "recommend", "mitigate", "prevent", 
        "improve", "secure", "protect", "guideline", "strategy"
    ]
    return any(keyword in query.lower() for keyword in general_keywords)

def generate_response(query):
    prompt = f"""
    Provide a concise response (2-3 sentences) to the following cybersecurity query:
    "{query}"
    
    Focus on:
    1. Relevant mitigation strategies
    2. Best practices for security
    3. General recommendations for improving cybersecurity hygiene
    4. Steps to prevent similar issues in the future

    Ensure the response is based on industry-standard guidelines and practices.
    Do not include any customer-specific information or log analysis.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing concise, general advice based on industry standards."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def process_sitrep(content):
    query = extract_query(content)
    if query and is_general_query(query):
        return query, generate_response(query)
    return query, "This query requires specific analysis or technical input. A Cybersecurity Analyst will review and respond shortly."

def main():
    st.title("Cybersecurity Inquiry Responder - Phase 1")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Sitrep content here:", height=200)
    
    if st.button("Process Inquiry"):
        if not content:
            st.error("Please provide the Sitrep content.")
        else:
            query, response = process_sitrep(content)
            
            if query:
                st.subheader("Extracted Query")
                st.write(query)
            
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main()
