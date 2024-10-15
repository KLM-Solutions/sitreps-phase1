import streamlit as st
import openai
import os
import json
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_json_string(json_string):
    return re.sub(r'^```json\n|```$', '', json_string, flags=re.MULTILINE).strip()

def process_sitrep(content):
    try:
        extraction_prompt = f"""
        Extract the following from the given content:
        1. SITREP TITLE
        2. CLIENT QUERY (Look for the most recent client comment or question)

        Content:
        {content}

        Provide the output in JSON format with keys: title, client_query
        The client_query should be a concise 1-2 sentence summary of the client's main concern or question.
        """

        extraction_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract key information concisely as JSON."},
                {"role": "user", "content": extraction_prompt}
            ]
        )
        
        cleaned_json = clean_json_string(extraction_response.choices[0].message['content'])
        extracted_info = json.loads(cleaned_json)

        response_prompt = f"""
        Title: {extracted_info['title']}
        Client Query: {extracted_info['client_query']}

        Provide a concise 2-3 line response addressing the client's query or concern.
        Focus on immediate next steps or brief recommendations.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cybersecurity assistant. Provide brief, focused responses."},
                {"role": "user", "content": response_prompt}
            ]
        )

        return extracted_info, response.choices[0].message['content']

    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def main():
    st.title("Sitrep Processor - Concise Output")
    
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
