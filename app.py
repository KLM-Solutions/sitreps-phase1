import streamlit as st
import openai
import os
import json
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_json_string(json_string):
    # Remove ```json and ``` if present
    json_string = re.sub(r'^```json\n|```$', '', json_string, flags=re.MULTILINE)
    # Remove any leading/trailing whitespace
    return json_string.strip()

def process_sitrep(content):
    try:
        # First LLM call to extract information and identify query
        extraction_prompt = f"""
        Extract the following information from the given content:
        1. SITREP TITLE
        2. SITREP STATUS
        3. ORGANIZATION
        4. LAST SUMMARY RESPONSE
        5. CLIENT QUERY

        Also, determine if the CLIENT QUERY is a general inquiry or requires specific analysis.

        Content:
        {content}

        Provide the output in JSON format with the following keys:
        title, status, organization, last_response, client_query, is_general_inquiry

        For is_general_inquiry, use true if it's a general query about best practices, recommendations, or mitigation strategies, and false if it requires specific log analysis or technical details.

        If there's no clear client query, set client_query to null and is_general_inquiry to false.

        Ensure your entire response is a valid JSON object.
        """

        extraction_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that extracts structured information from text and outputs it as valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ]
        )
        
        extraction_content = extraction_response.choices[0].message['content']
        st.text("Raw LLM extraction output:")
        st.code(extraction_content, language="json")
        
        # Clean the JSON string
        cleaned_json = clean_json_string(extraction_content)
        st.text("Cleaned JSON:")
        st.code(cleaned_json, language="json")

        try:
            extracted_info = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            st.error(f"JSON Decode Error: {str(e)}")
            return None, "Error: Unable to parse JSON response from LLM."

        # Second LLM call to generate response
        response_prompt = f"""
        Based on the following extracted information:
        Title: {extracted_info['title']}
        Status: {extracted_info['status']}
        Organization: {extracted_info['organization']}
        Last Response: {extracted_info['last_response']}
        Client Query: {extracted_info['client_query']}

        {'Provide a specific response addressing the client\'s query. Focus on relevant mitigation strategies, best practices, and recommendations related to the SITREP title.' if extracted_info['is_general_inquiry'] else 'This SITREP requires review. Explain that a Cybersecurity Analyst will review the anomalous internet traffic sessions and respond shortly with detailed analysis and recommendations.'}

        Ensure the response is tailored to the SITREP content and avoid generic advice.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant for a cybersecurity company. Provide specific, relevant responses to client inquiries based on the given SITREP information."},
                {"role": "user", "content": response_prompt}
            ]
        )

        return extracted_info, response.choices[0].message['content']

    except openai.error.AuthenticationError:
        return None, "Error: Invalid API key. Please check your OPENAI_API_KEY environment variable."
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
                for key, value in extracted_info.items():
                    st.write(f"{key.capitalize()}: {value}")
            
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main()
