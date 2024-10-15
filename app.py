import streamlit as st
import openai
import os
import json

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        """

        extraction_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that extracts structured information from text."},
                {"role": "user", "content": extraction_prompt}
            ]
        )
        
        extracted_info = json.loads(extraction_response.choices[0].message['content'])

        # Second LLM call to generate response
        response_prompt = f"""
        Based on the following extracted information:
        Title: {extracted_info['title']}
        Status: {extracted_info['status']}
        Organization: {extracted_info['organization']}
        Last Response: {extracted_info['last_response']}
        Client Query: {extracted_info['client_query']}

        {"Provide a specific response addressing the client's query. Focus on relevant mitigation strategies, best practices, and recommendations related to the SITREP title. If the query is about NTP (Network Time Protocol), provide specific best practices for NTP security." if extracted_info['is_general_inquiry'] else "This query requires specific analysis. Explain that a Cybersecurity Analyst will review and respond shortly."}

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
                st.write(f"Title: {extracted_info['title']}")
                st.write(f"Status: {extracted_info['status']}")
                st.write(f"Organization: {extracted_info['organization']}")
                st.write(f"Last Response: {extracted_info['last_response']}")
                st.write(f"Client Query: {extracted_info['client_query']}")
                st.write(f"Is General Inquiry: {'Yes' if extracted_info['is_general_inquiry'] else 'No'}")
            
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main()
