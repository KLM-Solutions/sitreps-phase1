import streamlit as st
import openai
import re
import os

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_sitrep_info(content):
    sitrep_title = re.search(r"SITREP TITLE: (.+)", content)
    sitrep_status = re.search(r"SITREP STATUS: (.+)", content)
    organization = re.search(r"ORGANIZATION: (.+)", content)
    last_response = re.search(r"LAST SUMMARY RESPONSE:\s*([\s\S]+?)(?=\n\n|\Z)", content)
    
    return {
        "title": sitrep_title.group(1) if sitrep_title else "",
        "status": sitrep_status.group(1) if sitrep_status else "",
        "organization": organization.group(1) if organization else "",
        "last_response": last_response.group(1).strip() if last_response else ""
    }

def is_general_inquiry(query):
    general_keywords = ["how", "what", "best practice", "recommend", "mitigate", "prevent", "improve"]
    return any(keyword in query.lower() for keyword in general_keywords)

def generate_response(sitrep_info, query):
    if not is_general_inquiry(query):
        return "This query requires specific analysis. A Cybersecurity Analyst will review and respond shortly."

    try:
        prompt = f"""
        Based on the following sitrep information:
        Title: {sitrep_info['title']}
        Status: {sitrep_info['status']}
        Organization: {sitrep_info['organization']}
        Last Response: {sitrep_info['last_response']}

        Customer Query: {query}

        Provide a specific response addressing the customer's query. Focus on:
        1. Relevant mitigation strategies
        2. Best practices related to the sitrep title
        3. Specific recommendations for improving cybersecurity hygiene
        4. Steps to prevent similar issues in the future

        Ensure the response is tailored to the sitrep content and avoid generic advice.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant for a cybersecurity company. Provide specific, relevant responses to customer inquiries based on the given sitrep information."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except openai.error.AuthenticationError:
        return "Error: Invalid API key. Please check your OPENAI_API_KEY environment variable."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Sitrep Processor - Phase 1")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Slack message content here:", height=200)
    query = st.text_input("Enter the customer's query:")
    
    if st.button("Process Sitrep"):
        if not content or not query:
            st.error("Please provide both the Slack message content and the customer's query.")
        else:
            sitrep_info = extract_sitrep_info(content)
            
            st.subheader("Extracted Information")
            st.write(f"Title: {sitrep_info['title']}")
            st.write(f"Status: {sitrep_info['status']}")
            st.write(f"Organization: {sitrep_info['organization']}")
            st.write(f"Last Response: {sitrep_info['last_response']}")
            
            response = generate_response(sitrep_info, query)
            
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main()
