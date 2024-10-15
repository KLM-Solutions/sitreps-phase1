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

def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an AI assistant for a cybersecurity company. Your task is to provide general responses to customer inquiries about cybersecurity best practices, recommendations, and mitigation strategies. Focus on industry-standard guidelines and avoid customer-specific details. If a query requires specific log analysis or technical details, suggest that a Cybersecurity Analyst should review it."""},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except openai.error.AuthenticationError:
        return "Error: Invalid API key. Please check your OPENAI_API_KEY environment variable."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Sitrep Processor")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Slack message content here:", height=200)
    
    if st.button("Process Sitrep"):
        if not content:
            st.error("Please paste the Slack message content before processing.")
        else:
            sitrep_info = extract_sitrep_info(content)
            
            st.subheader("Extracted Information")
            st.write(f"Title: {sitrep_info['title']}")
            st.write(f"Status: {sitrep_info['status']}")
            st.write(f"Organization: {sitrep_info['organization']}")
            st.write(f"Last Response: {sitrep_info['last_response']}")
            
            prompt = f"""
            Based on the following sitrep information:
            Title: {sitrep_info['title']}
            Status: {sitrep_info['status']}
            Organization: {sitrep_info['organization']}
            Last Response: {sitrep_info['last_response']}

            Please provide a general response with cybersecurity best practices, recommendations, or mitigation strategies related to this sitrep. Focus on industry-standard guidelines and avoid customer-specific details.
            """
            
            response = generate_response(prompt)
            
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main()
