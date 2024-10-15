import streamlit as st
import openai
import os
import re
from datetime import datetime, timedelta

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_query_and_name(content):
    match = re.search(r'LAST SUMMARY RESPONSE:(.*?)$', content, re.DOTALL)
    if match:
        query = match.group(1).strip()
        name_match = re.search(r'^(.*?),', query)
        name = name_match.group(1) if name_match else None
        return query, name
    return None, None

def generate_response(query, sitrep_title, name):
    current_time = datetime.utcnow() + timedelta(hours=1)  # Assuming GMT+1
    response_time = current_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Modified prompt to ensure the response is structured as required
    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed response based on the query, following this structure:
    1. Address {name} and thank them for their inquiry about the files or hash.
    2. Summarize the key findings in the sitrep related to the files or hash.
    3. Provide next steps for investigating the files, focusing on discovering file paths and assessing anomalies.
    4. Avoid giving generic responses; tailor the response based on the specific situation raised in the sitrep.
    5. Keep the response specific, actionable, and avoid unnecessary elaboration unless directly relevant to the query.

    Example response structure:

    {name}, {response_time}
    
    Thank you for your inquiry regarding the file associated with the detected hash.

    Based on the information available, the following files match the hash: "windows/win.ini" and "winnt/win.ini". These are common system configuration files, but to proceed further with your investigation, please consider the following steps:

    1. **Exact File Path Discovery**: Check your system for the full paths of these files to ensure there are no unexpected duplicates or modifications.
    2. **File Integrity Check**: Compare the hash of these files against a known baseline to verify if they have been altered.
    3. **Further Investigation**: Use file monitoring tools or consult Gradient logs for any history of file modification or unauthorized access.
    
    Let me know if additional information is needed, or if any other files or data points need to be analyzed.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Using GPT-4 for better context handling
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing focused responses to sitrep queries. Your responses should be concise, actionable, and directly address the issue raised in the sitrep."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']


def process_sitrep(content):
    try:
        sitrep_title_match = re.search(r'SITREP TITLE:(.*?)$', content, re.MULTILINE)
        sitrep_title = sitrep_title_match.group(1).strip() if sitrep_title_match else "Unknown Title"
        
        query, name = extract_query_and_name(content)
        if query:
            response = generate_response(query, sitrep_title, name)
            return response
        else:
            return "Failed to extract query from the sitrep content."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Sitrep Processor")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Sitrep content here:", height=300)
    
    if st.button("Process Sitrep"):
        if not content:
            st.error("Please provide the Sitrep content.")
        else:
            response = process_sitrep(content)
            st.markdown("### Generated Response")
            st.markdown(response)

if __name__ == "__main__":
    main()
