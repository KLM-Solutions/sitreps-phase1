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
    4. Provide detailed, actionable steps such as integrity checks, file path checks, and system behavior analysis.
    5. Offer additional recommendations based on standard cybersecurity practices (e.g., setting up monitoring for similar hashes, cross-referencing with malicious file databases).
    6. Avoid giving generic responses; tailor the response based on the specific situation raised in the sitrep.

    Example response structure:

    {name}, {response_time}

    Thank you for your inquiry regarding the file associated with the detected hash.

    Based on the information available, the following files match the hash: "windows/win.ini" and "winnt/win.ini". These are common system configuration files, but to proceed further with your investigation, please consider the following steps:

    1. **Exact File Path Discovery**: First, locate the exact file paths of "windows/win.ini" and "winnt/win.ini" on your system. You should check whether these files exist in any non-standard directories or have duplicate copies in unexpected locations. Malicious actors sometimes hide modified versions of such files in non-standard directories.

    2. **File Integrity Check**: Verify the integrity of the identified files by comparing their hash values against known good copies. If any differences are detected, this may indicate unauthorized changes, possibly signifying compromise or tampering.

    3. **Investigate Anomalies**: Review the file properties such as last modified date, file permissions, and ownership to identify any irregularities. Also, check if these files have any unusual access patterns or if they were modified by untrusted users or processes.

    4. **Cross-reference with Malicious File Databases**: Search for the identified file hashes in known malicious file databases or threat intelligence platforms to verify whether they have been associated with any security incidents.

    5. **System Behavior Monitoring**: It's essential to monitor the system for any unusual behavior, particularly in directories or processes related to these files. Set up alerts for any unexpected changes to these files or their hash values in the future. You can also review security logs or Gradient's logs for any associated events, such as unauthorized modifications or access attempts.

    6. **Correlate with Broader Activity**: Look for other system files that may share similar hashes, and check if other systems on your network exhibit the same file modifications. This can help you detect patterns of compromise or malware spreading across multiple systems.

    Please confirm whether you have already performed any checks on these file names or hashes, and let me know if further investigation or assistance is required. If you need specific tools or further data points for analysis, feel free to reach out.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Using GPT-4 for better context handling
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing focused responses to sitrep queries. Your responses should be comprehensive, actionable, and directly address the issue raised in the sitrep."},
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
