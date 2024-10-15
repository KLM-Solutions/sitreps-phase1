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

    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed response following this structure:
    1. Address the person by name (if provided) and thank them for their inquiry.
    2. Provide specific information about the alert mentioned in the SITREP TITLE.
    3. Explain the implications of the observed behavior.
    4. Suggest actionable steps for investigation or resolution.
    5. If applicable, provide information about thresholds or statistics related to the issue.
    6. Offer guidance on interpreting the information.
    7. Ask for any necessary confirmations or further information.

    Use the following format:
    {name}, {response_time}
    Robert Mettee, Wed, 09 Oct 2024 20:34:55 GMT
Gradient team can you provide us more information on this alert? What is actionable from this alert? What is the threshold for an unusual amount of kerberos requests?

Pranith Jain, Thu, 10 Oct 2024 9:30:30 GMT 
Hi Robert, Thank you for reaching out regarding the alert for Event ID 4769 and error code 0x1b (Server principal valid for user2user only).
Regarding more information on the alert, this essentially means that there were Kerberos service ticket requests where the Service Principal Name (SPN) was only valid for User-to-User (U2U) authentication. These types of failures generally happen due to a mismatch or misconfiguration, often linked to service account settings or improper delegation. Monitoring this closely helps to spot any potential misconfigurations or signs of Kerberos abuse.

In terms of actionable steps, we recommend investigating the service accounts or SPNs related to these requests. The spike in requests could indicate misconfigured delegation, an expired or invalid service principal, or, in some cases, a malicious attempt to exploit Kerberos. Reviewing the service account configurations—especially U2U-based authentications—and verifying if any recent changes may have impacted delegation or SPN settings will be essential.

As for the threshold for an unusual amount of Kerberos requests, we observed the following spikes in volume: 1,411 requests on September 27, 781 on September 28, and 642 on September 29. While every environment is different, and there's no universal threshold, this does represent a significant deviation from typical activity and warrants further review to ensure there’s no misconfiguration or security risk.
Please refer to the following links for guidance on viewing and resetting incorrect SPN names in your environment:
Viewing and Resetting Incorrect SPNs - Windows Server 2003
Viewing SPNs - Windows Server 2012 R2 and 2012
Please let us know if this activity is expected, and if these are high-value accounts that you would like to receive alerts for in the future.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed, contextual responses to sitrep queries. Your responses should be comprehensive and tailored to the specific situation, mimicking the style and depth of the example provided, but without any closing remarks or signatures."},
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
