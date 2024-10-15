import streamlit as st
import openai
import os
import json
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_json_string(json_string):
    return re.sub(r'^```json\n|```$', '', json_string, flags=re.MULTILINE).strip()

def extract_sitrep_info(content):
    extraction_prompt = f"""
    Analyze the following sitrep content and extract:
    1. SITREP TITLE
    2. SITREP STATUS
    3. ORGANIZATION
    4. FULL CONVERSATION HISTORY (including timestamps, names, and messages)
    5. MAIN ISSUE or ALERT DESCRIPTION

    Content:
    {content}

    Provide the output in JSON format with keys: title, status, organization, conversation_history, main_issue
    The conversation_history should be a list of dictionaries, each containing 'timestamp', 'name', and 'message' keys.
    """

    extraction_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract detailed information from the sitrep content, including the full conversation history."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    
    cleaned_json = clean_json_string(extraction_response.choices[0].message['content'])
    return json.loads(cleaned_json)

def generate_detailed_response(sitrep_info):
    response_prompt = f"""
    Based on the following sitrep information, generate a detailed response similar to the example provided:

    SITREP TITLE: {sitrep_info['title']}
    ORGANIZATION: {sitrep_info['organization']}
    MAIN ISSUE: {sitrep_info['main_issue']}
    CONVERSATION HISTORY:
    {json.dumps(sitrep_info['conversation_history'], indent=2)}

    Your response should:
    1. Address the latest query or concern in the conversation history
    2. Provide detailed information about the alert or issue
    3. Explain the implications of the observed behavior
    4. Suggest actionable steps for investigation or resolution
    5. Include relevant thresholds or statistics if applicable
    6. Offer guidance on how to interpret the information
    7. Ask for confirmation or further information if needed

    Format the response as follows:
    **Response Summary**
    [Full response here, maintaining a conversational yet informative tone]

    Ensure the response is comprehensive, tailored to the specific sitrep context, and provides valuable insights and recommendations.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed, contextual responses to sitrep queries. Your responses should be comprehensive and tailored to the specific situation."},
            {"role": "user", "content": response_prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        sitrep_info = extract_sitrep_info(content)
        response = generate_detailed_response(sitrep_info)
        return sitrep_info, response
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def main():
    st.title("Comprehensive Sitrep Processor")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the complete Sitrep content here:", height=300)
    
    if st.button("Process Sitrep"):
        if not content:
            st.error("Please provide the Sitrep content.")
        else:
            sitrep_info, response = process_sitrep(content)
            
            if sitrep_info:
                st.subheader("Extracted Sitrep Information")
                st.json(sitrep_info)
            
            st.subheader("Generated Detailed Response")
            st.markdown(response)

if __name__ == "__main__":
    main()
