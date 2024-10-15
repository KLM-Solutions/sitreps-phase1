import streamlit as st
import openai
import os
import json
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_and_parse_json(content):
    # Remove any potential markdown code block syntax
    content = re.sub(r'```json\s*|\s*```', '', content)
    # Remove any leading/trailing whitespace
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {str(e)}")
        st.text("Raw content received:")
        st.code(content)
        return None

def extract_sitrep_info(content):
    extraction_prompt = f"""
    Analyze the following sitrep content and extract:
    1. SITREP TITLE
    2. SITREP STATUS
    3. ORGANIZATION
    4. LAST SUMMARY RESPONSE (including timestamp, name, and message)

    Content:
    {content}

    Provide the output as a JSON object with keys: "SITREP TITLE", "SITREP STATUS", "ORGANIZATION", "LAST SUMMARY RESPONSE".
    Ensure the output is a valid JSON object.
    """

    extraction_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract key information from the sitrep content and output as valid JSON."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    
    extracted_content = extraction_response.choices[0].message['content']
    return clean_and_parse_json(extracted_content)

def generate_detailed_response(sitrep_info):
    response_prompt = f"""
    Based on the following sitrep information, generate a detailed response:

    SITREP TITLE: {sitrep_info['SITREP TITLE']}
    ORGANIZATION: {sitrep_info['ORGANIZATION']}
    LAST SUMMARY RESPONSE: {sitrep_info['LAST SUMMARY RESPONSE']}

    Your response should:
    1. Address the specific query or concern in the LAST SUMMARY RESPONSE
    2. Provide detailed information about the alert or issue mentioned in the SITREP TITLE
    3. Explain the implications of the observed behavior
    4. Suggest actionable steps for investigation or resolution
    5. Include relevant thresholds or statistics if applicable
    6. Offer guidance on how to interpret the information
    7. Ask for confirmation or further information if needed

    Format the response as follows:
    [Timestamp in GMT]
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
        if sitrep_info:
            response = generate_detailed_response(sitrep_info)
            return sitrep_info, response
        else:
            return None, "Failed to extract sitrep information. Please check the error message above."
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, "An unexpected error occurred. Please try again."

def main():
    st.title("Sitrep Processor with Detailed Responses")
    
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    content = st.text_area("Paste the Sitrep content here:", height=300)
    
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
