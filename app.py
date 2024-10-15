import streamlit as st
import openai
import os
import json
from datetime import datetime, timedelta

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_query(content):
    extraction_prompt = f"""
    Analyze the following sitrep content and extract:
    1. The most relevant user query or request
    2. The name of the person making the query (if available)

    Content:
    {content}

    Provide the output as a JSON object with keys: "query" and "name".
    If there's no clear query, infer one based on the context of the sitrep.
    """

    extraction_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with extracting the most relevant user query from sitrep content."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    
    try:
        extracted_info = json.loads(extraction_response.choices[0].message['content'])
        return extracted_info['query'], extracted_info['name']
    except json.JSONDecodeError:
        return None, None

def generate_response(query, sitrep_title, name):
    current_time = datetime.utcnow() + timedelta(hours=1)  # Assuming GMT+1
    response_time = current_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed response following this structure:
    1. Address the person by name (if provided) and acknowledge their query.
    2. Provide specific information about the alert mentioned in the SITREP TITLE.
    3. Explain the implications of the observed behavior.
    4. Suggest actionable steps for investigation or resolution.
    5. If applicable, provide information about thresholds or statistics related to the issue.
    6. Offer guidance on interpreting the information.
    7. Ask for any necessary confirmations or further information.

    Use the following format:
    {name}, {response_time}
    [Detailed response following the structure above]

    Do not include any closing remarks, "Best regards," signatures, or cybersecurity team mentions at the end.
    Ensure the response is comprehensive, tailored to the specific sitrep context, and provides valuable insights and recommendations.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed, contextual responses to sitrep queries. Your responses should be comprehensive and tailored to the specific situation, without any closing remarks or signatures."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        sitrep_title_match = content.split("SITREP TITLE:")[1].split("\n")[0].strip()
        query, name = extract_query(content)
        if query:
            response = generate_response(query, sitrep_title_match, name)
            return query, response
        else:
            return None, "Failed to extract a relevant query from the sitrep content."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

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
            query, response = process_sitrep(content)
            if query:
                st.subheader("Identified User Query")
                st.markdown(query)
                st.subheader("Generated Response")
                st.markdown(response)
            else:
                st.error(response)

if __name__ == "__main__":
    main()
