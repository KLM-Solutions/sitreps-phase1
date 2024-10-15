import streamlit as st
import openai
import os
import re

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

def generate_response(query, sitrep_title, name=None):
    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed, specific response addressing the query. The response should:
    1. Directly address the specific concerns raised in the query
    2. Provide detailed information about the alert mentioned in the SITREP TITLE
    3. Explain actionable steps tailored to this specific situation
    4. Discuss relevant thresholds or metrics specific to this case
    5. Offer recommendations that are directly applicable to the queried scenario

    Do not include any greetings, closings, signatures, or timestamps. 
    Focus solely on providing a comprehensive, tailored answer to the query.
    Avoid generic advice and ensure all information is specific to this sitrep and query.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed, contextual responses to sitrep queries. Your responses should be comprehensive, highly specific, and tailored to each unique situation."},
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
