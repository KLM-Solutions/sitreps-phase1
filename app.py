import streamlit as st
import openai
import os
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_query(content):
    match = re.search(r'LAST SUMMARY RESPONSE:(.*?)$', content, re.DOTALL)
    return match.group(1).strip() if match else None

def generate_response(query, sitrep_title):
    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed response addressing the query. The response should:
    1. Start with a greeting addressing the person who asked the query
    2. Provide information about the alert mentioned in the SITREP TITLE
    3. Explain actionable steps
    4. Discuss relevant thresholds
    5. Offer recommendations
    6. End with a closing statement

    Do not include any timestamps or repeat the query in your response.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed, contextual responses to sitrep queries. Your responses should be comprehensive and tailored to the specific situation."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def process_sitrep(content):
    try:
        sitrep_title_match = re.search(r'SITREP TITLE:(.*?)$', content, re.MULTILINE)
        sitrep_title = sitrep_title_match.group(1).strip() if sitrep_title_match else "Unknown Title"
        
        query = extract_query(content)
        if query:
            response = generate_response(query, sitrep_title)
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
