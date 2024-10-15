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
    [Detailed response following the structure above]

    Do not include any closing remarks, "Best regards," signatures, or cybersecurity team mentions at the end.
    Ensure the response is comprehensive, tailored to the specific sitrep context, and provides valuable insights and recommendations.
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
            return query, response
        else:
            return None, "Failed to extract query from the sitrep content."
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
                st.subheader("User Query")
                st.markdown(query)
                st.subheader("Generated Response")
                st.markdown(response)
            else:
                st.error(response)

if __name__ == "__main__":
    main()
