import streamlit as st
import openai
import os
import re

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_query(content):
    # Extract the last summary response
    match = re.search(r'LAST SUMMARY RESPONSE:(.*?)$', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def generate_response(query, sitrep_title):
    prompt = f"""
    Based on the following sitrep information:
    SITREP TITLE: {sitrep_title}
    QUERY: {query}

    Generate a detailed response in the following format:

    [Timestamp of the query in GMT]
    [Query content]
    [Timestamp for the response in GMT (current time)]
    [Detailed response addressing the query, providing information about the alert, actionable steps, thresholds, and recommendations. The response should be similar in style and depth to the example provided earlier.]

    Ensure the response is comprehensive, tailored to the specific sitrep context, and provides valuable insights and recommendations.
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
                st.subheader("Extracted Query and Generated Response")
                st.text(response)
            else:
                st.error(response)

if __name__ == "__main__":
    main()
