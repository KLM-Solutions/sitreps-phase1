import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import openai
import re
from typing import Dict, Optional, List

# API Configuration
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# Template definitions
SITREP_TEMPLATES = [
    "Anomalous Internal Traffic",
    "445 Blacklisted IP",
    "DNS Queries to bad domains",
    "Anomalous Internet Traffic: Size",
    "Anomalous Internet Traffic: Packets",
    "Anomalous Internet Traffic: Sessions",
    "Anonymization Services IP",
    "Bots IP",
    "Gradient 365 alert: Unusual sign-in activity detected",
    "Evaluated Addresses",
    "Scanning performed using automation tools",
    "Scanning IP",
    "TLS Traffic to bad domains",
    "Gradient 365 alert: Sign-in from a Blacklisted IP detected",
    "Social Engineering",
    "Tor IP",
    "Spam IP",
    "NTP TOR IP",
    "Malware IP"
]

class SitrepAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.setup_vector_store()
    
    def setup_vector_store(self):
        self.vector_store = FAISS.from_texts(SITREP_TEMPLATES, self.embeddings)
    
    def find_matching_template(self, sitrep_text: str) -> str:
        matches = self.vector_store.similarity_search(sitrep_text, k=1)
        return matches[0].page_content if matches else "Unknown Template"

    def extract_status(self, text: str) -> Optional[str]:
        """Extract status if present in the alert"""
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None

    def generate_analysis(self, alert_summary: str) -> str:
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a security analyst. Provide only the most critical technical insights.
            Focus on immediate security implications."""
        )
        
        human_template = """
        Alert Summary:
        {alert_summary}
        
        Provide a brief technical analysis focusing only on the key security implications.
        Do not include any dates, event IDs, or technical codes in your response.
        Focus on the security impact and risk.
        """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(alert_summary=alert_summary)

    def answer_query(self, alert_summary: str, query: str) -> str:
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a security analyst. Your task is to provide a detailed but focused answer 
            to the specific query. Include all relevant details but avoid generic security advice 
            unless specifically asked."""
        )
        
        human_template = """
        Alert Summary:
        {alert_summary}
        
        Query: {query}
        
        Rules:
        1. Address the specific query directly and completely
        2. Include all relevant technical details from the alert
        3. Do not include generic security advice unless specifically asked
        4. Avoid mentioning dates, event IDs, or specific codes
        
        Provide a comprehensive but focused response that directly answers the query.
        """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(alert_summary=alert_summary, query=query)

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.find_matching_template(alert_summary)
            status = self.extract_status(alert_summary)
            analysis = self.generate_analysis(alert_summary)
            
            query_response = None
            if client_query:
                query_response = self.answer_query(alert_summary, client_query)
            
            return {
                "template": template,
                "status": status,
                "analysis": analysis.strip(),
                "query_response": query_response.strip() if query_response else None
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .main-title { color: #2a5298; font-size: 24px; font-weight: bold; margin: 20px 0; }
        .header-box { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #2a5298;
        }
        .analysis-box { 
            background: white; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #2ecc71;
        }
        .query-response { 
            background: #fff; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }
        </style>
        """, unsafe_allow_html=True)
    
    analyzer = SitrepAnalyzer()
    
    st.markdown('<p class="main-title">Alert Analysis System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        alert_summary = st.text_area("Alert Details", height=300)

    with col2:
        client_query = st.text_area("Query (Optional)", height=150)
    
    if st.button("Analyze", type="primary"):
        if not alert_summary:
            st.error("Please enter alert details.")
            return
        
        with st.spinner("Analyzing..."):
            result = analyzer.analyze_sitrep(alert_summary, client_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display header information
                header_html = f'<div class="header-box"><strong>Matched Template:</strong> {result["template"]}'
                if result["status"]:
                    header_html += f'<br><strong>Status:</strong> {result["status"]}'
                header_html += '</div>'
                st.markdown(header_html, unsafe_allow_html=True)
                
                # Display technical analysis
                st.markdown('<div class="analysis-box">' + 
                          f'<strong>Technical Analysis:</strong><br>{result["analysis"]}' + 
                          '</div>', unsafe_allow_html=True)
                
                # Display query response if exists
                if result["query_response"]:
                    st.markdown('<div class="query-response">' +
                              f'<strong>Query Response:</strong><br>{result["query_response"]}' +
                              '</div>', unsafe_allow_html=True)
                
                # Download button with formatted content
                analysis_text = f"""# Alert Analysis Report

Matched Template: {result["template"]}
{f'Status: {result["status"]}' if result["status"] else ''}

Technical Analysis:
{result["analysis"]}

{f'Query Response:\n{result["query_response"]}' if result["query_response"] else ''}
"""
                
                st.download_button(
                    label="Download Analysis",
                    data=analysis_text,
                    file_name="analysis.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
