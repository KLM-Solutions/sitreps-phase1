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
        self.setup_template_matcher()
        self.setup_phase_classifier()
    
    def setup_vector_store(self):
        self.vector_store = FAISS.from_texts(SITREP_TEMPLATES, self.embeddings)
    
    def setup_template_matcher(self):
        """Setup dedicated LLM chain for template matching"""
        system_template = """You are a security alert template matcher. Your job is to match the given alert text to the most appropriate template from the provided list. Consider:
        1. Keywords and phrases that uniquely identify each template type
        2. The overall context and nature of the alert
        3. Specific indicators mentioned in the alert
        
        Available templates:
        {templates}
        
        Return ONLY the exact matching template name. If no close match is found, return "Unknown Template"."""
        
        human_template = """Alert text:
        {alert_text}
        
        Match this alert to the most appropriate template."""
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        self.template_matcher = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([system_message, human_message])
        )
    
    def setup_phase_classifier(self):
        """Setup dedicated LLM chain for Phase 1 classification"""
        system_template = """You are a query classifier for a security operations system. Your role is to determine if a given query falls under Phase 1 criteria.

        Phase 1 queries are general in nature and include:
        1. Requests for general best practices
        2. General mitigation strategies
        3. General security recommendations
        4. Standard security hygiene questions
        5. Non-customer-specific guidance

        Queries that are NOT Phase 1:
        1. Customer-specific log analysis
        2. Specific IP or system investigations
        3. Technical configuration questions
        4. Customer-specific infrastructure queries
        
        Return ONLY "PHASE_1" or "NOT_PHASE_1" as your response."""
        
        human_template = """Query:
        {query}
        
        Is this a Phase 1 query?"""
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        self.phase_classifier = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([system_message, human_message])
        )

    def find_matching_template(self, sitrep_text: str) -> str:
        return self.template_matcher.run(
            templates="\n".join(SITREP_TEMPLATES),
            alert_text=sitrep_text
        ).strip()

    def is_phase_1_query(self, query: str) -> bool:
        result = self.phase_classifier.run(query=query).strip()
        return result == "PHASE_1"

    def extract_status(self, text: str) -> Optional[str]:
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None
    
    def answer_query(self, alert_summary: str, query: str, is_phase_1: bool) -> str:
        """Generate a focused response based on whether it's a Phase 1 query"""
        if not is_phase_1:
            return ("⚠️ This query requires specific customer log analysis or technical details that are beyond "
                   "Phase 1 automation. Please escalate to a Customer Analyst for detailed review.")
        
        system_template = """You are a security analyst providing responses to general security queries.
        For Phase 1 queries, provide:
        - Industry standard best practices
        - General mitigation strategies
        - Common security recommendations
        - Standard security hygiene guidelines
        
        Rules:
        - Focus on general, widely-accepted security practices
        - Avoid customer-specific recommendations
        - Be clear and actionable in your advice
        - Don't reference specific customer data or logs
        - Keep responses concise and practical"""
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        
        human_template = """
        Alert Context:
        {alert_summary}
        
        Query: {query}
        
        Provide a general security recommendation response."""
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(alert_summary=alert_summary, query=query)

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.find_matching_template(alert_summary)
            status = self.extract_status(alert_summary)
            
            query_response = None
            is_phase_1 = False
            if client_query:
                is_phase_1 = self.is_phase_1_query(client_query)
                query_response = self.answer_query(alert_summary, client_query, is_phase_1)
            
            return {
                "template": template,
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response.strip() if query_response else None
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .main-title { 
            color: #2a5298; 
            font-size: 24px; 
            font-weight: bold; 
            margin: 20px 0; 
            text-align: center;
        }
        .header-box { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #2a5298;
        }
        .query-response { 
            background: white; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }
        .response-heading {
            color: #2a5298;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .phase-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: bold;
            margin: 5px 0;
        }
        .phase-1 {
            background-color: #2ecc71;
            color: white;
        }
        .not-phase-1 {
            background-color: #e74c3c;
            color: white;
        }
        .stButton>button {
            background-color: #2a5298;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    analyzer = SitrepAnalyzer()
    
    st.markdown('<h1 class="main-title">Sitreps Analysis System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        alert_summary = st.text_area("Summary Analysis", height=300)

    with col2:
        client_query = st.text_area("User Query", height=150)
    
    if st.button("Analyze", type="primary"):
        if not alert_summary:
            st.error("Please enter alert details.")
            return
        
        with st.spinner("Analyzing..."):
            result = analyzer.analyze_sitrep(alert_summary, client_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display template and status
                header_html = f'<div class="header-box"><strong>Matched Template:</strong> {result["template"]}'
                if result["status"]:
                    header_html += f'<br><strong>Status:</strong> {result["status"]}'
                header_html += '</div>'
                st.markdown(header_html, unsafe_allow_html=True)
                
                # Display query response if exists
                if result.get("query_response"):
                    phase_badge = ('<span class="phase-badge phase-1">Phase 1 Query</span>' 
                                 if result["is_phase_1"] 
                                 else '<span class="phase-badge not-phase-1">Not Phase 1 Query</span>')
                    
                    st.markdown(
                        '<div class="query-response">' +
                        '<div class="response-heading">USER RESPONSE:</div>' +
                        phase_badge +
                        f'<p>{result["query_response"]}</p>' +
                        '</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
