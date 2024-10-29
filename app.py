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
    "Malware IP",
    "Anomalous Kerberos Authentication",
    "Kerberos-related alert",
    "Kerberos Authentication Abuse"
]

class CrispResponseGenerator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_chain()

    def setup_chain(self):
        system_template = """You are a security expert providing direct, actionable responses.
    
        Response Guidelines:
        1. State the effectiveness of suggested approaches
        2. Provide alternative or complementary solutions
        3. Explain briefly why certain approaches are better
        4. Be specific but avoid customer-specific details
        5. Focus on industry best practices
        6. Be direct and concise
        7. Avoid generic advice
        
        Format:
        - Start with direct answer about suggested approach
        - List better or complementary solutions if applicable
        - Include brief technical justification if needed
        
        Example Good Response:
        "IP blocking provides limited protection. Enable SSL/TLS inspection with certificate validation and set threshold alerts for suspicious traffic patterns at 100 requests/minute."
        
        Example Bad Response:
        "You should consider multiple approaches including IP blocking which can help in some cases, and also think about SSL/TLS decryption as it might provide better visibility..."""

        human_template = """Alert Context: {alert_summary}
        Query: {query}
        
        Provide direct technical response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate(self, alert_summary: str, query: str) -> str:
        return self.chain.run(alert_summary=alert_summary, query=query).strip()

class PhaseClassifier:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_classifier()

    def setup_classifier(self):
        system_template = """You are a security query classifier. 

       # System Context
You are an AI assistant specialized in handling general customer inquiries about cybersecurity and IT best practices. Your role is to:
1. Determine if a query is general or specific
2. Provide standardized responses for general queries
3. Indicate when a query needs human analyst attention

# Query Classification Rules
- HANDLE queries that ask for:
  * Industry best practices
  * General recommendations
  * Standard mitigation strategies
  * Common security guidelines
  * Prevention techniques
  * Educational information
  * High-level process explanations

- ESCALATE queries that involve:
  * Specific customer logs
  * Custom configurations
  * System-specific issues
  * Detailed technical debugging
  * Customer-specific setups
  * Unique implementation details

# Response Format
Return ONLY 'PHASE_1' for general queries that can be automated, or 'PHASE_2' for specific queries requiring analyst review.
"""

        human_template = """Alert Context: {alert_summary}
Query: {query}

Classify as PHASE_1 or PHASE_2:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def classify(self, alert_summary: str, query: str) -> bool:
        result = self.chain.run(alert_summary=alert_summary, query=query).strip()
        return result == "PHASE_1"

class TemplateMatcher:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_matcher()
        self.templates = SITREP_TEMPLATES

    def setup_matcher(self):
        system_template = """You are a specialized security alert template matcher focused on exact pattern matching.
        
        Key matching criteria:
        1. Authentication patterns (Kerberos, sign-ins, access)
        2. Traffic patterns (anomalous, internal, internet)
        3. IP-based threats (blacklisted, tor, spam, malware)
        4. Protocol indicators (DNS, TLS, NTP)
        5. Specific services (bots, scanners, anonymization)
        
        Return ONLY the exact matching template name. If no clear match exists, return "Unknown Template"."""

        human_template = """Available Templates:
        {templates}

        Alert Text:
        {alert_text}

        Return exact matching template name:"""
        
        self.matcher_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def match_template(self, alert_text: str) -> str:
        result = self.matcher_chain.run(
            templates="\n".join(self.templates),
            alert_text=alert_text
        ).strip()
        return result if result in self.templates else "Unknown Template"

class SitrepAnalyzer:
    def __init__(self):
        try:
            self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
            self.response_generator = CrispResponseGenerator(OPENAI_API_KEY)
            self.phase_classifier = PhaseClassifier(OPENAI_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {str(e)}")
            st.stop()

    def extract_status(self, text: str) -> Optional[str]:
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.template_matcher.match_template(alert_summary)
            status = self.extract_status(alert_summary)
            
            query_response = None
            is_phase_1 = False
            
            if client_query:
                is_phase_1 = self.phase_classifier.classify(alert_summary, client_query)
                if is_phase_1:
                    query_response = self.response_generator.generate(alert_summary, client_query)
                else:
                    query_response = "⚠️ Requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template,
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}

def main():
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .header-box { 
            background: #f8f9fa; 
            padding: 10px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #2a5298;
        }
        .response-box { 
            background: white; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0; 
            border-left: 4px solid #3498db; 
        }
        </style>
        """, unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            alert_summary = st.text_area("Summary Analysis", height=300)

        with col2:
            client_query = st.text_area("User Query", height=150)
        
        if st.button("Analyze", type="primary"):
            if not alert_summary:
                st.error("Please enter alert details.")
                return
            
            analyzer = SitrepAnalyzer()
            with st.spinner("Analyzing..."):
                result = analyzer.analyze_sitrep(alert_summary, client_query)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Show template and status in header box if they exist
                    header_content = []
                    if result["template"] != "Unknown Template":
                        header_content.append(f"<strong>Matched Template:</strong> {result['template']}")
                    if result.get("status"):
                        header_content.append(f"<strong>Status:</strong> {result['status']}")
                    
                    if header_content:
                        st.markdown(
                            '<div class="header-box">' + 
                            '<br>'.join(header_content) + 
                            '</div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Show response if exists
                    if result.get("query_response"):
                        st.markdown(
                            '<div class="response-box">' +
                            '<strong>USER RESPONSE:</strong><br>' +
                            f'{result["query_response"]}' +
                            '</div>',
                            unsafe_allow_html=True
                        )
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
