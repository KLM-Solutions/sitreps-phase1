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
    "Kerberos-related alert",
    "Anomalous Kerberos Authentication",
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
        system_template = """You are a specialized security response system providing highly focused, actionable recommendations.

        Core Rules:
        1. Provide critical, directly actionable recommendations
        2. Focus on high-impact security measures specific to the alert
        3. Avoid generic advice
        4. No explanations or justifications
        5. Each point must be specific and implementable
        6. Responses must be technically precise
        7. Use exact numbers/thresholds where applicable

        Response Style:
        • Start each point with an action verb
        • Include specific technical parameters
        • Use precise technical terminology
        • Focus on immediate actions
        • Provide exact thresholds or ranges
        • Keep points clear and implementable

        Example Format:
        • Configure rate limiting to <exact number>
        • Restrict access to <specific sources>
        • Enable <specific security control> with <exact parameters>"""

        human_template = """Alert Context: {alert_summary}
        Query: {query}
        
        Provide precise technical response:"""

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
        system_template = """You are a query classifier for security operations that determines if a customer query falls under Phase 1 automation scope.

        PHASE 1 QUERIES CRITERIA:
        - Queries that are general in nature and don't require specific log analysis
        - Requests for general security guidance or information
        - Questions about industry-standard best practices
        - Questions about general mitigation strategies
        - Inquiries about general security recommendations
        - Questions about standard steps to prevent security issues
        - Queries that can be answered using general security knowledge

        NOT PHASE 1 QUERIES (Requires CA Review):
        - Queries requiring analysis of specific customer logs
        - Questions about specific technical configurations
        - Inquiries about specific system behaviors
        - Questions requiring deep customization
        - Queries about specific IPs or infrastructure
        - Anything requiring access to customer-specific data
        - Technical queries requiring system-specific knowledge

        Return ONLY "PHASE_1" if the query can be handled with general security knowledge, or "NOT_PHASE_1" if it requires specific analysis or CA review."""

        human_template = """Query: {query}
        
        Based ONLY on whether this query needs specific customer analysis or can be answered with general security knowledge, classify as PHASE_1 or NOT_PHASE_1."""
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def classify(self, query: str) -> bool:
        result = self.chain.run(query=query).strip()
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
        system_template = """Match alerts to exact template names based on technical indicators and context.
        Return ONLY the exact template name."""

        human_template = """Templates: {templates}
        Alert: {alert_text}
        Match to template:"""
        
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
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.response_generator = CrispResponseGenerator(OPENAI_API_KEY)
        self.phase_classifier = PhaseClassifier(OPENAI_API_KEY)

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
                is_phase_1 = self.phase_classifier.classify(client_query)
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
            return {"error": f"Error: {str(e)}"}

def main():
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .response-box { 
            background: white; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0; 
            border-left: 4px solid #3498db; 
        }
        </style>
        """, unsafe_allow_html=True)
    
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
                if result.get("query_response"):
                    st.markdown('<div class="response-box">' +
                              '<strong>USER RESPONSE:</strong><br>' +
                              f'{result["query_response"]}' +
                              '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
