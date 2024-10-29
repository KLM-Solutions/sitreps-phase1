import streamlit as st
from langchain.chat_models import ChatOpenAI
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

class TemplateMatcher:
    """Improved template matcher using direct LLM matching"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_matcher()
        self.templates = SITREP_TEMPLATES

    def setup_matcher(self):
        system_template = """You are an alert template matcher. Given the alert text and available templates, return the most appropriate matching template name. Only return the exact template name, nothing else."""

        human_template = """Alert Text: {alert_text}

Available Templates:
{templates}

Return only the matching template name:"""
        
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

class SecurityResponseGenerator:
    """Enhanced response generator with improved formatting"""
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key
        )
        self.setup_chain()

    def setup_chain(self):
        system_template = """You are a cybersecurity expert. Provide brief, actionable responses.

Response Requirements:
1. Maximum 3 short sentences total
2. Focus on immediate actions only
3. Use clear, simple language
4. No technical details unless crucial

Structure your response as:
Summary: [One line description]
Action: [What to do now]"""

        human_template = """Alert Type: {alert_type}
Alert Summary: {alert_summary}
Query: {query}

Provide response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate_response(self, query: str, alert_type: str = None, alert_summary: str = None) -> Dict[str, str]:
        try:
            response = self.chain.run(
                query=query,
                alert_type=alert_type or "Not Specified",
                alert_summary=alert_summary or "Not Provided"
            )
            
            return {
                'success': True,
                'response': response.strip()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class PhaseClassifier:
    """Classifier for Phase 1 queries"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_classifier()

    def setup_classifier(self):
        system_template = """Classify if a security query can be handled with general knowledge (Phase 1) or needs specific analysis.

Phase 1 queries ask about:
- General security practices
- Standard recommendations
- Common configurations
- Basic preventive measures

NOT Phase 1 queries involve:
- Specific log analysis
- Custom configurations
- Technical debugging
- Environment-specific issues

Return only PHASE_1 or NOT_PHASE_1"""

        human_template = """Alert: {alert_summary}
Query: {query}

Classification:"""

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

class SitrepAnalyzer:
    """Main analyzer with improved response formatting"""
    def __init__(self):
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.response_generator = SecurityResponseGenerator(OPENAI_API_KEY)
        self.phase_classifier = PhaseClassifier(OPENAI_API_KEY)

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.template_matcher.match_template(alert_summary)
            query_response = None
            
            if client_query:
                is_phase_1 = self.phase_classifier.classify(alert_summary, client_query)
                if is_phase_1:
                    response = self.response_generator.generate_response(
                        query=client_query,
                        alert_type=template,
                        alert_summary=alert_summary
                    )
                    query_response = response.get('response') if response.get('success') else None
                else:
                    query_response = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template,
                "query_response": query_response
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    """Streamlit UI with improved response formatting"""
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .template-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .response-box {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #2196F3;
        }
        .response-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #1976D2;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Security Alert Analyzer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        alert_summary = st.text_area("Alert Summary", height=300,
                                   placeholder="Paste SITREP here...")

    with col2:
        client_query = st.text_area("Client Query", height=150,
                                  placeholder="Enter your question here...")
    
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
                # Show template match
                if result["template"] != "Unknown Template":
                    st.markdown(
                        f'<div class="template-box">'
                        f'<strong>Matched Template:</strong> {result["template"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Show response with header
                if result.get("query_response"):
                    st.markdown(
                        '<div class="response-box">'
                        '<div class="response-header">Query Response:</div>'
                        f'{result["query_response"]}'
                        '</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
