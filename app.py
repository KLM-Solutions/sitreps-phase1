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
    """Matcher for identifying alert templates"""
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

class QueryPhaseAnalyzer:
    """Analyzer for determining query phase"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_analyzer()

    def setup_analyzer(self):
        system_template = """Determine if a security query can be answered with general knowledge.

Return EXACTLY "PHASE_1" if the query:
- Asks about security best practices
- Compares security solutions
- Seeks standard recommendations
- Asks about effectiveness of controls
- Questions about tools/features
- Basic mitigation strategies
- Security concepts and approaches

Return EXACTLY "NOT_PHASE_1" only if the query requires:
- Specific log analysis
- Customer environment details
- Technical debugging
- Incident investigation
- Custom configurations

Example PHASE_1 queries:
"Is SSL better than IP blocking?"
"What's the best approach for this alert?"
"Which security measure works better?"
"Should we implement this control?"

Return only PHASE_1 or NOT_PHASE_1."""

        human_template = """Query: {query}
Alert Context: {context}

Classification:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def analyze_phase(self, query: str, context: str = "") -> bool:
        try:
            response = self.chain.run(query=query, context=context).strip()
            return response == "PHASE_1"
        except Exception:
            return False

class SecurityResponseGenerator:
    """Enhanced response generator for security queries"""
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key
        )
        self.setup_chain()

    def setup_chain(self):
        system_template = """You are a cybersecurity expert providing very concise responses.

Guidelines:
1. Maximum 2-3 sentences
2. Direct and actionable advice
3. Focus on immediate steps
4. Simple, clear language

Format:
Risk: [Brief risk description]
Action: [Specific action to take]

Keep responses extremely concise and practical."""

        human_template = """Alert Type: {alert_type}
Alert Summary: {alert_summary}
Query: {query}

Provide brief response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate_response(self, query: str, alert_type: Optional[str] = None, alert_summary: Optional[str] = None) -> Dict[str, str]:
        try:
            raw_response = self.chain.run(
                query=query,
                alert_type=alert_type or "Not Specified",
                alert_summary=alert_summary or "Not Provided"
            )
            
            return {
                'success': True,
                'response': raw_response.strip()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def format_response(self, response_dict: Dict[str, str]) -> str:
        if not response_dict.get('success', True):
            return f"Error: {response_dict.get('error', 'Unknown error')}"
        return response_dict.get('response', '')

class SitrepAnalyzer:
    """Main analyzer with improved phase handling"""
    def __init__(self):
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.response_generator = SecurityResponseGenerator(OPENAI_API_KEY)
        self.phase_analyzer = QueryPhaseAnalyzer(OPENAI_API_KEY)

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
                is_phase_1 = self.phase_analyzer.analyze_phase(client_query, alert_summary)
                if is_phase_1:
                    response_dict = self.response_generator.generate_response(
                        query=client_query,
                        alert_type=template,
                        alert_summary=alert_summary
                    )
                    query_response = self.response_generator.format_response(response_dict)
                else:
                    query_response = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template,
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    """Streamlit UI setup and main application flow"""
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
    
    st.title("Sitreps Analyzer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        alert_summary = st.text_area("Alert Summary (paste SITREP here)", height=300)

    with col2:
        client_query = st.text_area("Client Query", height=150,
                                  help="Enter the client's question or request here")
    
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
                # Show template and status in header box
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
                        '<div class="response-box">'
                        '<div class="response-header">Query Response:</div>'
                        f'{result.get("query_response")}'
                        '</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
