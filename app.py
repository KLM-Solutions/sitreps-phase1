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
        system_template = """You are a cybersecurity expert providing concise responses. Focus on:
1. Direct actionable guidance
2. Critical information only
3. Maximum 2-3 sentences
4. No technical jargon unless necessary

Key Rules:
- Be extremely concise
- Prioritize immediate actions
- Only include essential details
- Use simple language

Format your response with ONLY:
Risk: [One sentence risk description]
Action: [One sentence immediate action]"""

        human_template = """Context Information:
Alert Type: {alert_type}
Alert Summary: {alert_summary}
Client Query: {query}

Provide a concise response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def parse_response(self, response: str) -> Dict[str, str]:
        components = {
            'risk': None,
            'action': None
        }
        
        risk_match = re.search(r"Risk:(.*?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:(.*?)$", response, re.DOTALL)
        
        if risk_match:
            components['risk'] = risk_match.group(1).strip()
        if action_match:
            components['action'] = action_match.group(1).strip()
            
        return components

    def generate_response(self, query: str, alert_type: Optional[str] = None, alert_summary: Optional[str] = None) -> Dict[str, str]:
        try:
            raw_response = self.chain.run(
                query=query,
                alert_type=alert_type or "Not Specified",
                alert_summary=alert_summary or "Not Provided"
            )
            
            structured_response = self.parse_response(raw_response)
            structured_response['success'] = True
            return structured_response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def format_response(self, response_dict: Dict[str, str]) -> str:
        if not response_dict.get('success', True):
            return f"Error: {response_dict.get('error', 'Unknown error')}"
            
        formatted_parts = []
        
        if response_dict.get('risk'):
            formatted_parts.append(f"Risk: {response_dict['risk']}")
        if response_dict.get('action'):
            formatted_parts.append(f"Action: {response_dict['action']}")
            
        return "\n".join(formatted_parts)

class PhaseClassifier:
    """Classifier for determining if queries can be handled in Phase 1"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_classifier()

    def setup_classifier(self):
        system_template = """You are a security query classifier specialized in analyzing customer queries.

# Classification Rules
- CLASSIFY AS PHASE_1 when query asks for:
  * Industry best practices
  * General recommendations
  * Standard mitigation strategies
  * Common security guidelines
  * Prevention techniques
  * General configuration advice
  * Effectiveness of security measures
  * Comparison of security approaches

- CLASSIFY AS NOT_PHASE_1 when query involves:
  * Specific customer logs analysis
  * Custom configurations review
  * System-specific troubleshooting
  * Detailed technical debugging
  * Customer-specific setups
  * Unique implementation details

# Response Format
RESPOND ONLY with exactly one of these two options:
- "PHASE_1" for general queries
- "NOT_PHASE_1" for specific queries"""

        human_template = """Alert Context: {alert_summary}
Query: {query}

Classify if this query can be answered with general security knowledge. 
Respond ONLY with PHASE_1 or NOT_PHASE_1:"""

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

class SitrepAnalyzer:
    """Main class for coordinating alert analysis"""
    def __init__(self):
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.response_generator = SecurityResponseGenerator(OPENAI_API_KEY)
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
                is_phase_1 = self.phase_classifier.classify(alert_summary, client_query)
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
    
    st.title("Security Alert Analysis System")
    
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
                        '<div class="response-box">' +
                        f'{result["query_response"]}' +
                        '</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
