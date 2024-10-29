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
    """Template matcher for alerts"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_matcher()
        self.templates = SITREP_TEMPLATES

    def setup_matcher(self):
        system_template = "Given the alert text, return the exact matching template name. If no match, return 'Unknown Template'."

        human_template = """Templates:
{templates}

Alert Text:
{alert_text}

Matching template:"""
        
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
        system_template = """Determine if the security query can be answered with general knowledge.

Return EXACTLY "PHASE_1" if the query:
- Asks about general best practices
- Compares security approaches
- Seeks standard recommendations
- Asks about effectiveness of security controls
- Questions about tool capabilities
- General mitigation strategies
- Basic security concepts

Return EXACTLY "NOT_PHASE_1" only if the query needs:
- Specific log analysis
- Customer environment details
- Technical debugging
- Incident investigation

Example PHASE_1 queries:
"Is SSL better than IP blocking?"
"What's the best way to handle this type of alert?"
"Which security measure is more effective?"
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
    """Generator for security response"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_chain()

    def setup_chain(self):
        system_template = """Provide extremely concise security responses.

Rules:
1. Maximum 2-3 sentences
2. Be direct and actionable
3. Focus on immediate steps
4. Simple, clear language

Format:
Risk: [Brief risk description]
Action: [Specific action to take]"""

        human_template = """Alert: {alert_type}
Context: {context}
Query: {query}

Response:"""

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

    def generate_response(self, query: str, alert_type: str = None, context: str = None) -> Dict[str, str]:
        try:
            raw_response = self.chain.run(
                query=query,
                alert_type=alert_type or "Not Specified",
                context=context or "No additional context"
            )
            
            structured_response = self.parse_response(raw_response)
            return {'success': True, 'structured_response': structured_response}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def format_response(self, response_dict: Dict[str, str]) -> str:
        if not response_dict.get('success', True):
            return f"Error: {response_dict.get('error', 'Unknown error')}"
            
        components = response_dict.get('structured_response', {})
        formatted_parts = []
        
        if components.get('risk'):
            formatted_parts.append(f"Risk: {components['risk']}")
        if components.get('action'):
            formatted_parts.append(f"Action: {components['action']}")
            
        return "\n".join(formatted_parts)

class SitrepAnalyzer:
    """Main analysis coordinator"""
    def __init__(self):
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.phase_analyzer = QueryPhaseAnalyzer(OPENAI_API_KEY)
        self.response_generator = SecurityResponseGenerator(OPENAI_API_KEY)

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.template_matcher.match_template(alert_summary)
            response = None
            
            if client_query:
                is_phase_1 = self.phase_analyzer.analyze_phase(client_query, alert_summary)
                if is_phase_1:
                    response_dict = self.response_generator.generate_response(
                        query=client_query,
                        alert_type=template,
                        context=alert_summary
                    )
                    response = self.response_generator.format_response(response_dict)
                else:
                    response = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template,
                "response": response
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    """Streamlit UI setup"""
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
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
            elif result.get("response"):
                st.markdown(
                    '<div class="response-box">'
                    '<div class="response-header">Query Response:</div>'
                    f'{result.get("response")}'
                    '</div>',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
