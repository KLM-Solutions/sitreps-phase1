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
from typing import Dict, Optional

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

class QueryPhaseAnalyzer:
    """Dedicated analyzer for determining query phase"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_analyzer()

    def setup_analyzer(self):
        system_template = """You are a query phase analyzer for security questions. 
Your only job is to determine if a question can be answered with general security knowledge.

RESPOND EXACTLY in this format:
PHASE: [1 or 2]
REASON: [One brief sentence why]

Phase 1 Queries (Return PHASE: 1):
- General best practices
- Security tool comparisons
- Standard configurations
- Common mitigation strategies
- Technology effectiveness questions
- Security measure comparisons
- Basic security concepts
- General recommendations
- Common security protocols
- Standard security features

Phase 2 Queries (Return PHASE: 2):
- Specific log analysis
- Environment-specific issues
- Custom configurations
- Technical debugging
- Specific incident details
- Performance tuning
- Custom architecture
- Specific metrics analysis
"""

        human_template = """Query: {query}
Context: {context}

Determine query phase:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def analyze_phase(self, query: str, context: str = "") -> Dict[str, str]:
        try:
            response = self.chain.run(query=query, context=context)
            phase_match = re.search(r"PHASE:\s*(\d)", response)
            reason_match = re.search(r"REASON:\s*(.+)", response)
            
            return {
                "phase": phase_match.group(1) if phase_match else "2",
                "reason": reason_match.group(1).strip() if reason_match else "Could not determine reason",
                "raw_response": response
            }
        except Exception as e:
            return {"phase": "2", "reason": f"Error in analysis: {str(e)}"}

class SecurityResponseGenerator:
    """Response generator for security queries"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_chain()

    def setup_chain(self):
        system_template = """You are a security expert providing brief, actionable responses.

Guidelines:
1. Maximum 3 sentences
2. Focus on immediate action
3. Clear, direct language
4. Include a recommendation

Format your response as:
ANALYSIS: [One sentence analysis]
RECOMMENDATION: [Clear action to take]"""

        human_template = """Alert Type: {alert_type}
Context: {context}
Query: {query}

Provide response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate_response(self, query: str, alert_type: str = None, context: str = None) -> Dict[str, str]:
        try:
            response = self.chain.run(
                query=query,
                alert_type=alert_type or "Not Specified",
                context=context or "No additional context"
            )
            return {'success': True, 'response': response.strip()}
        except Exception as e:
            return {'success': False, 'error': str(e)}

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
        system_template = """Match the alert to the exact template name. Return only the template name."""

        human_template = """Templates:
{templates}

Alert: {alert_text}

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

class SitrepAnalyzer:
    """Main analyzer with phase-based response handling"""
    def __init__(self):
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.phase_analyzer = QueryPhaseAnalyzer(OPENAI_API_KEY)
        self.response_generator = SecurityResponseGenerator(OPENAI_API_KEY)

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.template_matcher.match_template(alert_summary)
            response_info = None
            phase_info = None
            
            if client_query:
                phase_result = self.phase_analyzer.analyze_phase(client_query, alert_summary)
                phase_info = phase_result
                
                if phase_result["phase"] == "1":
                    response = self.response_generator.generate_response(
                        query=client_query,
                        alert_type=template,
                        context=alert_summary
                    )
                    response_info = response.get('response') if response.get('success') else None
                else:
                    response_info = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template,
                "phase_info": phase_info,
                "response": response_info
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    """Streamlit UI with improved response display"""
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .template-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .phase-box {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #1976D2;
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
                
                # Show phase information
                if result.get("phase_info"):
                    phase_text = "Phase 1 - Automated Response" if result["phase_info"]["phase"] == "1" else "Phase 2 - Requires Analyst"
                    st.markdown(
                        f'<div class="phase-box">'
                        f'<strong>Query Classification:</strong> {phase_text}<br>'
                        f'<strong>Reason:</strong> {result["phase_info"]["reason"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Show response
                if result.get("response"):
                    st.markdown(
                        '<div class="response-box">'
                        '<div class="response-header">Query Response:</div>'
                        f'{result["response"]}'
                        '</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
