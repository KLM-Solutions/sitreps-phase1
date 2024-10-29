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

class TemplateMatcher:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.templates = SITREP_TEMPLATES
        self.setup_matcher()

    def setup_matcher(self):
        system_template = """You are a specialized security alert template matcher with advanced pattern recognition capabilities.

KEY MATCHING CRITERIA:
1. Authentication Patterns:
   - Kerberos authentication events
   - Sign-in activities
   - Access patterns

2. Traffic Analysis:
   - Anomalous traffic (internal/external)
   - Size/packet/session anomalies
   - Network behavior patterns

3. IP-Based Threats:
   - Blacklisted IPs
   - TOR/VPN endpoints
   - Known malicious sources

4. Protocol Indicators:
   - DNS queries
   - TLS traffic
   - NTP communications

5. Service-Specific Alerts:
   - Bot activities
   - Scanning detection
   - Automation tool usage

MATCHING RULES:
1. Exact matches take priority
2. Consider semantic equivalence for near-matches
3. Pattern-based matching for similar alerts
4. Context-aware template selection

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

    def extract_status(self, text: str) -> Optional[str]:
        """Extract status information from alert text using enhanced pattern matching"""
        patterns = [
            r"Status:([^\n]*)",
            r"Current Status:([^\n]*)",
            r"Alert Status:([^\n]*)",
            r"Status\s*:\s*([^:\n]*)",
        ]
        
        for pattern in patterns:
            status_match = re.search(pattern, text, re.IGNORECASE)
            if status_match:
                return status_match.group(1).strip()
        return None

    def match_template(self, alert_text: str) -> str:
        """Match alert text to template with improved accuracy"""
        try:
            result = self.matcher_chain.run(
                templates="\n".join(self.templates),
                alert_text=alert_text
            ).strip()
            return result if result in self.templates else "Unknown Template"
        except Exception as e:
            return "Unknown Template"

class QueryClassifier:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_classifier()

    def setup_classifier(self):
        system_template = """You are a security query classifier focused solely on determining if user queries are general or specific in nature.

CLASSIFY AS PHASE_1 IF THE QUERY:
- Asks about general security best practices
- Requests standard mitigation strategies
- Seeks common security guidelines
- Asks about industry-standard approaches
- Requires general technical explanations
- Asks about typical configurations
- Seeks understanding of common alerts
- Requests general prevention advice

CLASSIFY AS PHASE_2 IF THE QUERY:
- Mentions specific IP addresses
- References particular log entries
- Asks about custom configurations
- Requires analysis of specific incidents
- Mentions unique system setups
- Requests investigation of particular events
- Needs specific infrastructure details
- Involves customer-specific data

RESPONSE FORMAT:
Return ONLY 'PHASE_1' or 'PHASE_2' based on the query type."""

        human_template = """USER QUERY: {query}

CLASSIFY AS PHASE_1 OR PHASE_2:"""

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
        - Include brief technical justification if needed"""

        human_template = """Context: {alert_summary}
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

class SitrepAnalyzer:
    def __init__(self):
        try:
            self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
            self.response_generator = CrispResponseGenerator(OPENAI_API_KEY)
            self.query_classifier = QueryClassifier(OPENAI_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {str(e)}")
            st.stop()

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.template_matcher.match_template(alert_summary)
            status = self.template_matcher.extract_status(alert_summary)
            
            query_response = None
            is_phase_1 = False
            
            if client_query:
                is_phase_1 = self.query_classifier.classify(client_query)
                if is_phase_1:
                    query_response = self.response_generator.generate(alert_summary, client_query)
                else:
                    query_response = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template,
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}

def main():
    st.set_page_config(page_title="Security Alert Analyzer", layout="wide")
    
    # Professional minimal styling
    st.markdown("""
        <style>
        .alert-box {
            padding: 20px;
            border-radius: 5px;
            background-color: #f8f9fa;
            margin: 10px 0;
            border-left: 4px solid #0d6efd;
        }
        .alert-header {
            font-size: 16px;
            font-weight: 500;
            color: #1a1a1a;
            margin-bottom: 10px;
        }
        .alert-content {
            font-size: 15px;
            color: #2c3e50;
            line-height: 1.5;
        }
        .phase-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 15px;
        }
        .phase-1 {
            background-color: #e3f2fd;
            color: #0d47a1;
        }
        .phase-2 {
            background-color: #ffebee;
            color: #c62828;
        }
        </style>
        """, unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            alert_summary = st.text_area("Summary Analysis", height=300, 
                help="Enter the alert summary for analysis")

        with col2:
            client_query = st.text_area("User Query (Optional)", height=150,
                help="Enter any specific questions about the alert")
        
        if st.button("Analyze", type="primary"):
            if not alert_summary:
                st.error("Please enter alert details for analysis.")
                return
            
            analyzer = SitrepAnalyzer()
            with st.spinner("Processing analysis..."):
                result = analyzer.analyze_sitrep(alert_summary, client_query)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display Analysis Results
                    analysis_content = []
                    
                    if result["template"] != "Unknown Template":
                        analysis_content.append(f"Template Match: {result['template']}")
                    
                    if result.get("status"):
                        analysis_content.append(f"Status: {result['status']}")
                    
                    if analysis_content:
                        st.markdown(
                            '<div class="alert-box">'
                            '<div class="alert-header">Analysis Results</div>'
                            '<div class="alert-content">' +
                            '<br>'.join(analysis_content) +
                            '</div></div>',
                            unsafe_allow_html=True
                        )
                    
                    # Display Query Response
                    if result.get("query_response"):
                        phase_class = "phase-1" if result["is_phase_1"] else "phase-2"
                        phase_text = "Phase 1 - Automated Response" if result["is_phase_1"] else "Phase 2 - Analyst Review Required"
                        
                        st.markdown(
                            f'<div class="phase-badge {phase_class}">{phase_text}</div>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
                            '<div class="alert-box">'
                            '<div class="alert-header">Response</div>'
                            '<div class="alert-content">' +
                            result["query_response"] +
                            '</div></div>',
                            unsafe_allow_html=True
                        )
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
