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
        system_template = """You are an AI assistant specialized in handling general customer inquiries about cybersecurity and IT best practices. Your role is to:
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
When responding, follow this structure:
1. Query Type: [GENERAL or SPECIFIC]
2. Confidence: [HIGH or MEDIUM or LOW]
3. Response Category: [Best Practice/Mitigation/Recommendation/Prevention]
4. Response: [Your detailed response]
5. Next Steps: [Additional recommendations or escalation notes]

# Response Guidelines
- For GENERAL queries:
  * Provide industry-standard recommendations
  * Include relevant security frameworks or standards
  * Offer clear, actionable steps
  * Link to official documentation when applicable
  * Keep responses vendor-neutral unless specifically asked

- For SPECIFIC queries:
  * Indicate need for Customer Analyst review
  * Explain why the query requires specialized attention
  * Note any specific information needed for analysis

# Important Notes:
- Always prioritize security best practices
- Maintain professional tone
- Be clear when escalation is needed
- Avoid making assumptions about customer environment
- Stay within scope of general recommendations

End your responses with a clear indication of whether follow-up with a Customer Analyst is recommended."""

        human_template = """Context Information:
Alert Type: {alert_type}
Alert Summary: {alert_summary}
Client Query: {query}

Generate a comprehensive response following the specified format and guidelines."""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def parse_response(self, response: str) -> Dict[str, str]:
        components = {
            'query_type': None,
            'confidence': None,
            'response_category': None,
            'response': None,
            'next_steps': None
        }
        
        query_type_match = re.search(r"Query Type:\s*(.*?)(?:\n|$)", response)
        confidence_match = re.search(r"Confidence:\s*(.*?)(?:\n|$)", response)
        category_match = re.search(r"Response Category:\s*(.*?)(?:\n|$)", response)
        response_match = re.search(r"Response:(.*?)(?=Next Steps:|$)", response, re.DOTALL)
        next_steps_match = re.search(r"Next Steps:(.*?)$", response, re.DOTALL)
        
        if query_type_match:
            components['query_type'] = query_type_match.group(1).strip()
        if confidence_match:
            components['confidence'] = confidence_match.group(1).strip()
        if category_match:
            components['response_category'] = category_match.group(1).strip()
        if response_match:
            components['response'] = response_match.group(1).strip()
        if next_steps_match:
            components['next_steps'] = next_steps_match.group(1).strip()
            
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
            structured_response['raw_response'] = raw_response
            
            return structured_response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query_type': None,
                'confidence': None,
                'response_category': None,
                'response': None,
                'next_steps': None,
                'raw_response': None
            }

    def format_response(self, response_dict: Dict[str, str]) -> str:
        if not response_dict.get('success', True):
            return f"Error generating response: {response_dict.get('error', 'Unknown error')}"
            
        formatted_parts = []
        
        if response_dict.get('query_type'):
            formatted_parts.append(f"Query Type: {response_dict['query_type']}")
        
        if response_dict.get('confidence'):
            formatted_parts.append(f"Confidence: {response_dict['confidence']}")
            
        if response_dict.get('response_category'):
            formatted_parts.append(f"Response Category: {response_dict['response_category']}")
            
        if response_dict.get('response'):
            formatted_parts.append(f"Response:\n{response_dict['response']}")
            
        if response_dict.get('next_steps'):
            formatted_parts.append(f"Next Steps:\n{response_dict['next_steps']}")
            
        return "\n\n".join(formatted_parts)

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
                        '<strong>RESPONSE:</strong><br>' +
                        f'{result["query_response"]}' +
                        '</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
