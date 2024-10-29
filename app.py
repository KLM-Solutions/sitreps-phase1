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
from typing import Dict, Optional, List
import re

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
        system_template = """You are an AI assistant specialized in handling general customer inquiries about cybersecurity and IT best practices. Provide direct, single-sentence responses focused on actionable insights or recommendations.

For queries about:
* Industry best practices
* General recommendations
* Standard mitigation strategies
* Common security guidelines
* Prevention techniques
* Educational information
* High-level process explanations

Provide a clear, technical response that directly addresses the query in one sentence.

Format your response as:
User Response: [Single concise technical response]"""

        human_template = """Alert Context: {alert_summary}
Query: {query}

Provide a single concise response that directly addresses the security concern or recommendation:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate(self, alert_summary: str, query: str) -> str:
        response = self.chain.run(alert_summary=alert_summary, query=query).strip()
        if not response.startswith("User Response:"):
            response = f"User Response: {response}"
        return response

class PhaseClassifier:
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

Example PHASE_1 queries:
- "What are best practices for handling this type of alert?"
- "Is IP blocking effective for this threat?"
- "What's the recommended way to respond?"
- "Should we enable additional monitoring?"
- "What threshold should we set?"

Example NOT_PHASE_1 queries:
- "Can you check our specific logs?"
- "Why is this IP appearing repeatedly?"
- "What's causing these errors in our system?"
- "Can you investigate this specific incident?"
- "What's wrong with our current setup?"

Remember: If the query asks for general guidance or best practices, it's PHASE_1 even if it references specific tools or technologies."""

        human_template = """Alert Context: {alert_summary}
Query: {query}

Based on the classification rules, is this a PHASE_1 or NOT_PHASE_1 query?
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

# Rest of the classes/functions remain the same as your previous code
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
        self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
        self.response_generator = CrispResponseGenerator(OPENAI_API_KEY)
        self.phase_classifier = PhaseClassifier(OPENAI_API_KEY)

    def extract_fields(self, text: str) -> Dict[str, str]:
        fields = {}
        field_patterns = {
            'status': r"Status:([^\n]*)",
            'command': r"Command:([^\n]*)",
            'ip': r"IP:([^\n]*)",
            'protocol': r"Protocol:([^\n]*)",
            'hash': r"Hash:([^\n]*)",
            'source': r"Source:([^\n]*)",
            'destination': r"Destination:([^\n]*)",
            'timestamp': r"Timestamp:([^\n]*)",
            'severity': r"Severity:([^\n]*)",
            'reputation': r"Reputation:([^\n]*)",
            'geolocation': r"Geolocation:([^\n]*)"
        }
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field] = match.group(1).strip()
        return fields

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            template = self.template_matcher.match_template(alert_summary)
            fields = self.extract_fields(alert_summary)
            
            query_response = None
            is_phase_1 = False
            
            if client_query:
                is_phase_1 = self.phase_classifier.classify(alert_summary, client_query)
                if is_phase_1:
                    query_response = self.response_generator.generate(alert_summary, client_query)
                else:
                    query_response = "User Response: Requires analyst review - escalating for detailed investigation."
            
            return {
                "template": template,
                "fields": fields,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
            return {"error": f"Error generating analysis: {str(e)}"}

# The main() function remains exactly the same as your previous code
def main():
    st.set_page_config(page_title="Sitreps Analyzer", layout="wide")
    
    # Styling remains the same
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px !important;
            font-weight: bold;
            background: linear-gradient(45deg, #1e3c72, #2a5298);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px 0;
            text-align: center;
            letter-spacing: 2px;
            margin-bottom: 30px;
        }
        .section-header {
            font-size: 24px !important;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
            padding: 10px 0;
            border-bottom: 2px solid #eee;
        }
        .analysis-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2ecc71;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .fields-box {
            background-color: #f7f9fc;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #e67e22;
            margin: 10px 0;
        }
        .template-match {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
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
    
    st.markdown('<p class="main-title">Sitreps Analysis System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">Alert Summary</p>', unsafe_allow_html=True)
        alert_summary = st.text_area(
            "Paste your security alert details here",
            height=300,
            placeholder="Enter the complete alert summary..."
        )

    with col2:
        st.markdown('<p class="section-header">Client Query</p>', unsafe_allow_html=True)
        client_query = st.text_area(
            "Enter client questions",
            height=150,
            placeholder="Enter any specific questions..."
        )
    
    if st.button("Generate Analysis", type="primary"):
        if not alert_summary:
            st.error("Please enter an alert summary to analyze.")
            return
        
        with st.spinner("Analyzing security alert..."):
            analyzer = SitrepAnalyzer()
            result = analyzer.analyze_sitrep(alert_summary, client_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display template match
                st.markdown('<p class="section-header">Matched Template</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="template-match">{result["template"]}</div>', 
                          unsafe_allow_html=True)
                
                # Display extracted fields if present
                if result["fields"]:
                    st.markdown('<p class="section-header">Extracted Fields</p>', unsafe_allow_html=True)
                    st.markdown('<div class="fields-box">', unsafe_allow_html=True)
                    for field, value in result["fields"].items():
                        st.markdown(f"**{field.title()}:** {value}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display analysis/response if present
                if result.get("query_response"):
                    st.markdown('<p class="section-header">Analysis</p>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="response-box">' +
                        f'{result["query_response"]}' +
                        '</div>',
                        unsafe_allow_html=True
                    )
                
                # Download button
                combined_analysis = f"""
                # SITREP ANALYSIS REPORT
                
                ## Matched Template
                {result['template']}
                
                ## Extracted Fields
                {result['fields']}
                
                ## Analysis
                {result.get('query_response', 'No analysis generated.')}
                """
                
                st.download_button(
                    label="Download Analysis",
                    data=combined_analysis,
                    file_name="sitrep_analysis_report.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
