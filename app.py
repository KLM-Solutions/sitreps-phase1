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
5. Next Steps: [Additional recommendations or escalation notes]"""

        human_template = """Alert Context: {alert_summary}
Query: {query}

Provide a comprehensive response following the specified format:"""

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
  * Unique implementation details"""

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

class SitrepAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.template_matcher_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        self.response_generator = CrispResponseGenerator(OPENAI_API_KEY)
        self.phase_classifier = PhaseClassifier(OPENAI_API_KEY)
        self.setup_vector_store()
        self.setup_template_matcher()
    
    def setup_vector_store(self):
        """Initialize FAISS vector store with templates"""
        self.vector_store = FAISS.from_texts(SITREP_TEMPLATES, self.embeddings)
    
    def setup_template_matcher(self):
        """Setup the template matching prompt"""
        system_template = """You are a precise security alert template matcher. Your task is to:
        1. Analyze the given security alert
        2. Match it to the most relevant template from the provided list
        3. Return ONLY the exact template name that matches best
        4. If no exact match exists, return the closest matching template

        Focus on key alert characteristics and pattern matching."""

        human_template = """
        AVAILABLE TEMPLATES:
        {templates}

        ALERT TO ANALYZE:
        {alert}

        Return only the best matching template name from the list. No explanation needed."""

        self.template_matcher_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def find_matching_template(self, sitrep_text: str) -> str:
        """Find most similar template using GPT-4o-mini"""
        try:
            chain = LLMChain(llm=self.template_matcher_llm, prompt=self.template_matcher_prompt)
            matched_template = chain.run(
                templates="\n".join(SITREP_TEMPLATES),
                alert=sitrep_text
            ).strip()
            
            if matched_template in SITREP_TEMPLATES:
                return matched_template
            return "Unknown Template"
        except Exception as e:
            print(f"Template matching error: {str(e)}")
            return "Unknown Template"
    
    def extract_fields(self, text: str) -> Dict[str, str]:
        """Extract various fields from the alert summary"""
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
        """Complete sitrep analysis pipeline"""
        template = self.find_matching_template(alert_summary)
        fields = self.extract_fields(alert_summary)
        
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
            "fields": fields,
            "is_phase_1": is_phase_1,
            "query_response": query_response
        }

def main():
    st.set_page_config(page_title="Sitreps Analyzer", layout="wide")
    
    # Styling
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
