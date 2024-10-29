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
    "Malware IP"
]

class SitrepAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # Separate GPT-4o-mini model for template matching
        self.template_matcher_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        # Original model for analysis
        self.analysis_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
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
            
            # Verify the template exists in our list
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
        """Complete sitrep analysis pipeline with prioritized client query response"""
        template = self.find_matching_template(alert_summary)
        fields = self.extract_fields(alert_summary)
        
        # Modified system message for more concise analysis
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a precise security analyst focused on delivering ultra-concise insights. Your analysis must be:

            1. Bullet-pointed only
            2. Maximum 3-4 words per point
            3. Only critical findings
            4. No explanations or context
            5. Action-oriented commands
            6. No repetition of alert data

            Use this structure:
            • [Threat Level]: High/Medium/Low
            • [Finding]: Key security issue
            • [Impact]: Business risk
            • [Action]: Required response"""
        )
        
        if client_query:
            # Modified human template for queries to be more concise
            human_template = """
            Alert: {alert_summary}
            Fields: {fields}
            Query: {query}
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        try:
            chain = LLMChain(llm=self.analysis_llm, prompt=chat_prompt)
            analysis = chain.run(
                template=template,
                alert_summary=alert_summary,
                fields=fields,
                query=client_query if client_query else ""
            )
            
            return {
                "template": template,
                "fields": fields,
                "analysis": analysis,
                "has_query": bool(client_query)
            }
        except Exception as e:
            return {"error": f"Error generating analysis: {str(e)}"}

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
        .subsection-header {
            font-size: 20px !important;
            font-weight: bold;
            color: #34495e;
            margin-top: 15px;
            padding: 8px 0;
        }
        .analysis-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2ecc71;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .analysis-section {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .client-query-response {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
        }
        .template-match {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .status-box {
            background-color: #fff3e6;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .fields-box {
            background-color: #f7f9fc;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #e67e22;
            margin: 10px 0;
        }
        .bullet-point {
            margin-left: 20px;
            position: relative;
        }
        .bullet-point:before {
            content: "•";
            position: absolute;
            left: -15px;
            color: #3498db;
        }
        h2 {
            font-size: 24px !important;
            font-weight: bold !important;
            color: #2c3e50 !important;
            margin-top: 25px !important;
            margin-bottom: 15px !important;
        }
        h3 {
            font-size: 20px !important;
            font-weight: bold !important;
            color: #34495e !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }
        ul {
            margin-left: 20px !important;
            margin-bottom: 15px !important;
        }
        li {
            margin-bottom: 8px !important;
        }
        .stButton>button {
            background-color: #2a5298;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stTextArea>div>div>textarea {
            border-radius: 5px;
            border-color: #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    analyzer = SitrepAnalyzer()
    
    # Main interface
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
                
                # Display analysis
                st.markdown('<p class="section-header">Analysis</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="analysis-box">{result["analysis"]}</div>', 
                          unsafe_allow_html=True)

if __name__ == "__main__":
    main()
