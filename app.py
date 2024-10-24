import streamlit as st
import os
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")
        st.stop()

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
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Initialize FAISS vector store with templates"""
        self.vector_store = FAISS.from_texts(SITREP_TEMPLATES, self.embeddings)
    
    def find_matching_template(self, sitrep_text: str, top_k: int = 1) -> List[str]:
        """Find most similar template(s) using similarity search"""
        matches = self.vector_store.similarity_search(sitrep_text, k=top_k)
        return [match.page_content for match in matches]
    
    def extract_status(self, text: str) -> str:
        """Extract status information from the alert summary"""
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).strip()
            return status if status else "No status found"
        return "No status found"

    def analyze_status(self, status: str, template: str) -> Dict:
        """Analyze the status code with context from the matching template"""
        if status == "No status found":
            return {"status_analysis": "No status information found in the alert summary."}
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert security analyst specializing in status code analysis.
            Provide precise, technical interpretations of status codes in security contexts."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Analyze this status code in the context of the template:
            Template: {template}
            Status: {status}
            
            Provide a concise technical analysis including:
            1. Status code meaning
            2. Security implications
            3. Relevance to template
            
            Be precise and technical. Max 50 words."""
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        
        try:
            result = chain.run(template=template, status=status)
            return {"status_analysis": result.strip()}
        except Exception as e:
            return {"error": f"Status analysis error: {str(e)}"}

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        """Complete sitrep analysis pipeline with prioritized client query response"""
        matching_templates = self.find_matching_template(alert_summary)
        template = matching_templates[0] if matching_templates else "Unknown Template"
        
        status = self.extract_status(alert_summary)
        status_analysis = self.analyze_status(status, template)
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert security analyst with deep experience in threat detection and incident response.
            Your task is to analyze security alerts and provide clear, actionable insights.
            Format your response using markdown headers and bullet points."""
        )
        
        if client_query:
            human_template = """
            Context:
            Template Type: {template}
            Status: {status}
            Alert Summary:
            {alert_summary}
            
            Client Query: {query}
            
            Provide analysis using this format:
            
            ## Query Response
            
            ### Direct Answer to Client's Question
            [Provide direct, clear answer]
            
            ### Technical Justification
            [Provide technical explanation]
            
            ### Relevant Context
            [Provide important contextual information]
            
            ## Technical Summary
            
            ### Key Findings
            - [Finding 1]
            - [Finding 2]
            - [Finding 3]
            
            ### Critical Indicators
            - [Indicator 1]
            - [Indicator 2]
            - [Indicator 3]
            
            ## Required Actions
            
            ### Immediate Steps
            - [Step 1]
            - [Step 2]
            
            ### Investigation Points
            - [Point 1]
            - [Point 2]
            - [Point 3]
            
            ### Mitigation Measures
            - [Measure 1]
            - [Measure 2]
            - [Measure 3]
            """
        else:
            human_template = """
            Context:
            Template Type: {template}
            Status: {status}
            Alert Summary:
            {alert_summary}
            
            Provide analysis using this format:
            
            ## Technical Summary
            
            ### Key Findings
            - [Finding 1]
            - [Finding 2]
            - [Finding 3]
            
            ### Critical Indicators
            - [Indicator 1]
            - [Indicator 2]
            - [Indicator 3]
            
            ## Required Actions
            
            ### Immediate Steps
            - [Step 1]
            - [Step 2]
            
            ### Investigation Points
            - [Point 1]
            - [Point 2]
            - [Point 3]
            
            ### Mitigation Measures
            - [Measure 1]
            - [Measure 2]
            - [Measure 3]
            """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        try:
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            analysis = chain.run(
                template=template,
                status=status,
                alert_summary=alert_summary,
                query=client_query if client_query else ""
            )
            
            return {
                "template": template,
                "status": status,
                "status_analysis": status_analysis.get("status_analysis", ""),
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
            placeholder="Enter the complete alert summary including status information..."
        )

    with col2:
        st.markdown('<p class="section-header">Client Query</p>', unsafe_allow_html=True)
        client_query = st.text_area(
            "Enter client questions",
            height=150,
            placeholder="Enter any specific questions from the client..."
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
                # Template Match
                st.markdown('<p class="section-header">Matched Template</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="template-match">{result["template"]}</div>', 
                          unsafe_allow_html=True)
                
                # Status Analysis
                st.markdown('<p class="section-header">Status Analysis</p>', unsafe_allow_html=True)
                st.markdown(f'''<div class="status-box">
                    <strong>Status Code:</strong> {result["status"]}<br>
                    <strong>Analysis:</strong> {result["status_analysis"]}
                    </div>''', unsafe_allow_html=True)
                
                # Alert Analysis
                st.markdown('<p class="section-header">Alert Analysis</p>', unsafe_allow_html=True)
                st.markdown(result["analysis"], unsafe_allow_html=True)
                
                # Download button
                combined_analysis = f"""
                # SITREP ANALYSIS REPORT
                
                ## Matched Template
                {result['template']}
                
                ## Status Analysis
                Status Code: {result['status']}
                {result['status_analysis']}
                
                ## Alert Analysis
                {result['analysis']}
                """
                
                st.download_button(
                    label="Download Complete Analysis",
                    data=combined_analysis,
                    file_name="sitrep_analysis_report.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
