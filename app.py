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

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        """Analyze sitrep and provide concise, focused response"""
        matching_templates = self.find_matching_template(alert_summary)
        template = matching_templates[0] if matching_templates else "Unknown Template"
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert security analyst. Extract and analyze all relevant fields from the alert summary 
            (such as status, command line, hash, reputation score, geolocation, network protocol, etc.).
            Provide extremely concise, technical responses focused exactly on what was asked."""
        )
        
        if client_query:
            human_template = """
            Alert Summary:
            {alert_summary}
            
            Template Type: {template}
            
            Client Query: {query}
            
            Provide a direct, concise answer focused specifically on the query, incorporating relevant details 
            from any fields found in the alert summary. Be brief and technical.
            """
        else:
            human_template = """
            Alert Summary:
            {alert_summary}
            
            Template Type: {template}
            
            Provide a brief technical analysis of the most critical aspects found in the alert summary.
            Focus on any specific fields or indicators present in the data.
            """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        try:
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            analysis = chain.run(
                template=template,
                alert_summary=alert_summary,
                query=client_query if client_query else ""
            )
            
            return {
                "template": template,
                "analysis": analysis.strip(),
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
            margin-bottom: 30px;
        }
        .analysis-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2ecc71;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .template-match {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .stButton>button {
            background-color: #2a5298;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    analyzer = SitrepAnalyzer()
    
    st.markdown('<p class="main-title">Sitreps Analysis System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        alert_summary = st.text_area(
            "Alert Summary",
            height=300,
            placeholder="Enter the complete alert summary..."
        )

    with col2:
        client_query = st.text_area(
            "Client Query (Optional)",
            height=150,
            placeholder="Enter any specific questions..."
        )
    
    if st.button("Analyze", type="primary"):
        if not alert_summary:
            st.error("Please enter an alert summary to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            result = analyzer.analyze_sitrep(alert_summary, client_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown('<div class="template-match">' +
                          f'<strong>Template:</strong> {result["template"]}' +
                          '</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="analysis-box">' +
                          f'{result["analysis"]}' +
                          '</div>', unsafe_allow_html=True)
                
                combined_analysis = f"""
                # SITREP ANALYSIS REPORT
                
                ## Template
                {result['template']}
                
                ## Analysis
                {result['analysis']}
                """
                
                st.download_button(
                    label="Download Analysis",
                    data=combined_analysis,
                    file_name="sitrep_analysis.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
