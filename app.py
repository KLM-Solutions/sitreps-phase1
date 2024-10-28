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
        """Analyze sitrep and provide focused, relevant response"""
        matching_templates = self.find_matching_template(alert_summary)
        template = matching_templates[0] if matching_templates else "Unknown Template"
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert security analyst. Focus only on fields actually present in the alert summary.
            Do not make assumptions about fields that aren't mentioned.
            Provide extremely concise, specific responses based solely on the available data.
            Avoid generic statements or conclusions not directly supported by the alert details.
            If a specific field is queried but not present in the alert, clearly state its absence."""
        )
        
        if client_query:
            human_template = """
            Alert Details:
            {alert_summary}
            
            Query: {query}
            
            Rules:
            1. Focus only on information explicitly present in the alert
            2. Do not include generic conclusions or assumptions
            3. If queried about missing information, state its absence clearly
            4. Provide direct, specific answers based only on available data
            5. Include only relevant technical details that directly answer the query
            
            Provide a crisp, technical response focusing only on the query and present data.
            """
        else:
            human_template = """
            Alert Details:
            {alert_summary}
            
            Rules:
            1. Focus only on information explicitly present in the alert
            2. Do not include generic conclusions or assumptions
            3. Analyze only the fields actually present
            4. Provide specific technical insights based solely on available data
            
            Generate a concise analysis focusing only on the most critical present indicators.
            """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        try:
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            analysis = chain.run(
                alert_summary=alert_summary,
                query=client_query if client_query else ""
            )
            
            return {
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
            font-size: 32px !important;
            font-weight: bold;
            color: #2a5298;
            padding: 15px 0;
            text-align: center;
            margin-bottom: 20px;
        }
        .analysis-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
    
    st.markdown('<p class="main-title">Alert Analysis System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        alert_summary = st.text_area(
            "Alert Details",
            height=300,
            placeholder="Enter the complete alert details..."
        )

    with col2:
        client_query = st.text_area(
            "Specific Query (Optional)",
            height=150,
            placeholder="Enter your specific question..."
        )
    
    if st.button("Analyze", type="primary"):
        if not alert_summary:
            st.error("Please enter alert details to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            result = analyzer.analyze_sitrep(alert_summary, client_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown('<div class="analysis-box">' +
                          f'{result["analysis"]}' +
                          '</div>', unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Analysis",
                    data=result["analysis"],
                    file_name="alert_analysis.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
