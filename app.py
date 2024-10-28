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
        self.vector_store = FAISS.from_texts(SITREP_TEMPLATES, self.embeddings)
    
    def find_matching_template(self, sitrep_text: str, top_k: int = 1) -> List[str]:
        matches = self.vector_store.similarity_search(sitrep_text, k=top_k)
        return [match.page_content for match in matches]

    def generate_alert_analysis(self, alert_summary: str) -> str:
        """Generate general alert analysis"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert security analyst. Analyze the alert summary and extract critical information.
            Only mention status if explicitly present in the alert.
            Focus on technical details and avoid generic statements."""
        )
        
        human_template = """
        Alert Summary:
        {alert_summary}
        
        Rules:
        1. Only mention status if explicitly present
        2. Focus on concrete technical details
        3. Identify specific indicators and patterns
        4. Provide clear, specific insights
        
        Generate a concise technical analysis.
        """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(alert_summary=alert_summary)

    def answer_query(self, alert_summary: str, query: str) -> str:
        """Generate focused response to user query"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert security analyst. 
            Provide a direct, specific answer to the user's query based on the alert details.
            Focus only on information relevant to the query."""
        )
        
        human_template = """
        Alert Summary:
        {alert_summary}
        
        User Query:
        {query}
        
        Rules:
        1. Answer only what was asked
        2. Use only information present in the alert
        3. Be clear and specific
        4. Keep the response concise
        
        Provide a focused answer to the query.
        """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(alert_summary=alert_summary, query=query)

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        """Complete analysis pipeline"""
        try:
            # Generate base analysis
            alert_analysis = self.generate_alert_analysis(alert_summary)
            
            # Generate query response if query exists
            query_response = None
            if client_query:
                query_response = self.answer_query(alert_summary, client_query)
            
            return {
                "alert_analysis": alert_analysis.strip(),
                "query_response": query_response.strip() if query_response else None,
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
        .query-response {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 10px 0;
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
            "Your Query",
            height=150,
            placeholder="What would you like to know about this alert?"
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
                          f'<strong>Alert Analysis:</strong><br>{result["alert_analysis"]}' +
                          '</div>', unsafe_allow_html=True)
                
                if result["has_query"] and result["query_response"]:
                    st.markdown('<div class="query-response">' +
                              f'<strong>Query Response:</strong><br>{result["query_response"]}' +
                              '</div>', unsafe_allow_html=True)
                
                combined_analysis = f"""
                # Alert Analysis Report
                
                ## General Analysis
                {result["alert_analysis"]}
                
                {f"## Query Response\n{result['query_response']}" if result["query_response"] else ""}
                """
                
                st.download_button(
                    label="Download Analysis",
                    data=combined_analysis,
                    file_name="alert_analysis.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
