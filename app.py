import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import re
from typing import Dict, Optional, List
import numpy as np

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

class FAISSTemplateMatcher:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.templates = SITREP_TEMPLATES
        self.initialize_faiss()

    def initialize_faiss(self):
        try:
            self.vector_store = FAISS.load_local("faiss_index", self.embeddings)
            print("Loaded existing FAISS index")
        except:
            print("Creating new FAISS index")
            texts = [{"template": t, "content": t} for t in self.templates]
            self.vector_store = FAISS.from_texts(
                texts=[str(t) for t in self.templates],
                embedding=self.embeddings,
                metadatas=texts
            )
            self.vector_store.save_local("faiss_index")

    def match_template(self, alert_text: str, similarity_threshold: float = 0.7) -> Dict[str, any]:
        try:
            results = self.vector_store.similarity_search_with_score(alert_text, k=1)
            
            if not results:
                return {
                    "template": "Unknown Template",
                    "similarity": 0.0,
                    "status": "No match found"
                }
            
            doc, score = results[0]
            similarity = 1.0 - score
            
            if similarity < similarity_threshold:
                return {
                    "template": "Unknown Template",
                    "similarity": similarity,
                    "status": f"Best match below threshold ({similarity:.2f})"
                }
            
            return {
                "template": doc.page_content,
                "similarity": similarity,
                "status": "Match found"
            }
            
        except Exception as e:
            print(f"Template matching error: {str(e)}")
            return {
                "template": "Unknown Template",
                "similarity": 0.0,
                "status": f"Error: {str(e)}"
            }

class PhaseClassifier:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )

    def classify(self, alert_summary: str, query: str) -> bool:
        messages = [
            {"role": "system", "content": "Determine if this query is Phase 1 (general security question) or needs analyst review. Return ONLY 'PHASE_1' or 'NOT_PHASE_1'."},
            {"role": "user", "content": f"Alert: {alert_summary}\nQuery: {query}"}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content.strip() == "PHASE_1"
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return False

class SecurityResponseGenerator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )

    def generate_response(self, query: str, alert_type: Optional[str] = None, alert_summary: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": "Provide a direct, concise response to this security query. If it's too specific or requires log analysis, indicate that analyst review is needed."},
            {"role": "user", "content": f"Alert Type: {alert_type or 'Not Specified'}\nAlert Summary: {alert_summary or 'Not Provided'}\nQuery: {query}"}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class SitrepAnalyzer:
    def __init__(self):
        self.template_matcher = FAISSTemplateMatcher(OPENAI_API_KEY)
        self.response_generator = SecurityResponseGenerator(OPENAI_API_KEY)
        self.phase_classifier = PhaseClassifier(OPENAI_API_KEY)

    def extract_status(self, text: str) -> Optional[str]:
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            match_result = self.template_matcher.match_template(alert_summary)
            status = self.extract_status(alert_summary)
            
            query_response = None
            is_phase_1 = False
            
            if client_query:
                is_phase_1 = self.phase_classifier.classify(alert_summary, client_query)
                if is_phase_1:
                    query_response = self.response_generator.generate_response(
                        query=client_query,
                        alert_type=match_result["template"],
                        alert_summary=alert_summary
                    )
                else:
                    query_response = "This query requires analyst review due to its specific nature."
            
            return {
                "template": match_result["template"],
                "similarity": match_result["similarity"],
                "match_status": match_result["status"],
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

def main():
    st.set_page_config(page_title="Alert Analyzer", layout="wide")
    
    # Simple CSS for clean styling
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
                # Show template and status
                if result["template"] != "Unknown Template":
                    st.markdown(
                        f'<div class="header-box">' +
                        f'Template: {result["template"]}<br>' +
                        f'Similarity: {result["similarity"]:.2f}<br>' +
                        f'Status: {result["match_status"]}' +
                        '</div>',
                        unsafe_allow_html=True
                    )
                
                if result.get("status"):
                    st.info(f"Alert Status: {result['status']}")
                
                # Show response
                if result.get("query_response"):
                    st.markdown(
                        f'<div class="response-box">{result["query_response"]}</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
