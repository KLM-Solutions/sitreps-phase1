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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """FAISS-based template matching system"""
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.templates = SITREP_TEMPLATES
        self.setup_vector_store()

    def setup_vector_store(self):
        """Initialize FAISS vector store with templates"""
        try:
            # Create embeddings for templates
            template_embeddings = self.embeddings.embed_documents(self.templates)
            self.vector_store = FAISS.from_embeddings(
                template_embeddings,
                self.templates,
                self.embeddings
            )
            logger.info("FAISS vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {str(e)}")
            raise

    def find_similar_templates(self, alert_text: str, k: int = 3) -> List[Dict]:
        """Find most similar templates using FAISS"""
        try:
            # Get similar templates with scores
            similar_templates = self.vector_store.similarity_search_with_score(
                alert_text,
                k=k
            )
            
            # Format results
            results = []
            for template, score in similar_templates:
                results.append({
                    "template": template,
                    "similarity_score": 1 - score  # Convert distance to similarity
                })
            
            return results
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return []

class TemplateMatcher:
    """LLM-based template verification"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_matcher()

    def setup_matcher(self):
        system_template = """You are a security alert template verifier.
        Given a list of potential template matches from FAISS and an alert summary,
        verify which template is the most accurate match.
        
        Response format:
        {
            "best_match": "template name",
            "confidence": <0-100>,
            "reason": "brief explanation"
        }"""

        human_template = """Alert Summary:
        {alert_text}
        
        Potential Templates (with similarity scores):
        {potential_matches}
        
        Verify the best matching template:"""
        
        self.matcher_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def verify_template(self, alert_text: str, potential_matches: List[Dict]) -> Dict:
        try:
            matches_str = "\n".join([
                f"- {m['template']} (similarity: {m['similarity_score']:.2f})"
                for m in potential_matches
            ])
            
            result = self.matcher_chain.run(
                alert_text=alert_text,
                potential_matches=matches_str
            )
            
            return eval(result)
        except Exception as e:
            logger.error(f"Template verification error: {str(e)}")
            return {
                "best_match": "Unknown Template",
                "confidence": 0,
                "reason": f"Verification failed: {str(e)}"
            }

class QueryClassifier:
    """Dedicated classifier for analyzing user queries only"""
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

        human_template = "USER QUERY: {query}"

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
        7. Avoid generic advice"""

        human_template = """Context: {alert_summary}
        Query: {query}
        Identified Template: {template}
        
        Provide direct technical response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate(self, alert_summary: str, query: str, template: str) -> str:
        return self.chain.run(
            alert_summary=alert_summary,
            query=query,
            template=template
        ).strip()

class SitrepAnalyzer:
    def __init__(self):
        try:
            self.faiss_matcher = FAISSTemplateMatcher(OPENAI_API_KEY)
            self.template_matcher = TemplateMatcher(OPENAI_API_KEY)
            self.response_generator = CrispResponseGenerator(OPENAI_API_KEY)
            self.query_classifier = QueryClassifier(OPENAI_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {str(e)}")
            st.stop()

    def extract_status(self, text: str) -> Optional[str]:
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            # Step 1: Find similar templates using FAISS
            similar_templates = self.faiss_matcher.find_similar_templates(alert_summary)
            
            if not similar_templates:
                return {"error": "No matching templates found"}
            
            # Step 2: Verify best template match using LLM
            template_result = self.template_matcher.verify_template(
                alert_summary,
                similar_templates
            )
            
            status = self.extract_status(alert_summary)
            
            query_response = None
            is_phase_1 = False
            
            if client_query:
                is_phase_1 = self.query_classifier.classify(client_query)
                if is_phase_1:
                    query_response = self.response_generator.generate(
                        alert_summary,
                        client_query,
                        template_result["best_match"]
                    )
                else:
                    query_response = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                "template": template_result["best_match"],
                "confidence": template_result["confidence"],
                "reason": template_result["reason"],
                "similar_templates": similar_templates,
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {"error": f"Analysis error: {str(e)}"}

def main():
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
        .phase-indicator {
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .phase-1 {
            background-color: #d4edda;
            color: #155724;
        }
        .phase-2 {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
        """, unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            alert_summary = st.text_area("Alert Summary", height=300)

        with col2:
            client_query = st.text_area("User Query", height=150)
        
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
                    st.markdown(
                        f'<div class="header-box">'
                        f'<strong>Matched Template:</strong> {result["template"]}<br>'
                        f'<strong>Confidence:</strong> {result["confidence"]}%<br>'
                        f'<strong>Reason:</strong> {result["reason"]}<br>'
                        f'<strong>Status:</strong> {result.get("status", "Not specified")}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show similar templates
                    st.markdown("### Similar Templates")
                    for template in result["similar_templates"]:
                        st.markdown(
                            f"- {template['template']} "
                            f"(similarity: {template['similarity_score']:.2f})"
                        )
                    
                    # Show query response if exists
                    if result.get("query_response"):
                        phase_class = "phase-1" if result["is_phase_1"] else "phase-2"
                        phase_text = "Phase 1 - Automated Response" if result["is_phase_1"] else "Phase 2 - Requires Analyst"
                        
                        st.markdown(
                            f'<div class="response-box">'
                            f'<div class="phase-indicator {phase_class}">{phase_text}</div>'
                            f'<strong>Response:</strong><br>{result["query_response"]}'
                            '</div>',
                            unsafe_allow_html=True
                        )
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
