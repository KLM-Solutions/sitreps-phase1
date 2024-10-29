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

class EnhancedTemplateMatcher:
    """Enhanced template matcher using FAISS and LLM verification"""
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.templates = SITREP_TEMPLATES
        self.setup_matcher()
        self.setup_vector_store()

    def setup_vector_store(self):
        """Initialize FAISS vector store with templates"""
        template_embeddings = self.embeddings.embed_documents(self.templates)
        self.vector_store = FAISS.from_embeddings(
            template_embeddings,
            self.templates,
            self.embeddings
        )

    def setup_matcher(self):
        """Setup LLM chain for template verification"""
        system_template = """You are a security alert template analyzer. Your task is to:
        1. Analyze the alert summary
        2. Review potential template matches
        3. Determine the most accurate template match
        4. Extract key information and patterns

        Return your analysis in the following format:
        {
            "best_template": "exact template name",
            "confidence": <0-100>,
            "key_patterns": ["pattern1", "pattern2"],
            "explanation": "brief explanation of match"
        }"""

        human_template = """Alert Summary:
        {alert_text}

        Similar Templates:
        {similar_templates}

        Analyze and determine best template match:"""

        self.verification_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def match_template(self, alert_text: str) -> Dict:
        """Match template using FAISS and LLM verification"""
        # Find similar templates using FAISS
        similar_docs = self.vector_store.similarity_search_with_score(alert_text, k=3)
        similar_templates = [{"template": doc.page_content, "score": score} 
                           for doc, score in similar_docs]

        # Format templates for LLM verification
        templates_str = "\n".join([
            f"- {t['template']} (similarity: {1 - t['score']:.2f})"
            for t in similar_templates
        ])

        # Verify using LLM
        result = self.verification_chain.run(
            alert_text=alert_text,
            similar_templates=templates_str
        )
        
        try:
            analysis = eval(result)
            return {
                "template": analysis["best_template"],
                "confidence": analysis["confidence"],
                "key_patterns": analysis["key_patterns"],
                "explanation": analysis["explanation"],
                "similar_matches": similar_templates
            }
        except Exception as e:
            return {
                "template": "Unknown Template",
                "confidence": 0,
                "key_patterns": [],
                "explanation": f"Error in template matching: {str(e)}",
                "similar_matches": []
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
        # [Previous QueryClassifier code remains the same]
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
        # [Previous CrispResponseGenerator code remains the same]
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
        
        Provide direct technical response:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate(self, alert_summary: str, query: str) -> str:
        return self.chain.run(alert_summary=alert_summary, query=query).strip()

class SitrepAnalyzer:
    def __init__(self):
        try:
            self.template_matcher = EnhancedTemplateMatcher(OPENAI_API_KEY)
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
            # Get enhanced template matching results
            template_results = self.template_matcher.match_template(alert_summary)
            status = self.extract_status(alert_summary)
            
            query_response = None
            is_phase_1 = False
            
            if client_query:
                is_phase_1 = self.query_classifier.classify(client_query)
                if is_phase_1:
                    query_response = self.response_generator.generate(alert_summary, client_query)
                else:
                    query_response = "⚠️ This query requires analyst review - beyond Phase 1 automation scope."
            
            return {
                **template_results,
                "status": status,
                "is_phase_1": is_phase_1,
                "query_response": query_response
            }
        except Exception as e:
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
        .pattern-box {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .confidence-high {
            color: #155724;
            background-color: #d4edda;
            padding: 3px 8px;
            border-radius: 3px;
        }
        .confidence-medium {
            color: #856404;
            background-color: #fff3cd;
            padding: 3px 8px;
            border-radius: 3px;
        }
        .confidence-low {
            color: #721c24;
            background-color: #f8d7da;
            padding: 3px 8px;
            border-radius: 3px;
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
                    # Show template matching results
                    confidence_class = (
                        "confidence-high" if result["confidence"] >= 80
                        else "confidence-medium" if result["confidence"] >= 50
                        else "confidence-low"
                    )
                    
                    st.markdown(
                        f'<div class="header-box">'
                        f'<strong>Matched Template:</strong> {result["template"]}<br>'
                        f'<strong>Confidence:</strong> <span class="{confidence_class}">{result["confidence"]}%</span><br>'
                        f'<strong>Status:</strong> {result.get("status", "Not specified")}<br>'
                        f'<strong>Match Explanation:</strong> {result["explanation"]}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show key patterns
                    if result["key_patterns"]:
                        st.markdown("### Key Patterns Identified")
                        for pattern in result["key_patterns"]:
                            st.markdown(f'<div class="pattern-box">{pattern}</div>', 
                                      unsafe_allow_html=True)
                    
                    # Show similar matches
                    if result["similar_matches"]:
                        st.markdown("### Similar Templates")
                        for match in result["similar_matches"]:
                            similarity = 1 - match["score"]
                            st.markdown(f"- {match['template']} (similarity: {similarity:.2f})")
                    
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
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
