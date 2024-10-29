import streamlit as st
from langchain.chat_models import ChatOpenAI
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

class TemplateMatcher:
    """Dedicated class for template matching using a separate LLM instance"""
    def __init__(self, openai_api_key: str):
        # Use a separate LLM instance for template matching
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,  # Lower temperature for more consistent matching
            openai_api_key=openai_api_key
        )
        self.templates = SITREP_TEMPLATES
        self.setup_matcher()

    def setup_matcher(self):
        system_template = """You are a specialized security alert template matcher. Your only task is to match the given alert text to the most appropriate template from the provided list.

        MATCHING RULES:
        1. Compare the alert text against each available template
        2. Look for key terms and patterns that indicate a match
        3. Consider semantic similarity, not just exact matches
        4. If multiple templates could match, choose the most specific one
        5. If no template matches well, return "Unknown Template"

        FOCUS AREAS:
        - Authentication patterns (Kerberos, sign-ins)
        - Network traffic (anomalous, internal, internet)
        - IP-based threats (blacklisted, tor, spam)
        - Protocol indicators (DNS, TLS, NTP)
        - Service types (bots, scanners)

        RESPONSE FORMAT:
        Return ONLY the exact matching template name from the provided list. Nothing else."""

        human_template = """AVAILABLE TEMPLATES:
        {templates}

        ALERT TEXT TO MATCH:
        {alert_text}

        MATCHING TEMPLATE:"""
        
        self.matcher_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def match_template(self, alert_text: str) -> str:
        """Match the alert text to a template, returning the best match or Unknown Template"""
        try:
            result = self.matcher_chain.run(
                templates="\n".join(self.templates),
                alert_text=alert_text
            ).strip()
            return result if result in self.templates else "Unknown Template"
        except Exception as e:
            st.error(f"Template matching error: {str(e)}")
            return "Unknown Template"

class QueryClassifier:
    """Classifier for determining if queries are Phase 1 or Phase 2"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_classifier()

    def setup_classifier(self):
        system_template = """You are a security query classifier that determines if queries are general (Phase 1) or specific (Phase 2).

        CLASSIFY AS PHASE_1 IF:
        - Asks about general security practices
        - Requests standard mitigations
        - Seeks common guidelines
        - Asks about typical configurations
        - Needs general technical explanations
        - Inquires about alert types or categories

        CLASSIFY AS PHASE_2 IF:
        - References specific IPs/events
        - Mentions particular timestamps
        - Requires specific incident analysis
        - Asks about unique configurations
        - Involves customer-specific details
        - Needs investigation of exact occurrences

        RESPONSE FORMAT:
        Return ONLY 'PHASE_1' or 'PHASE_2'"""

        human_template = """QUERY: {query}

        CLASSIFICATION:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def classify(self, query: str) -> bool:
        """Classify a query as Phase 1 (True) or Phase 2 (False)"""
        try:
            result = self.chain.run(query=query).strip()
            return result == "PHASE_1"
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return False

class ResponseGenerator:
    """Generator for creating responses to Phase 1 queries"""
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_chain()

    def setup_chain(self):
        system_template = """You are a security expert providing responses to general security queries.
        
        RESPONSE GUIDELINES:
        1. Be direct and actionable
        2. Include specific technical details
        3. Explain why certain approaches work
        4. Focus on industry best practices
        5. Provide concrete examples
        6. Include relevant mitigation steps
        7. Reference standard security frameworks when applicable

        FORMAT:
        1. Direct answer
        2. Technical explanation
        3. Recommended actions
        4. Additional considerations"""

        human_template = """CONTEXT:
        Alert Template: {template}
        Alert Summary: {alert_summary}
        Query: {query}

        RESPONSE:"""

        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def generate(self, template: str, alert_summary: str, query: str) -> str:
        """Generate a response based on the template, alert summary, and query"""
        try:
            return self.chain.run(
                template=template,
                alert_summary=alert_summary,
                query=query
            ).strip()
        except Exception as e:
            st.error(f"Response generation error: {str(e)}")
            return "Error generating response. Please try again."

class SitrepAnalyzer:
    """Main analyzer class that coordinates template matching, classification, and response generation"""
    def __init__(self, openai_api_key: str):
        self.template_matcher = TemplateMatcher(openai_api_key)
        self.query_classifier = QueryClassifier(openai_api_key)
        self.response_generator = ResponseGenerator(openai_api_key)

    def extract_status(self, text: str) -> Optional[str]:
        """Extract status from alert text if present"""
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        """Analyze the situation report and generate appropriate response"""
        try:
            # Always perform template matching regardless of query phase
            matched_template = self.template_matcher.match_template(alert_summary)
            status = self.extract_status(alert_summary)
            
            result = {
                "template": matched_template,
                "status": status,
                "is_phase_1": None,
                "query_response": None
            }
            
            # Process query if provided
            if client_query:
                is_phase_1 = self.query_classifier.classify(client_query)
                result["is_phase_1"] = is_phase_1
                
                if is_phase_1:
                    response = self.response_generator.generate(
                        matched_template,
                        alert_summary,
                        client_query
                    )
                else:
                    response = "⚠️ This query requires analyst review - Phase 2 query detected."
                    
                result["query_response"] = response
            
            return result
            
        except Exception as e:
            return {
                "error": f"Analysis error: {str(e)}",
                "template": "Unknown Template",
                "status": None,
                "is_phase_1": None,
                "query_response": None
            }

def main():
    st.set_page_config(page_title="Security Alert Analyzer", layout="wide")
    
    st.markdown("""
        <style>
        .header-box { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #2a5298;
        }
        .response-box { 
            background: white; 
            padding: 20px; 
            border-radius: 5px; 
            margin: 10px 0; 
            border-left: 4px solid #3498db; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .phase-indicator {
            padding: 8px 15px;
            border-radius: 3px;
            font-weight: bold;
            margin-bottom: 15px;
            display: inline-block;
        }
        .phase-1 {
            background-color: #d4edda;
            color: #155724;
        }
        .phase-2 {
            background-color: #f8d7da;
            color: #721c24;
        }
        .template-match {
            background-color: #e2e3e5;
            color: #383d41;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Security Alert Analyzer")
    
    try:
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Alert Summary")
            alert_summary = st.text_area(
                "Enter alert details",
                height=300,
                help="Paste the complete alert summary here"
            )

        with col2:
            st.subheader("User Query")
            client_query = st.text_area(
                "Enter your question",
                height=150,
                help="Enter your question about the alert"
            )
        
        if st.button("Analyze Alert", type="primary"):
            if not alert_summary:
                st.error("Please enter alert details to analyze.")
                return
            
            analyzer = SitrepAnalyzer(OPENAI_API_KEY)
            
            with st.spinner("Analyzing alert..."):
                result = analyzer.analyze_sitrep(alert_summary, client_query)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display template match and status
                    header_content = []
                    if result["template"] != "Unknown Template":
                        header_content.append(f"<strong>Matched Template:</strong> {result['template']}")
                    if result.get("status"):
                        header_content.append(f"<strong>Status:</strong> {result['status']}")
                    
                    if header_content:
                        st.markdown(
                            '<div class="header-box">' + 
                            '<br>'.join(header_content) + 
                            '</div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Display query response if query exists
                    if client_query and result.get("is_phase_1") is not None:
                        phase_class = "phase-1" if result["is_phase_1"] else "phase-2"
                        phase_text = "Phase 1 - Automated Response" if result["is_phase_1"] else "Phase 2 - Requires Analyst Review"
                        
                        st.markdown(
                            f'<div class="phase-indicator {phase_class}">{phase_text}</div>',
                            unsafe_allow_html=True
                        )
                        
                        if result.get("query_response"):
                            st.markdown(
                                '<div class="response-box">' +
                                '<strong>Analysis Response:</strong><br><br>' +
                                f'{result["query_response"]}' +
                                '</div>',
                                unsafe_allow_html=True
                            )
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
