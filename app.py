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

# API Configuration - Make sure to set this in your Streamlit secrets
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
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=openai_api_key
        )
        self.templates = SITREP_TEMPLATES
        self.setup_matcher()

    def setup_matcher(self):
        system_template = """You are a security alert template matcher. Your task is to match alerts to specific templates.

        RULES:
        1. Compare incoming alert text with the provided templates
        2. Return ONLY the exact matching template name
        3. Do not modify or paraphrase template names
        4. Match based on key terms and overall context
        5. Return "Unknown Template" if no clear match exists

        MATCHING PRIORITIES:
        1. Exact phrase matches
        2. Key security terms (Kerberos, TOR, DNS, etc.)
        3. Alert type indicators (traffic, authentication, scanning)
        4. IP-based threats
        5. Service-specific alerts"""

        human_template = """AVAILABLE TEMPLATES:
{templates}

ALERT TEXT:
{alert_text}

MATCHED TEMPLATE (return exact template name):"""
        
        self.matcher_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        )

    def match_template(self, alert_text: str) -> str:
        try:
            # Debug logging
            st.write("Debug: Starting template matching...")
            
            formatted_templates = "\n".join(f"- {template}" for template in self.templates)
            result = self.matcher_chain.run({
                'templates': formatted_templates,
                'alert_text': alert_text
            }).strip()
            
            # Debug logging
            st.write(f"Debug: Raw matcher result: {result}")
            
            # Clean and validate result
            cleaned_result = result.strip()
            if cleaned_result in self.templates:
                st.write(f"Debug: Exact match found: {cleaned_result}")
                return cleaned_result
            
            # Attempt partial matching if no exact match
            for template in self.templates:
                if template.lower() in cleaned_result.lower():
                    st.write(f"Debug: Partial match found: {template}")
                    return template
            
            st.write("Debug: No match found, returning Unknown Template")
            return "Unknown Template"
            
        except Exception as e:
            st.error(f"Template matching error: {str(e)}")
            return "Unknown Template"

class QueryClassifier:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_classifier()

    def setup_classifier(self):
        system_template = """You are a security query classifier that determines if queries are Phase 1 (general) or Phase 2 (specific).

        PHASE 1 INDICATORS:
        - General security practices
        - Standard configurations
        - Common alert types
        - Best practices
        - Generic mitigation strategies
        - Documentation requests
        - Learning about alert categories

        PHASE 2 INDICATORS:
        - Specific IP addresses
        - Exact timestamps
        - Customer data
        - Unique incidents
        - Custom configurations
        - Detailed investigations
        - Specific event analysis

        OUTPUT:
        Return ONLY "PHASE_1" or "PHASE_2" """

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
        try:
            st.write("Debug: Starting query classification...")
            result = self.chain.run(query=query).strip()
            st.write(f"Debug: Classification result: {result}")
            return result == "PHASE_1"
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return False

class ResponseGenerator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.setup_generator()

    def setup_generator(self):
        system_template = """You are a security expert providing responses to Phase 1 security queries.

        RESPONSE GUIDELINES:
        1. Focus on actionable advice
        2. Include technical details when relevant
        3. Reference industry best practices
        4. Provide clear mitigation steps
        5. Explain security concepts
        6. Give concrete examples
        7. Reference common frameworks

        FORMAT:
        1. Direct answer
        2. Technical explanation
        3. Recommended actions
        4. Additional context"""

        human_template = """CONTEXT:
        Template: {template}
        Alert: {alert_summary}
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
        try:
            st.write("Debug: Generating response...")
            response = self.chain.run(
                template=template,
                alert_summary=alert_summary,
                query=query
            ).strip()
            st.write("Debug: Response generated successfully")
            return response
        except Exception as e:
            st.error(f"Response generation error: {str(e)}")
            return "Error generating response. Please try again."

class SitrepAnalyzer:
    def __init__(self, openai_api_key: str):
        self.template_matcher = TemplateMatcher(openai_api_key)
        self.query_classifier = QueryClassifier(openai_api_key)
        self.response_generator = ResponseGenerator(openai_api_key)

    def extract_status(self, text: str) -> Optional[str]:
        status_match = re.search(r"Status:([^\n]*)", text, re.IGNORECASE)
        return status_match.group(1).strip() if status_match else None

    def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            # Always perform template matching
            st.write("Debug: Starting SITREP analysis...")
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
                st.write("Debug: Processing client query...")
                is_phase_1 = self.query_classifier.classify(client_query)
                result["is_phase_1"] = is_phase_1
                
                if is_phase_1:
                    st.write("Debug: Generating Phase 1 response...")
                    response = self.response_generator.generate(
                        matched_template,
                        alert_summary,
                        client_query
                    )
                else:
                    response = "⚠️ This query requires analyst review - Phase 2 query detected."
                    
                result["query_response"] = response
            
            st.write("Debug: Analysis complete")
            return result
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return {
                "error": f"Analysis error: {str(e)}",
                "template": "Unknown Template",
                "status": None,
                "is_phase_1": None,
                "query_response": None
            }

def main():
    st.set_page_config(page_title="Security Alert Analyzer", layout="wide")
    
    # Custom CSS for better UI
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
        .debug-box {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            margin: 5px 0;
            font-family: monospace;
        }
        .template-match {
            background: #e9ecef;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Security Alert Analyzer")
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")
    
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Alert Summary")
            alert_summary = st.text_area(
                "Enter alert details",
                height=300,
                help="Paste the complete alert summary here",
                key="alert_input"
            )

        with col2:
            st.subheader("User Query")
            client_query = st.text_area(
                "Enter your question",
                height=150,
                help="Enter your question about the alert",
                key="query_input"
            )
        
        if st.button("Analyze Alert", type="primary"):
            if not alert_summary:
                st.error("Please enter alert details to analyze.")
                return
            
            # Create analyzer instance
            analyzer = SitrepAnalyzer(OPENAI_API_KEY)
            
            with st.spinner("Analyzing alert..."):
                # If debug mode is enabled, redirect debug output
                if debug_mode:
                    st.write("Debug: Starting analysis process...")
                
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
                    
                    # Display debug information if enabled
                    if debug_mode:
                        st.subheader("Debug Information")
                        st.json(result)
                    
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
