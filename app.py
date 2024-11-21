import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai
from typing import Dict, Optional, List
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your existing SITREP_TEMPLATES_DETAILED dictionary
SITREP_TEMPLATES_DETAILED = {
    "Anomalous Internal Traffic": {
        "name": """<div id="name">Test Anomalous Internal Traffic - IDS Alert</div>""",
        "title": """<div id="title">Test Anomalous Internal Traffic - IDS Alert</div>""",
        "single_summary": """<div id="single-summary">
            <p>From {{interval}}, traffic initiated from asset with last known IP {{src_ip}}, on {{src_ports}}, to asset with last known IP {{destination_ip}}, on port {{destination_ports}}, threw an anomaly on {{unit}}.</p>
            <p>Our IDS<< Please add the IDS link and delete it>> Detected an alert for the following signatures
            << Please add the signatures from the IDS Tab and delete it>></p>
            <p>Previous traffic between these two assets WAS THIS MUCH add count on sessions or sum for size / packets on average / day in the interval Gradient Cyber keeps your data for analysis. However, during the above mentioned interval it's THIS MUCH - add count on sessions or sum for size / packets on average / day. //////// OR /////////// This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.</p>
            <p>DYNAMIC_CONTENT:WITH_EVENT_TYPE_LINK</p>
            <b>PLEASE CHANGE THE WORDINGS IF THE FIELDS MENTIONED ARE EMPTY AND DELETE THIS AS WELL</b>
            <p>Please let us know if this traffic was expected and <strong>if anomalies on it are something you'd want to be informed of in the future.</strong></p>
            <p>Please check this <strong>LINK_WITHOUT_DATE_SELECTION</strong> to know more about traffic between the above-mentioned IPs. <strong>------- Delete this line if there was no date filter when creating sitrep</strong></p>
            </div>""",
        "plural_summary": """[Your plural summary content]""",
        "bidirectional_single_summary": """[Your bidirectional single summary content]""",
        "bidirectional_plural_summary": """[Your bidirectional plural summary content]""",
        "applications": """<div id="applications">East - West Route CM</div>""",
        "message_priority": """<div id="message-priority">5: Informational</div>""",
        "other_related_events": """<div id="other-related-events">Anomaly: Packets</div>""",
        "organizational_wordtags": """<div id="organizational-wordtags">cyber hygiene, internal</div>"""
    },
    
    "DNS Queries to bad domains": {
        "name": """<div id="name">DNS Queries to bad domains</div>""",
        "title": """<div id="title">DNS Queries to bad domains</div>""",
        "single_summary": """[Your DNS queries single summary content]""",
        "event_description": """[Your event description content]""",
        "applications": """<div id="applications">Perimeter (CM)</div>""",
        "message_priority": """<div id="message-priority">3: Medium</div>""",
        "other_related_events": """<div id="other-related-events">Malware</div>"""
    },
    "DNS Queries to bad domains": {
        "name": """<div id="name">
DNS Queries to bad domains
</div>""",
        
        "title": """<div id="title">
DNS Queries to bad domains
</div>""",
        
        "single_summary": """<div id="single-summary"> 
Our IDS has identified DNS queries to the below domain(s), associated with malicious activity by our threat intelligence sources.
<p>
vt_domains_key. 
ioc_sources_key
<p>
Please access the following links for screenshots: urlscan_domains_key.
<b>CHECK IF THERE IS A LINK, IF NOT, DELETE THE SCREENSHOTS PART AND THIS AS WELL</b>
<p>
link_to_ids_key
<p>
<b>Description and Risks associated with Bad Domains</b>
Malicious domains are identified by Gradient Cyber using different criteria:
1. Domains labeled as malicious/malware/phishing and scored by Virus Total Engine.
2. Domains labeled as malicious based on TLDs. Out of more than 1,000 TLDs, the top 25 TLDs (by the number of malicious domains) account for more than 90% of all malicious domain names. While these 25 TLDs are not malicious, they are well-positioned to help mitigate malicious domain registrations. We find that TLDs offering free domain registration are among the top preferred TLDs for phishing domains.<a href="https://unit42.paloaltonetworks.com/top-level-domains-cybercrime/" target="_blank"><b>[1]</b></a>
3. Domains labeled as malicious based on DGAs. Adversaries may make use of Domain Generation Algorithms (DGAs) to dynamically identify a destination domain for command and control traffic rather than relying on a list of static IP addresses or domains. This has the advantage of making it much harder for defenders to block, track, or take over the command and control channel, as there potentially could be thousands of domains that malware can check for instructions.<a href="https://umbrella.cisco.com/blog/domain-generation-algorithms-effective" target="_blank"><b>[2]</b></a><a href="https://unit42.paloaltonetworks.com/threat-brief-understanding-domain-generation-algorithms-dga/" target="_blank"><b>[3]</b></a>. DGAs can take the form of apparently random or "gibberish" strings (ex: istgmxdejdnxuyla[.]ru) when they construct domain names by generating each letter. Gradient Cyber uses Threat Intel to identify and label domains which are DGAs.
4. Domains that are labeled as risky and suspicious without garnering a score on Virus Total (check the Details Tab and look for the Categories section).
5. Gradient's own research into malicious domains that stems from malware reverse engineering and IR.
<p>
<b>Technique</b>
<a href="https://attack.mitre.org/techniques/T1568/002/" target="_blank"><b>MITRE: Dynamic Resolution: Domain Generation Algorithms - T1568.002</b></a>.
<p>
<b> Mitigations</b>
1. <a href="https://attack.mitre.org/mitigations/M1031/" target="_blank"><b>MITRE_M1031_Network Intrusion Prevention</b></a>: Network intrusion detection and prevention systems that use network signatures to identify traffic for specific adversary malware can be used to mitigate activity at the network level. Malware researchers can reverse engineer malware variants that use DGAs and determine future domains that the malware will attempt to contact, but this is a time and resource intensive effort.<a href="https://umbrella.cisco.com/blog/domain-generation-algorithms-effective" target="_blank"><b>[2]</b></a><a href="https://umbrella.cisco.com/blog/at-high-noon-algorithms-do-battle" target="_blank"><b>[6]</b></a> Malware is also increasingly incorporating seed values that can be unique for each instance, which would then need to be determined to extract future generated domains. In some cases, the seed that a particular sample uses can be extracted from DNS traffic. Even so, there can be thousands of possible domains generated per day; this makes it impractical for defenders to preemptively register all possible C2 domains due to the cost.
2. <a href="https://attack.mitre.org/mitigations/M1031/" target="_blank"><b>MITRE_M1031_Restrict Web-Based Content</b></a>: In some cases a local DNS sinkhole may be used to help prevent DGA-based command and control at a reduced cost.
3. Blocking the URI using a content filter or ACL.
4. Investigate for any signs of infection in the local network.
<p>
  
Please access the Files tab for instruction sets on mitigation solutions for Windows and Palo Alto firewalls, if applicable.
<p>
</p></p></p><p><b>Gradient Cyber recommends applying the security measures across all your locations. Please notify the analyst if the above measures are not scalable across the entire organization.</b></p>
<p>
</p></div>""",

        "event_description": """<div id="event-description">
For more information please refer to the links given below:
http://blog.talosintelligence.com/2017/09/avast-distributes-malware.html
https://umbrella.cisco.com/blog/domain-generation-algorithms-effective
https://unit42.paloaltonetworks.com/threat-brief-understanding-domain-generation-algorithms-dga/
https://www.mandiant.com/resources/dissecting-one-ofap
https://www.welivesecurity.com/2017/12/21/sednit-update-fancy-bear-spent-year/
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
3: Medium
</div>""",

        "other_related_events": """<div id="other-related-events">
Malware
</div>"""
    },
"Anomalous Internet Traffic: Size": {
        "name": """<div id="name" override>
Anomalous Internet Traffic: Size
</div>""",

        "title": """<div id="title" override>
Anomalous Internet Traffic: Size
</div>""",

        "single_summary": """<div id="single-summary">
During {{interval}}, an unusual increase in traffic size was observed  (____ TO OR FROM______) the below IP.
<p>
Previous traffic size with this IP was - add sum for size / average / day. However, during the above mentioned interval is- add sum for size / average / day.
 ////// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT
</div>""",

        "plural_summary": """<div id="plural-summary">
During {{interval}}, an unusual increase in traffic size was observed  (____ TO OR FROM______) the below IPs.
<p>
Previous traffic size with these IPs was - add sum for size / average / day. However, during the above mentioned interval is - add sum for size / average / day. 
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
DYNAMIC_CONTENT
</div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
During {{interval}}, an unusual increase in traffic size was observed with Internet IP {{internet_ip}}. Traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
<p>
Previous traffic size with this IP was - add sum for size / average / day in the interval Gradient Cyber keeps your data for analysis. However, during the above mentioned interval is - add sum for size / average / day. 
////// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT
</div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
During {{interval}}, an unusual increase in traffic size was observed with the below IPs. Traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
<p>
Previous traffic size with these IPs was - add sum for size / average / day in the interval Gradient Cyber keeps your data for analysis. However, during the above mentioned interval it's THIS MUCH - add sum for size / average / day. 
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
DYNAMIC_CONTENT
</div>""",

        "event_description": """<div id="event-description">
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
5: Informational
</div>""",

        "other_related_events": """<div id="other-related-events">
Anomaly: Size
</div>""",

        "organizational_wordtags": """<div id="organizational-wordtags">
cyber hygiene, web hosting, incoming, outgoing
</div>"""
    },
    "Blacklisted IP": {
        "name": """<div id="name">
Traffic {{direction}} {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary">
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. 
This IP has been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
<p>
Unless there is a business requirement for this traffic, Gradient Cyber recommends blocking the IP address. Please access the Files tab for advanced recommendations on web application security.  
For more recommendations please refer to <i>Related Events</i> tab.

</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary">
Traffic was found {{direction}} these IP addresses in the Perimeter Threat (PCAP) application. <b>[Add info about ports, sessions/packets, assets if relevant and delete this part after]
</b>
These IPs have been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Unless there is a business requirement for this traffic, Gradient Cyber recommends blocking these IP addresses. Please access the Files tab for advanced recommendations on web application security. 
For more recommendations please refer to <i>Related Events</i> tab.
</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
Traffic was found with Internet IP {{internet_ip}}. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
This IP has been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Unless there is a business requirement for this traffic, Gradient Cyber recommends blocking the IP address. Please access the Files tab for advanced recommendations on web application security. 
For more recommendations please refer to <i>Related Events</i> tab.
</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
Traffic was found with multiple IP addresses. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
These IPs have been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Unless there is a business requirement for this traffic, Gradient Cyber recommends blocking these IP addresses. Please access the Files tab for advanced recommendations on web application security.  
For more recommendations please refer to <i>Related Events</i> tab.
</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "event_description": """<div id="event-description">
Avoiding exposure to malicious content perpetrators is a matter of cyber hygiene defined as a suite of actions taken in order to implement and maintain the cybersecurity of IT systems and devices.
Some common such actions include:
- Educate users on practicing good cyber behavior, including password management, identifying potential phishing efforts, and which devices to connect to the network\
- Turn to industry-accepted secure configurations/standards like NIST and CIS Benchmark. These can help organizations define items like password length, encryption, port access, and double authentication
 - Utilize two-factor authentication such as a password and SMS text code or a password and a security token to prevent unauthorized remote access
- Build redundancy necessary to facilitate patches and updates

Further reading on this topic can be found in the following sources:
https://www.sentinelone.com/blog/practice-these-10-basic-cyber-hygiene-tips-for-risk-mitigation/
https://digitalguardian.com/blog/enterprise-cyber-security-hygiene-best-practices
https://www.helpnetsecurity.com/2018/08/09/state-of-cyber-hygiene/
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
4: Low
</div>""",

        "other_related_events": """<div id="other-related-events">
Blacklisted
</div>"""
    },
"Anomalous Internet Traffic: Packets": {
        "name": """<div id="name"override>
Anomalous Internet Traffic: Packets
</div>""",

        "title": """<div id="title"override>
Anomalous Internet Traffic: Packets
</div>""",

        "single_summary": """<div id="single-summary">
During {{interval}}, an unusual increase in number of packets was observed for traffic  (____ TO OR FROM______) the above IP.
<p>
Previous traffic with this IP had - add count for packets / average / day. However, during the above mentioned interval it has - add count of packets / average / day.
////// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT
</div>""",

        "plural_summary": """<div id="plural-summary">
During {{interval}}, an unusual increase in number of packets was observed for traffic (____ TO OR FROM______) the below IPs.
<p>
Previous traffic with these IPs had - add count of packets / average / day. However, during the above mentioned interval it has - add count of packets / average / day. 
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
DYNAMIC_CONTENT
</div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
During {{interval}}, an unusual increase in number of packets was observed for traffic with Internet IP {{internet_ip}}. Traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
<p>
Previous traffic with this IP had - add count of packets / average / day in the interval Gradient Cyber keeps your data for analysis. However, during the above mentioned interval has - add count of packets / average / day.
////// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT
</div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
During {{interval}}, an unusual increase in number of packets was observed for traffic with multiple IP addresses. Traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
<p>
Previous traffic with these IPs had - add count of packets / average / day in the interval Gradient Cyber keeps your data for analysis. However, during the above mentioned interval it has - add count of packets / on average / day.  
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
DYNAMIC_CONTENT
</div>""",

        "event_description": """<div id="event-description">

Packet spikes in internet traffic may have various causes and while some of these causes might not be inherently malicious, performance could be hampered by improper administration.

Some of the causes for packet spikes include:

File Sharing - Programs based on the BitTorrent standard turn users into nodes in a file-sharing network, utilizing their connection to provide data to other users while they download. These programs can overwhelm the upstream capacity of any Internet connection, slowing downloads as well.

Streaming Media - Sites like YouTube, Spotify and Pandora provide video and audio content to users on demand, usually on low-bitrate streams that might not interfere with a household Internet connection. In large companies, however, where dozens or even hundreds of users might access these services simultaneously, the total bandwidth load can interfere with business operations.

Videoconferencing - Lowering the bandwidth and resolution of each individual stream can help maximize the amount of users that can share a single connection without issue.

Malware - Many types of malware infect systems in order to take control of the servers and use them for various nefarious purposes, such as the distribution of spam e-mail or distributed attacks on other servers. When activated, these programs can flood your upstream connection with data, though some might try to camouflage their activity by keeping their bandwidth usage low. However, spyware will cause high amounts of bandwidth.

Gaming - On-line game playing and video/audio chatting also uses higher bandwidth.

Other related information here: https://smallbusiness.chron.com/causes-internet-traffic-47796.html 
and here https://blog.paessler.com/the-top-5-causes-of-sudden-network-spikes

</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
5: Informational
</div>""",

        "other_related_events": """<div id="other-related-events">
Anomaly: Packets
</div>""",

        "organizational_wordtags": """<div id="organizational-wordtags">
cyber hygiene, web hosting, incoming, outgoing
</div>"""
    },
"Anomalous Internet Traffic: Sessions": {
        "name": """<div id="name"override>
Test Anomalous Internet Traffic: Sessions Test
</div>""",

        "title": """<div id="title"override>
Test Anomalous Internet Traffic: Sessions Test
</div>""",

        "single_summary": """<div id="single-summary">
During {{interval}}, an unusual increase in session number was observed for traffic  (____ TO OR FROM______) the above IP. 
<p>
<b>PLEASE CHECK BELOW WHICH IS APPLICABLE AND DELETE THIS AS WELL</b>
This IP is not reported on our Threat Intelligence sources.
///// OR ////// 
<b>PLEASE INCREASE THE PRIORITY OF THE SITREP TO MEDIUM IF THE IP HAD BEEN REPORTED BY THREAT INTELLIGENCE SOURCES AND DELETE THIS AS WELL</b><p>
This IP has been reported for malicious activity on our Threat Intelligence sources
THREAT_INTEL_KEY
<p>
Previous traffic with this IP had -add the sum of sessions/average/day- in the interval Gradient Cyber keeps your data for analysis.
However, during the above-mentioned interval, it has - add the sum of sessions/average/day. 
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT:WITH_EVENT_TYPE_LINK
<b>PLEASE CHANGE THE WORDINGS IF THE FIELDS MENTIONED ARE EMPTY AND DELETE THIS AS WELL</b><p>
</div>""",

        "plural_summary": """<div id="plural-summary">
During {{interval}}, an unusual increase in session number was observed for traffic (____ TO OR FROM______) the below IPs.
<p>
<b>PLEASE CHECK BELOW WHICH IS APPLICABLE AND DELETE THIS AS WELL</b>
These IPs are not reported on our Threat Intelligence sources.
///// OR ////// 
<b>PLEASE INCREASE THE PRIORITY OF THE SITREP TO MEDIUM IF THE IP HAD BEEN REPORTED BY THREAT INTELLIGENCE SOURCES AND DELETE THIS AS WELL</b><p>
The following IPs have been reported for malicious activity on our Threat Intelligence sources
THREAT_INTEL_KEY
<p>
Previous traffic with these IPs had -add the sum of sessions/average/day- in the interval Gradient Cyber keeps your data for analysis.
However, during the above-mentioned interval, it has - add the sum of sessions/average/day.
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
DYNAMIC_CONTENT:WITH_EVENT_TYPE_LINK
<b>PLEASE CHANGE THE WORDINGS IF THE FIELDS MENTIONED ARE EMPTY AND DELETE THIS AS WELL</b><p>
</div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
During {{interval}}, an unusual increase in session number was observed for traffic with Internet IP {{internet_ip}}. Traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
<p>
<b>PLEASE CHECK BELOW WHICH IS APPLICABLE AND DELETE THIS AS WELL</b>
This IP is not reported on our Threat Intelligence sources.
///// OR ////// 
<b>PLEASE INCREASE THE PRIORITY OF THE SITREP TO MEDIUM IF THE IP HAD BEEN REPORTED BY THREAT INTELLIGENCE SOURCES AND DELETE THIS AS WELL</b><p>
This IP has been reported for malicious activity on our Threat Intelligence sources
THREAT_INTEL_KEY
<p>
Previous traffic with this IP had -add the sum of sessions/average/day- in the interval Gradient Cyber keeps your data for analysis.
However, during the above-mentioned interval, it has - add the sum of sessions/average/day. 
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT:WITH_EVENT_TYPE_LINK
<b>PLEASE CHANGE THE WORDINGS IF THE FIELDS MENTIONED ARE EMPTY AND DELETE THIS AS WELL</b><p>
</div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
During {{interval}}, an unusual increase in session number was observed for traffic with multiple Internet IPs. Traffic was initiated from {{The_Internet_or_The_Local_Network}}. 
<p>
<b>PLEASE CHECK BELOW WHICH IS APPLICABLE AND DELETE THIS AS WELL</b>
These IPs are not reported on our Threat Intelligence sources.
///// OR ////// 
<b>PLEASE INCREASE THE PRIORITY OF THE SITREP TO MEDIUM IF THE IP HAD BEEN REPORTED BY THREAT INTELLIGENCE SOURCES AND DELETE THIS AS WELL</b><p>
The following IPs have been reported for malicious activity on our Threat Intelligence sources
THREAT_INTEL_KEY
<p>
Previous traffic with these IPs had -add the sum of sessions/average/day- in the interval Gradient Cyber keeps your data for analysis.
However, during the above-mentioned interval, it has - add the sum of sessions/average/day.
///// OR ////// 
This is the first time we've seen this traffic in the interval Gradient Cyber keeps your data for analysis.
<p>
Please let us know if this traffic was expected and <b>if anomalies on it are something you'd want to be informed of in the future.</b>
<p>
DYNAMIC_CONTENT:WITH_EVENT_TYPE_LINK
<b>PLEASE CHANGE THE WORDINGS IF THE FIELDS MENTIONED ARE EMPTY AND DELETE THIS AS WELL</b><p>
</div>""",

        "event_description": """<div id="event-description">

Session spikes in internet traffic may have various causes and while some of this internet traffic might not be inherently malicious, performance could be hampered by improper administration. 

Usually, a high number of sessions, aside from high traffic causes, are due to the following:

Malware - high number of sessions, typically within a few-seconds duration, are specific to malicious entities trying to perform activities such as vulnerability scans, denial-of-service attacks, SSH attacks, and spam. 

</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
5: Informational
</div>""",

        "other_related_events": """<div id="other-related-events">
Anomaly: Sessions
</div>""",

        "organizational_wordtags": """<div id="organizational-wordtags">
cyber hygiene, web hosting, incoming, outgoing
</div>"""
    },
    "Anonymization Services IP": {
        "name": """<div id="name">Traffic {{direction}} {{threatType}} - {{country}}</div>""",
        
        "single_summary": """<div id="single-summary"><span style="background-color: rgb(35, 42, 58); color: rgb(209, 222, 247);">Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application..&nbsp;</span> This IP is an anonymization service. Traffic like this wants to remain anonymous, which can be a threat as cyber criminals use these services to scout particular targets. This IP has been reported for malicious activities by the intelligence sources listed below: THREAT_INTEL_KEY DYNAMIC_CONTENT If the firewall settings allow it, please block the anonymization URL. Otherwise, if there is no business requirement for this traffic, Gradient Cyber recommends blocking the above IP.
<div id="single-summary">Gradient Cyber recommends applying the security measures across all your locations.</div>
<div id="single-summary"><br></div>
<div id="single-summary"> 
Description: IP anonymization, also known as IP masking, is a method of replacing the original IP address with one that cannot be associated with or traced back to an individual user. This can be done by setting the last octet of IPV4 addresses or the last 80 bits of IPv6 addresses to zeros.</div>
<div id="single-summary">
Technique: 
1. <a href="https://attack.mitre.org/techniques/T1133/" rel="noopener noreferrer" target="_blank">MITRE: External Remote Services-T1133</a>. 
2. <a href="https://attack.mitre.org/techniques/T1090/" rel="noopener noreferrer" target="_blank">MITRE:Proxy-T1090</a>.</div>
<div id="single-summary"> 
Mitigations: 
1. <a href="https://attack.mitre.org/mitigations/M1042/" rel="noopener noreferrer" target="_blank">MITRE_M1042_Disable or Remove Feature or Program</a>: Disable or block remotely available services that may be unnecessary. 
2. <a href="href=" rel="noopener noreferrer" target="_blank">MITRE_M1035_Limit Access to Resource Over Network</a>: Limit access to remote services through centrally managed concentrators such as VPNs and other managed remote access systems. 
3. <a href="href=" rel="noopener noreferrer" target="_blank">MITRE_M1032_Multi-factor Authentication</a>: Use strong two-factor or multi-factor authentication for remote service accounts to mitigate an adversary's ability to leverage stolen credentials, but be aware of <a href="https://attack.mitre.org/techniques/T1111/" rel="noopener noreferrer" target="_blank">Multi-Factor Authentication Interception</a> techniques for some two-factor authentication implementations. 
4. <a href="href=" rel="noopener noreferrer" target="_blank">MITRE_M1030_Network Segmentation</a>: Deny direct remote access to internal systems through the use of network proxies, gateways, and firewalls. 
5. <a href="href=" rel="noopener noreferrer" target="_blank">MITRE_M1030_Filter Network Traffic</a>: Traffic to known anonymity networks and C2 infrastructure can be blocked through the use of network allow and block lists. It should be noted that this kind of blocking may be circumvented by other techniques like <a href="https://attack.mitre.org/techniques/T1090/004/" rel="noopener noreferrer" target="_blank">Domain Fronting</a>. 
6. <a href="href=" rel="noopener noreferrer" target="_blank">MITRE_M1031_Network Intrusion Prevention</a>: Network intrusion detection and prevention systems that use network signatures to identify traffic for specific adversary malware can be used to mitigate activity at the network level. Signatures are often for unique indicators within protocols and may be based on the specific C2 protocol used by a particular adversary or tool, and will likely be different across various malware families and versions. Adversaries will likely change tool C2 signatures over time or construct protocols in such a way as to avoid detection by common defensive tools. <a href="https://arxiv.org/ftp/arxiv/papers/1408/1408.1136.pdf" rel="noopener noreferrer" target="_blank">[1]</a>. 
7. <a href="href=" rel="noopener noreferrer" target="_blank">MITRE_M1020_SSL/TLS Inspection</a>: If it is possible to inspect HTTPS traffic, the captures can be analyzed for connections that appear to be domain fronting. </div></div>""",

        "plural_summary": """<div id="plural-summary">Traffic was found {{direction}} these IP addresses in the Perimeter Threat (PCAP) application. These IPs are an anonymization service. Traffic like this wants to remain anonymous, which can be a threat as cyber criminals use these services to scout particular targets. THREAT_INTEL_KEY DYNAMIC_CONTENT If the firewall settings allow it, please block the anonymization URLs. Otherwise, if there is no business requirement for this traffic, Gradient Cyber recommends blocking the below IPs.</div>
<div id="plural-summary">Gradient Cyber recommends applying the security measures across all your locations.</div>
<div id="plural-summary"><br></div>
<div id="plural-summary">Description: IP anonymization, also known as IP masking, is a method of replacing the original IP address with one that cannot be associated with or traced back to an individual user. This can be done by setting the last octet of IPV4 addresses or the last 80 bits of IPv6 addresses to zeros.</div>
<div id="plural-summary">
[Continues with same Technique and Mitigations sections as single summary]</div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">Traffic was found with Internet IP {{internet_ip}}. The traffic was initiated from {{The_Internet_or_The_Local_Network}}.  This IP is an anonymization service. Traffic like this wants to remain anonymous, which can be a threat as cyber criminals use these services to scout particular targets. THREAT_INTEL_KEY DYNAMIC_CONTENT If the firewall settings allow it, please block the anonymization URL. Otherwise, if there is no business requirement for this traffic, Gradient Cyber recommends blocking the above IP.</div>
[Continues with same format and sections as single summary]""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">Traffic was found with multiple IP addresses. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. These IPs are an anonymization service. Traffic like this wants to remain anonymous, which can be a threat as cyber criminals use these services to scout particular targets. THREAT_INTEL_KEY DYNAMIC_CONTENT If the firewall settings allow it, please block the anonymization URLs. Otherwise, if there is no business requirement for this traffic, Gradient Cyber recommends blocking the below IPs.</div>
[Continues with same format and sections as single summary]""",

        "event_description": """<div id="event-description">Anonymous Web surfing allows a user to visit Web sites without allowing anyone to gather information about which sites the user visited. Services that provide anonymity disable pop-up windows and cookies and conceal the visitor's IP address. These services typically use a proxy server to process each HTTP request. When the user requests a Web page by clicking a hyperlink or typing a URL into their browser, the service retrieves and displays the information using its own server. The remote server (where the requested Web page resides) receives information about the anonymous Web surfing service in place of the user's information. Anonymous Web surfing is popular for two reasons: to protect the user's privacy and/or to bypass blocking applications that would prevent access to Web sites or parts of sites that the user wants to visit. An anonymous surfing service can make a user feel more secure on the Internet, but it doesn't permit a site to often the visitor personalization. This means that the a site cannot tailor its content or advertising to suit the individual user. SafeWeb and the Anonymizer are the most commonly used such services. Lucent's Bell Labs created its own version, called Lucent Personalized Web Assistant (LPWA), as did the Naval Research Labs, whose project was called Onion Routing. For more information, please check this link: https://searchsecurity.techtarget.com/definition/anonymous-Web-surfing</div>""",

        "applications": """<div id="applications">Perimeter (CM)</div>"""
    },
   "Bots IP": {
        "name": """<div id="name">
Traffic with {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary">
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. This is a Bot IP. A bot (short for "robot") is a program that operates as an agent for a user or another program or simulates a human activity. 
<p>
Gradient Cyber recommends blocking the above IP. Please access the Files tab for advanced recommendations on web application security.  

DYNAMIC_CONTENT
<p>
Recommendations: 
1. All software should be kept up to date with security patches.
2. Users should be trained to refrain from activity that puts them at risk of bot infections or other malware. This includes opening emails or messages, downloading attachments, or clicking links from untrusted or unfamiliar sources.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary">
Traffic was found  {{direction}} these IP addresses in the Perimeter Threat (PCAP) application. These are Bot IPs. A bot (short for "robot") is a program that operates as an agent for a user or another program or simulates a human activity. 
<p>
Gradient Cyber recommends blocking the below IPs. Please access the Files tab for advanced recommendations on web application security.  

DYNAMIC_CONTENT
<p>
Recommendations: 
1. All software should be kept up to date with security patches.
2. Users should be trained to refrain from activity that puts them at risk of bot infections or other malware. This includes opening emails or messages, downloading attachments, or clicking links from untrusted or unfamiliar sources.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
Traffic was found with Internet IP {{internet_ip}}. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. The Internet IP was reported for BOT activity. A bot (short for "robot") is a program that operates as an agent for a user or another program or simulates a human activity. 
<p>
Gradient Cyber recommends blocking the above IP. Please access the Files tab for advanced recommendations on web application security.  

DYNAMIC_CONTENT
<p>
Recommendations: 
1. All software should be kept up to date with security patches.
2. Users should be trained to refrain from activity that puts them at risk of bot infections or other malware. This includes opening emails or messages, downloading attachments, or clicking links from untrusted or unfamiliar sources.
</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
Traffic was found with multiple IP addresses. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. These Internet IPs were reported for BOT activity. A bot (short for "robot") is a program that operates as an agent for a user or another program or simulates a human activity. 
<p>
Gradient Cyber recommends blocking the below IPs. Please access the Files tab for advanced recommendations on web application security.  

DYNAMIC_CONTENT
<p>
Recommendations: 
1. All software should be kept up to date with security patches.
2. Users should be trained to refrain from activity that puts them at risk of bot infections or other malware. This includes opening emails or messages, downloading attachments, or clicking links from untrusted or unfamiliar sources.
</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "event_description": """<div id="event-description">
A malicious use of bots is the coordination and operation of an automated attack on networked computers, such as a denial-of-service attack by a botnet. Internet bots can also be used to commit click fraud and more recently have seen usage around MMORPG games as computer game bots. Nowadays such kinds of bots are also being used in video games such as PUBG. PUBG mobile bots are also related to the family of malicious bots. A spambot is an internet bot that attempts to spam large amounts of content on the Internet, usually adding advertising links. More than 94.2% of websites have experienced a bot attack.
There are malicious bots (and botnets) of the following types:
Spambots that harvest email addresses from contact or guestbook pages
Downloader programs that suck bandwidth by downloading entire websites
Website scrapers that grab the content of websites and re-use it without permission on automatically generated doorway pages
Registration bots which sign up a specific email address to numerous services in order to have the confirmation messages flood the email inbox and distract from important messages indicating a security breach.
Viruses and worms
DDoS attacks
Botnets, zombie computers, etc.
Spambots that try to redirect people onto a malicious website, sometimes found in comment sections or forums of various websites.
Bots are also used to buy up good seats for concerts, particularly by ticket brokers who resell the tickets. Bots are employed against entertainment event-ticketing sites. The bots are used by ticket brokers to unfairly obtain the best seats for themselves while depriving the general public of also having a chance to obtain the good seats. The bot runs through the purchase process and obtains better seats by pulling as many seats back as it can.
Bots are often used in Massively Multiplayer Online Roleplaying Games to farm for resources that would otherwise take significant time or effort to obtain; this is a concern for most online in-game economies.[citation needed
Bots are also used to increase views for YouTube videos.
Bots are used to increase traffic counts on analytics reporting to extract money from advertisers. A study by Comscore found that 54 percent of display ads shown in thousands of campaigns between May 2012 and February 2013 never appeared in front of a human being.
Bots may be used on internet forums to automatically post inflammatory or nonsensical posts to disrupt the forum and anger users.
The most widely used anti-bot technique is the use of CAPTCHA, which is a form of Turing test used to distinguish between a human user and a less-sophisticated AI-powered bot, by the use of graphically-encoded human-readable text. Examples of providers include Recaptcha, and commercial companies such as Minteye, Solve Media, and NuCaptcha. Captchas, however, are not foolproof in preventing bots as they can often be circumvented by computer character recognition, security holes, and even by outsourcing captcha solving to cheap laborers.
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>"""
    },
"Gradient 365 alert: Unusual sign-in activity detected": {
        "name": """<div id="name">
Gradient 365 alert: Unusual sign-in activity detected
</div>""",

        "title": """<div id="title">
Gradient 365 alert: Unusual sign-in activity detected
</div>""",

        "single_summary": """<div id="single-summary">

<div>The user {add user} was involved in an unusual sign in incident. The user signed in from a never seen before IP address: {Add IP address along with Country info} at {Date and Time login occurred using local time}, using {Logon type}. If you determine that the IP in this Sign-in is safe, please let us know in the response summary section of the sitrep.
<br>
We have included the following links from several threat intel sources to provide additional information on the IP:
<br>
{Threat intel links}
<br>
Link to 365 Traffic {The word link should be a hyperlink to the 365 traffic}
<br>
This user has a user type of {UserType Value} which per Microsoft is a {Decode User Type}.
<br>
This alert falls under the following <a href="https://attack.mitre.org/tactics/TA0001/" target="_blank"><b>MITRE tactic: Initial Access - TA0001</b></a>
<br>
<b>Description</b>
Microsoft 365 offers cloud based services for a number of different applications such as Microsoft Word, Exchange Online, Azure Active Directory, SharePoint and OneDrive. This enables user to access their documents and services from anywhere with Internet access. Unfortunately this opens up the environment to attacks from pretty much anywhere so long as the attackers have gathered information on Valid Accounts.  
<br>
<b>Risk associated</b>
Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion. Compromised credentials may be used to bypass access controls placed on various resources on systems within the network and may even be used for persistent access to remote systems and externally available services, such as Azure AD, Outlook Web Access and Exchange Online. Compromised credentials may also grant an adversary increased privilege to specific systems or access to restricted areas of the network. Adversaries may choose not to use malware or tools in conjunction with the legitimate access those credentials provide to make it harder to detect their presence.
<br>
The overlap of permissions for cloud accounts across a network of systems is of concern because the adversary may be able to pivot across accounts and systems to reach a high level of access (i.e., domain or enterprise administrator) to bypass access controls set within the enterprise.
<br>
<b>Techniques</b>
1. <a href="https://attack.mitre.org/techniques/T1078/002/" target="_blank"><b>MITRE: Valid Accounts: Domain Accounts - T1078.002</b></a>.
2. <a href="https://attack.mitre.org/techniques/T1078/004/" target="_blank"><b>MITRE: Valid Accounts: Cloud Accounts - T1078.004</b></a>.
<p>
<b> Mitigations</b>
1. <a href="https://attack.mitre.org/mitigations/M1032/" target="_blank"><b>MITRE_M1032_Multi-factor Authentication</b></a>: Use multi-factor authentication for cloud accounts, especially privileged accounts. This can be implemented in a variety of forms (e.g. hardware, virtual, SMS), and can also be audited using administrative reporting features. Integrating multi-factor authentication (MFA) as part of organizational policy can also greatly reduce the risk of an adversary gaining control of valid credentials. Please refer to this Microsoft 365 link for more information on how to implement this. <a href="https://docs.microsoft.com/en-us/microsoft-365/admin/security-and-compliance/set-up-multi-factor-authentication?view=o365-worldwide" target="_blank"><b>[1]</b></a></li>
2.  <a href="https://attack.mitre.org/mitigations/M1027/" target="_blank"><b>MITRE_M1027_Password Policies</b></a>: Ensure strong password length and complexity for service accounts and that these passwords periodically expire. Refer to NIST guidelines when creating password policies. <a href="https://pages.nist.gov/800-63-3/sp800-63b.html" target="_blank"><b>[2]</b></a></li>
3. <a href="https://attack.mitre.org/mitigations/M1018/" target="_blank"><b>MITRE_M1018_User Account Management</b></a>: Manage the creation, modification, use, and roles associated to accounts. Review account roles routinely to look for those that could allow an adversary to gain wide access and proactively reset accounts that are known to be part of breached credentials either immediately, or after detecting a compromise. Microsoft 365 uses roles instead of assigning privileges to accounts please refer to this Microsoft document for more Information on how to manage roles in MS365.<a href="https://docs.microsoft.com/en-us/microsoft-365/admin/add-users/about-admin-roles?view=o365-worldwide" target="_blank"><b>[3]</b></a></li>
4. <a href="https://attack.mitre.org/mitigations/M1017/" target="_blank"><b>MITRE_M1017_User Training</b></a>: Train users to be aware of access or manipulation attempts by an adversary to reduce the risk of successful spearphishing, social engineering, and other techniques that involve user interaction.
<br>
<br>
Please let us know if this event was expected, if you have specific Office 365 properties that should be monitored, and <b>if occurrences similar to this are something you'd want to be informed of in the future.</b>

</div>""",

        "message_priority": """<div id="message-priority">
4: Low
</div>"""
    },
"Evaluated Addresses": {
        "name": """<div id="name">
Evaluated Addresses
</div>""",

        "title": """<div id="title">
Network Access
</div>""",

        "single_summary": """<div id="single-summary"> 
The below names/addresses were evaluated and open ports are identified below:
[[[ENTER TABLE HERE]]]
</div>""",

        "event_description": """<div id="event-description">
</div>""",

        "organizational_wordtags": """<div id="organizational-wordtags">
network
</div>""",

        "other_related_events": """<div id="other-related-events">
</div>""",

        "categories": """<div id="categories">
Cyber Threat Data
</div>""",

        "message_priority": """<div id="message-priority">
5: Informational
</div>"""
    },
 "Scanning performed using automation tools": {
        "name": """<div id="name">
        Scanning performed using automation tools
    </div>""",

        "single_summary": """<div id="single-summary">
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. According to our IDS entries and further analysis, there are traces of scanning performed using ADD TOOL HERE WITH HYPERLINK <a href="https://github.com/robertdavidgraham/masscan" target="_blank">Masscan</a> // <a href="https://www.openvas.org/" target="_blank">OpenVAS</a> // <a href="https://zmap.io/" target="_blank">ZGrab</a> // <a href="https://nmap.org/" target="_blank">Nmap</a> // <a href="https://www.extremenetworks.com/extreme-networks-blog/detecting-mirai-botnet-scans/" target="_blank">Mirai</a> // <a href="https://en.wikipedia.org/wiki/ZmEu_(vulnerability_scanner)" target="_blank">ZmEu</a> // <a href="https://www.tenable.com/products/nessus" target="_blank">Nessus</a> automation tool. Scanning is done by cyber criminals to assess their target and conduct reconnaissance before an attack or by organizations in order to assess their asset or application security.
<p>
COUNTRY_IP_ISP
DYNAMIC_CONTENT
<p>
<b>Description </b>
Network Scanning is the procedure of identifying active hosts, ports and the services used by the target application.</p>
<p>
<b>Risks Associated</b>
1. Adversaries may execute active reconnaissance scans to gather information that can be used during targeting. Active scans are those where the adversary probes victim infrastructure via network traffic, as opposed to other forms of reconnaissance that do not involve direct interaction. 
2. Adversaries may attempt to get a listing of services running on remote hosts, including those that may be vulnerable to remote software exploitation. Methods to acquire this information include port scans and vulnerability scans using tools that are brought onto a system.
3. Within cloud environments, adversaries may attempt to discover services running on other cloud hosts. Additionally, if the cloud environment is connected to a on-premises environment, adversaries may be able to identify services running on non-cloud systems as well.
<p>
<b>Technique</b>
<a href="https://attack.mitre.org/techniques/T1595/" target="_blank"><b>MITRE: Active Scanning - T1595</b></a>.
<a href="https://attack.mitre.org/techniques/T1046/" target="_blank"><b>MITRE: Network Service Scanning - T1046</b></a>.
<p>
<b>Mitigations</b>
1. <a href="https://attack.mitre.org/mitigations/M1056" target="_blank"><b>MITRE_M1056_Pre-compromise</b></a>: This technique cannot be easily mitigated with preventive controls since it is based on behaviors performed outside of the scope of enterprise defenses and controls. Efforts should focus on minimizing the amount and sensitivity of data available to external parties.
2. <a href="https://attack.mitre.org/mitigations/M1042/" target="_blank"><b>MITRE_M1042_Disable or Remove Feature or Program</b></a>: Ensure that unnecessary ports and services are closed to prevent risk of discovery and potential exploitation.
3. <a href="https://attack.mitre.org/mitigations/M1031/" target="_blank"><b>MITRE_M1031_Network Intrusion Prevention</b></a>: Use network intrusion detection/prevention systems to detect and prevent remote service scans.
4. <a href="https://attack.mitre.org/mitigations/M1030/" target="_blank"><b>MITRE_M1030_Network Segmentation</b></a>: Ensure proper network segmentation is followed to protect critical servers and devices.
5. Minimise Attack Surface.<a href="https://cheatsheetseries.owasp.org/cheatsheets/Attack_Surface_Analysis_Cheat_Sheet.html" target="_blank"><b>[1]</b></a></li>
6. Implement Zero-trust Policies.<a href="https://www.cisa.gov/zero-trust-maturity-model" target="_blank"><b>[2]</b></a></li>
<p>
Gradient Cyber would like to know if this was purposely initiated, and if not, we recommend blocking the above IP.
</p>
Please access the Files tab for advanced recommendations on web application security.  
</p><p><b>Gradient Cyber recommends applying the security measures across all your locations.</b></p></p></div>""",

        "plural_summary": """[Full plural summary content as provided]""",
        
        "bidirectional_single_summary": """[Full bidirectional single summary content as provided]""",
        
        "bidirectional_plural_summary": """[Full bidirectional plural summary content as provided]""",

        "event_description": """<div id="event-description">
        Network scanning is a procedure for identifying active hosts on a network, either for the purpose of attacking
        them or for network security assessment. Scanning procedures, such as ping sweeps and port scans, return
        information about which IP addresses map to live hosts that are active on the Internet and what services they
        offer. Another scanning method, inverse mapping, returns information about what IP addresses do not map to live
        hosts; this enables an attacker to make assumptions about viable addresses
        Scanning is one of three components of intelligence gathering for an attacker. In the foot printing phase, the
        attacker creates a profile of the target organization, with information such as its domain name system (DNS) and
        e-mail servers, and its IP address range. Most of this information is available online. In the scanning phase,
        the attacker finds information about the specific IP addresses that can be accessed over the Internet, their
        operating systems, the system architecture, and the services running on each computer. In the enumeration phase,
        the attacker gathers information such as network user and group names, routing tables, and Simple Network
        Management Protocol (SNMP) data.

        For more related information, please check this link:
        https://searchmidmarketsecurity.techtarget.com/definition/network-scanning.
    </div>""",

        "applications": """<div id="applications">
        Perimeter (CM)
    </div>""",

        "message_priority": """<div id="message-priority">
        5: Informational
    </div>"""
    },
    "Scanning IP": {
        "name": """<div id="name">
Traffic with {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary">
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. This is a scanning IP.  Scanning is done by cyber criminals to assess their target and conduct reconnaissance before an attack.
 <p>
This IP has been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Gradient Cyber recommends blocking the above IP. Please access the Files tab for instruction sets on mitigation solutions for web application, if applicable.
</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary">
Traffic was found  {{direction}} these IP addresses in the Perimeter Threat (PCAP) application. These are scanning IPs. Scanning is done by cyber criminals to assess their target and conduct reconnaissance before an attack. 
<p>
These IPs have been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Gradient Cyber recommends blocking the below IPs. Please access the Files tab for instruction sets on mitigation solutions for web application, if applicable.

</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
Traffic was found with Internet IP {{internet_ip}}. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. The Internet IP was reported for scanning. Scanning is done by cyber criminals to assess their target and conduct reconnaissance before an attack. 
<p>
This IP has been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Gradient Cyber recommends blocking the above IP. Please access the Files tab for instruction sets on mitigation solutions for web application, if applicable.
</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
Traffic was found with multiple IP addresses. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. These Internet IPs were reported for scanning. Scanning is done by cyber criminals to assess their target and conduct reconnaissance before an attack.
<p>
These IPs have been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY
<p>
Gradient Cyber recommends blocking the below IPs. Please access the Files tab for instruction sets on mitigation solutions for web application, if applicable.
</p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "event_description": """<div id="event-description">
Network scanning is a procedure for identifying active hosts on a network, either for the purpose of attacking them or for network security assessment. Scanning procedures, such as ping sweeps and port scans, return information about which IP addresses map to live hosts that are active on the Internet and what services they offer. Another scanning method, inverse mapping, returns information about what IP addresses do not map to live hosts; this enables an attacker to make assumptions about viable addresses
Scanning is one of three components of intelligence gathering for an attacker. In the foot printing phase, the attacker creates a profile of the target organization, with information such as its domain name system (DNS) and e-mail servers, and its IP address range. Most of this information is available online. In the scanning phase, the attacker finds information about the specific IP addresses that can be accessed over the Internet, their operating systems, the system architecture, and the services running on each computer. In the enumeration phase, the attacker gathers information such as network user and group names, routing tables, and Simple Network Management Protocol (SNMP) data.

For more related information, please check this link: https://searchmidmarketsecurity.techtarget.com/definition/network-scanning.
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
5: Informational
</div>"""
    },
 "TLS Traffic to bad domains": {
        "name": """<div id="name">
TLS Traffic to bad domains
</div>""",

        "title": """<div id="title">
TLS Traffic to bad domains
</div>""",

        "single_summary": """<div id="single-summary"> 
Our IDS has identified TLS handshake involving the below domain(s), associated with malicious activity by our threat intelligence sources.
<p>
vt_domains_key. 
ioc_sources_key
<p>
Please access the following links for screenshots: urlscan_domains_key.
<b>CHECK IF THERE IS A LINK, IF NOT, DELETE THE SCREENSHOTS PART AND THIS AS WELL</b>
<p>
link_to_ids_key
<p>
<b>Description and Risks associated with Bad Domains</b>
Malicious domains are identified by Gradient Cyber using different criteria:
1. Domains labeled as malicious/malware/phishing and scored by Virus Total Engine.
2. Domains labeled as malicious based on TLDs. Out of more than 1,000 TLDs, the top 25 TLDs (by the number of malicious domains) account for more than 90% of all malicious domain names. While these 25 TLDs are not malicious, they are well-positioned to help mitigate malicious domain registrations. We find that TLDs offering free domain registration are among the top preferred TLDs for phishing domains.<a href="https://unit42.paloaltonetworks.com/top-level-domains-cybercrime/" target="_blank"><b>[1]</b></a>
3. Domains labeled as malicious based on DGAs. Adversaries may make use of Domain Generation Algorithms (DGAs) to dynamically identify a destination domain for command and control traffic rather than relying on a list of static IP addresses or domains. This has the advantage of making it much harder for defenders to block, track, or take over the command and control channel, as there potentially could be thousands of domains that malware can check for instructions.<a href="https://umbrella.cisco.com/blog/domain-generation-algorithms-effective" target="_blank"><b>[2]</b></a><a href="https://unit42.paloaltonetworks.com/threat-brief-understanding-domain-generation-algorithms-dga/" target="_blank"><b>[3]</b></a>. DGAs can take the form of apparently random or "gibberish" strings (ex: istgmxdejdnxuyla[.]ru) when they construct domain names by generating each letter. Gradient Cyber uses Threat Intel to identify and label domains which are DGAs.
4. Domains that are labeled as risky and suspicious without garnering a score on Virus Total (check the Details Tab and look for the Categories section).
5. Gradient's own research into malicious domains that stems from malware reverse engineering and IR.
<p>
<b>Technique</b>
<a href="https://attack.mitre.org/techniques/T1568/002/" target="_blank"><b>MITRE: Dynamic Resolution: Domain Generation Algorithms - T1568.002</b></a>.
<p>
<b> Mitigations</b>
1. <a href="https://attack.mitre.org/mitigations/M1031/" target="_blank"><b>MITRE_M1031_Network Intrusion Prevention</b></a>: Network intrusion detection and prevention systems that use network signatures to identify traffic for specific adversary malware can be used to mitigate activity at the network level. Malware researchers can reverse engineer malware variants that use DGAs and determine future domains that the malware will attempt to contact, but this is a time and resource intensive effort.<a href="https://umbrella.cisco.com/blog/domain-generation-algorithms-effective" target="_blank"><b>[2]</b></a><a href="https://umbrella.cisco.com/blog/at-high-noon-algorithms-do-battle" target="_blank"><b>[6]</b></a> Malware is also increasingly incorporating seed values that can be unique for each instance, which would then need to be determined to extract future generated domains. In some cases, the seed that a particular sample uses can be extracted from DNS traffic. Even so, there can be thousands of possible domains generated per day; this makes it impractical for defenders to preemptively register all possible C2 domains due to the cost.
2. <a href="https://attack.mitre.org/mitigations/M1031/" target="_blank"><b>MITRE_M1031_Restrict Web-Based Content</b></a>: In some cases a local DNS sinkhole may be used to help prevent DGA-based command and control at a reduced cost.
3. Blocking the URI using a content filter or ACL.
4. Investigate for any signs of infection in the local network.
<p>
  
Please access the Files tab for instruction sets on mitigation solutions for Windows and Palo Alto firewalls, if applicable.
<p>
</p></p></p><p><b>Gradient Cyber recommends applying the security measures across all your locations. Please notify the analyst if the above measures are not scalable across the entire organization.</b></p>
<p>
</p></div>""",

        "event_description": """<div id="event-description">
For more information please refer to the links given below:
http://blog.talosintelligence.com/2017/09/avast-distributes-malware.html
https://umbrella.cisco.com/blog/domain-generation-algorithms-effective
https://unit42.paloaltonetworks.com/threat-brief-understanding-domain-generation-algorithms-dga/
https://www.mandiant.com/resources/dissecting-one-ofap
https://www.welivesecurity.com/2017/12/21/sednit-update-fancy-bear-spent-year/
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
3: Medium
</div>""",

        "other_related_events": """<div id="other-related-events">
Malware
</div>"""
    },
"Gradient 365 alert: Sign-in from a Blacklisted IP detected": {
        "name": """<div id="name">
Gradient 365 alert: Sign-in from a Blacklisted IP detected
</div>""",

        "title": """<div id="title">
Gradient 365 alert: Sign-in from a Blacklisted IP detected
</div>""",

        "single_summary": """<div id="single-summary">

<div>The user {Add user} logged in from the following blacklisted ip address, {IP address with location info}, at {date and time using local time}, using {Login Type}. If you determine that the IP in this Sign-in is safe, please let us know in the response summary section of the sitrep.
<br>
We have included the following links from several threat intel sources to provide additional information on the IP:
<br>
{Threat intel links}
<br>
Link to 365 Traffic {The word link should be a hyperlink to the 365 traffic}
<br>
This user has a user type of {UserType Value} which per Microsoft is a {Decode User Type}.
<br>
This alert falls under the following <a href="https://attack.mitre.org/tactics/TA0001/" target="_blank"><b>MITRE tactic: Initial Access - TA0001</b></a>
<br>
<b>Description</b>
Microsoft 365 offers cloud based services for a number of different applications such as Microsoft Word, Exchange Online, Azure Active Directory, SharePoint and OneDrive. This enables user to access their documents and services from anywhere with Internet access. Unfortunately this opens up the environment to attacks from pretty much anywhere so long as the attackers have gathered information on Valid Accounts.  
<br>
<b>Risk associated</b>
Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion. Compromised credentials may be used to bypass access controls placed on various resources on systems within the network and may even be used for persistent access to remote systems and externally available services, such as Azure AD, Outlook Web Access and Exchange Online. Compromised credentials may also grant an adversary increased privilege to specific systems or access to restricted areas of the network. Adversaries may choose not to use malware or tools in conjunction with the legitimate access those credentials provide to make it harder to detect their presence.
<br>
The overlap of permissions for cloud accounts across a network of systems is of concern because the adversary may be able to pivot across accounts and systems to reach a high level of access (i.e., domain or enterprise administrator) to bypass access controls set within the enterprise.
<br>
<b>Techniques</b>
1. <a href="https://attack.mitre.org/techniques/T1078/002/" target="_blank"><b>MITRE: Valid Accounts: Domain Accounts - T1078.002</b></a>.
2. <a href="https://attack.mitre.org/techniques/T1078/004/" target="_blank"><b>MITRE: Valid Accounts: Cloud Accounts - T1078.004</b></a>.
<p>
<b> Mitigations</b>
1. <a href="https://attack.mitre.org/mitigations/M1032/" target="_blank"><b>MITRE_M1032_Multi-factor Authentication</b></a>: Use multi-factor authentication for cloud accounts, especially privileged accounts. This can be implemented in a variety of forms (e.g. hardware, virtual, SMS), and can also be audited using administrative reporting features. Integrating multi-factor authentication (MFA) as part of organizational policy can also greatly reduce the risk of an adversary gaining control of valid credentials. Please refer to this Microsoft 365 link for more information on how to implement this. <a href="https://docs.microsoft.com/en-us/microsoft-365/admin/security-and-compliance/set-up-multi-factor-authentication?view=o365-worldwide" target="_blank"><b>[1]</b></a></li>
2.  <a href="https://attack.mitre.org/mitigations/M1027/" target="_blank"><b>MITRE_M1027_Password Policies</b></a>: Ensure strong password length and complexity for service accounts and that these passwords periodically expire. Refer to NIST guidelines when creating password policies. <a href="https://pages.nist.gov/800-63-3/sp800-63b.html" target="_blank"><b>[2]</b></a></li>
3. <a href="https://attack.mitre.org/mitigations/M1018/" target="_blank"><b>MITRE_M1018_User Account Management</b></a>: Manage the creation, modification, use, and roles associated to accounts. Review account roles routinely to look for those that could allow an adversary to gain wide access and proactively reset accounts that are known to be part of breached credentials either immediately, or after detecting a compromise. Microsoft 365 uses roles instead of assigning privileges to accounts please refer to this Microsoft document for more Information on how to manage roles in MS365.<a href="https://docs.microsoft.com/en-us/microsoft-365/admin/add-users/about-admin-roles?view=o365-worldwide" target="_blank"><b>[3]</b></a></li>
4. <a href="https://attack.mitre.org/mitigations/M1017/" target="_blank"><b>MITRE_M1017_User Training</b></a>: Train users to be aware of access or manipulation attempts by an adversary to reduce the risk of successful spearphishing, social engineering, and other techniques that involve user interaction.
<br>
<br>
Please let us know if this event was expected, if you have specific Office 365 properties that should be monitored, and <b>if occurrences similar to this are something you'd want to be informed of in the future.</b>

</div>""",

        "message_priority": """<div id="message-priority">
3: Medium
</div>"""
    },
"Social Engineering": {
        "name": """<div id="name">
Possible Social Engineering Attempt
</div>""",

        "title": """<div id="title">
Possible Social Engineering Attempt
</div>""",

        "single_summary": """<div id="single-summary"> 
Our IDS has issued an alert for a possible social engineering attempt. The VirusTotal link to the URL is: vt_domains_key. The Google Safe Browsing link to the URL is: gsb_domains_key.
Social engineering is content that tricks visitors into doing something dangerous, such as revealing confidential information or downloading software.
Please access the following links for screenshots: urlscan_domains_key.
link_to_ids_key
Check the involved assets for any signs of infection.
<b>Gradient Cyber recommends taking the following steps to mitigate subsequent threats:</b>
<ul>
<li>Blocking the access to the above URL.</li>
</ul>
Please access Files tab for instruction sets on mitigation solutions for Windows and Palo Alto firewalls, if applicable.
<p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary"> 
Our IDS has issued an alert for a possible social engineering attempts. The VirusTotal link to the URL is: vt_domains_key. The Google Safe Browsing link to the URL is: gsb_domains_key.
Social engineering is content that tricks visitors into doing something dangerous, such as revealing confidential information or downloading software.
Please access the following links for screenshots: urlscan_domains_key.
link_to_ids_key
Check the involved assets for any signs of infection.
<b>Gradient Cyber recommends taking the following steps to mitigate subsequent threats:</b>
<ul>
<li>Blocking the access to the above URLs.</li>
</ul>
Please access Files tab for instruction sets on mitigation solutions for Windows and Palo Alto firewalls, if applicable.
<p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "event_description": """<div id="event-description">
What is social engineering?
A social engineering attack is when a web user is tricked into doing something dangerous online.
There are different types of social engineering attacks:
Phishing: The site tricks users into revealing their personal information (for example, passwords, phone numbers, or credit cards). In this case, the content pretends to act, or looks and feels, like a trusted entity — for example, a browser, operating system, bank, or government.
Deceptive content: The content tries to trick you into doing something you'd only do for a trusted entity — for example, sharing a password, calling tech support, downloading software, or the content contains an ad that falsely claims that device software is out-of-date, prompting users into installing unwanted software.
Insufficiently labeled third-party services: A third-party service is someone that operates a site or service on behalf of another entity. If you (third party) operate a site on behalf of another (first) party without making the relationship clear, that might be flagged as social engineering. For example, if you (first party) run a charity website that uses a donation management website (third party) to handle collections for your site, the donation site must clearly identify that it is a third-party platform acting on behalf of that charity site, or else it could be considered social engineering.
For more information, check https://support.google.com/webmasters/answer/6350487?hl=en.
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>""",

        "message_priority": """<div id="message-priority">
3: Medium
</div>""",

        "other_related_events": """<div id="other-related-events">
Social Engineering
</div>"""
    },
"Tor IP": {
        "name": """<div id="name">
Traffic with {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary"> 
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. This IP has been labeled as <a href="https://attack.mitre.org/software/S0183/" target="_blank"><b>Tor</b></a> by our threat intelligence sources.
This IP has been reported for malicious activities by the intelligence sources listed below: 
THREAT_INTEL_KEY 
DYNAMIC_CONTENT
<b>Description and Risks associated with Tor</b>
<a href="https://attack.mitre.org/software/S0183/" target="_blank"><b>Tor</b></a> is a software suite and network that provides increased anonymity on the Internet. It creates a multi-hop proxy network and utilizes multilayer encryption to protect both the message and routing information. Tor utilizes "Onion Routing," in which messages are encrypted with multiple layers of encryption; at each step in the proxy network, the topmost layer is decrypted and the contents forwarded on to the next node until it reaches its destination. 
This type of communication can be used to attempt to avoid detection while conducting an XFIL (exfiltration) operation. An XFIL operation is used to steal data from an organization. 
<p>
<b>Software</b>
 <a href="https://attack.mitre.org/software/S0183/" target="_blank"><b>MITRE_Tor</b></a>
<p>
<b>Techniques</b>
1. <a href="https://attack.mitre.org/techniques/T1573/002/" target="_blank"><b>MITRE: Encrypted Channel: Asymmetric Cryptography - T1573.002</b></a>: <a href="https://attack.mitre.org/software/S0183/" target="_blank"><b>Tor</b></a> encapsulates traffic in multiple layers of encryption, using TLS by default.
2. <a href="https://attack.mitre.org/techniques/T1090/003/" target="_blank"><b>MITRE: Proxy: Multi-hop Proxy - T1090.003</b></a>: Traffic traversing the <a href="https://attack.mitre.org/software/S0183/" target="_blank"><b>Tor</b></a> network will be forwarded to multiple nodes before exiting the <a href="https://attack.mitre.org/software/S0183/" target="_blank"><b>Tor</b></a> network and continuing on to its intended destination.
<p>
<b> Mitigations</b>
1. <a href="https://attack.mitre.org/mitigations/M1031/" target="_blank"><b>MITRE_M1031_Network Intrusion Prevention</b></a>: Network intrusion detection and prevention systems that use network signatures to identify traffic for specific adversary malware can be used to mitigate activity at the network level. 
2. <a href="https://attack.mitre.org/mitigations/M1020/" target="_blank"><b>MITRE_M1020_SSL/TLS Inspection</b></a>: SSL/TLS inspection can be used to see the contents of encrypted sessions to look for network-based indicators of malware communication protocols.
3. Block any applications that are categorized as unknown-TCP, unknown-UDP, and unknown-p2p in your network.
4. Gradient Cyber recommends blocking the above IP if there is no business requirement for it.
<p>
 Please access the Files tab for advanced recommendations on web application security.  
</p><p><b>Gradient Cyber recommends applying the security measures across all your locations.</b></p>
</div>""",

        "plural_summary": """[Full plural summary content as provided]""",
        
        "bidirectional_single_summary": """[Full bidirectional single summary content as provided]""",
        
        "bidirectional_plural_summary": """[Full bidirectional plural summary content as provided]""",

        "event_description": """<div id="event-description">
[Full detailed event description about Tor networks, anonymity, and related topics as provided]
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>"""
    },
"Spam IP": {
        "name": """<div id="name">
Traffic with {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary">
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. 
This IP has been labeled as spam by: <b>// add your source in here //.</b>
Spam emails are sent out in mass quantities by the spammers and cybercriminals that are looking to make money from the recipients that actually respond to the message or click on the URLs contained within the message. They run phishing scams to obtain passwords, identity details, credit card numbers, bank account details &amp; more. They also spread malicious code onto recipients' computers. 
<p>
Gradient Cyber recommends blocking the above IP. 

DYNAMIC_CONTENT
<p>
Recommendations:
1. Train users to validate attachments before opening them.
2. Keep applications and operating systems running at the current released patch level.
3. Ensure anti-virus software and associated files are up to date.
4. Use spam filters to reduce spam traffic to email servers.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary">
Traffic was found  {{direction}} these IP addresses in the Perimeter Threat (PCAP) application. 
These IPs have been labeled as spam by: <b>// add your source in here //.</b>
Spam emails are sent out in mass quantities by the spammers and cybercriminals that are looking to make money from the recipients that actually respond to the message or click on the URLs contained within the message. They run phishing scams to obtain passwords, identity details, credit card numbers, bank account details &amp; more. They also spread malicious code onto recipients' computers.
<p>
Gradient Cyber recommends blocking the below IPs.

DYNAMIC_CONTENT
<p>
Recommendations:
1. Train users to validate attachments before opening them.
2. Keep applications and operating systems running at the current released patch level.
3. Ensure anti-virus software and associated files are up to date.
4. Use spam filters to reduce spam traffic to email servers.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
Traffic was found with Internet IP {{internet_ip}}. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. The Internet IP has been labeled as spam by: <b>// add your source in here //.</b>
Spam emails are sent out in mass quantities by the spammers and cybercriminals that are looking to make money from the recipients that actually respond to the message or click on the URLs contained within the message. They run phishing scams to obtain passwords, identity details, credit card numbers, bank account details &amp; more. They also spread malicious code onto recipients' computers. 
<p>
Gradient Cyber recommends blocking the above IP. 

DYNAMIC_CONTENT
<p>
Recommendations:
1. Train users to validate attachments before opening them.
2. Keep applications and operating systems running at the current released patch level.
3. Ensure anti-virus software and associated files are up to date.
4. Use spam filters to reduce spam traffic to email servers.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
Traffic was found with multiple IP addresses. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. These IPs have been labeled as spam by: <b>// add your source in here //.</b>
Spam emails are sent out in mass quantities by the spammers and cybercriminals that are looking to make money from the recipients that actually respond to the message or click on the URLs contained within the message. They run phishing scams to obtain passwords, identity details, credit card numbers, bank account details &amp; more. They also spread malicious code onto recipients' computers.
<p>
Gradient Cyber recommends blocking the below IPs.   

DYNAMIC_CONTENT
<p>
Recommendations:
1. Train users to validate attachments before opening them.
2. Keep applications and operating systems running at the current released patch level.
3. Ensure anti-virus software and associated files are up to date.
4. Use spam filters to reduce spam traffic to email servers.
</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "event_description": """<div id="event-description">
Spam is the electronic equivalent of the 'junk mail' that arrives on your doormat or in your postbox. However, spam is more than just annoying. It can be dangerous – especially if it's part of a phishing scam.

Spam emails are sent out in mass quantities by spammers and cybercriminals that are looking to do one or more of the following:

a. Make money from the small percentage of recipients that actually respond to the message
b. Run phishing scams – in order to obtain passwords, credit card numbers, bank account details, and more
c. Spread malicious code onto recipients' computers

How to protect yourself against spam:

1. Set up multiple email addresses: Private email address  / Public email address 
2. Never respond to any spam 
3. Think before you click 'unsubscribe' 
4. Keep your browser updated 
5. Use anti-spam filters 

For more related information, please check these links: 
https://www.kaspersky.com/resource-center/threats/spam-phishing
https://www.malwarebytes.com/spam
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>"""
    },
"NTP TOR IP": {
        "name": """<div id="name">
Traffic with {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary"> 
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. This IP is a TOR IP. TOR is a common anonymization service. Traffic like this is someone who wants to remain anonymous, which can be a threat as cybercriminals use these services for scouting particular targets. This type of communication can also be used to attempt to avoid detection while conducting an XFIL (exfiltration) operation. An XFIL operation is used to steal data from an organization. Additionally, the source port for this traffic is  <a href="https://www.speedguide.net/port.php?port=123" target="_blank">123</a>, which could indicate the possibility of an NTP attack. 
<p>
Gradient Cyber recommends blocking the above IP.

DYNAMIC_CONTENT
<p>
Recommendations:
1. Use only a few devices for external NTP
2. Point all external NTP requests to trusted external time servers, such as NIST time servers. List of NIST time servers can be found at: https://tf.nist.gov/tf-cgi/servers.cgi
3. Have all other assets point to trusted internal time servers. 

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary">
Traffic was found {{direction}} these IP addresses in the Perimeter Threat (PCAP) application. These are TOR IPs. TOR is a common anonymization service. Traffic like this is someone who wants to remain anonymous, which can be a threat as cybercriminals use these services for scouting particular targets. This type of communication can also be used to attempt to avoid detection while conducting an XFIL (exfiltration) operation. An XFIL operation is used to steal data from an organization. Additionally, the source port for this traffic is <a href="https://www.speedguide.net/port.php?port=123" target="_blank">123</a>, which could indicate the possibility of an NTP attack. 
<p>
Gradient Cyber recommends blocking the below IPs.

DYNAMIC_CONTENT
<p>
Recommendations:
1. Use only a few devices for external NTP
2. Point all external NTP requests to trusted external time servers, such as NIST time servers. List of NIST time servers can be found at: https://tf.nist.gov/tf-cgi/servers.cgi
3. Have all other assets point to trusted internal time servers.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
Traffic was found with Internet IP {{internet_ip}}. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. Internet IP is a TOR IP. TOR is a common anonymization service. Traffic like this is someone who wants to remain anonymous, which can be a threat as cybercriminals use these services for scouting particular targets. This type of communication can also be used to attempt to avoid detection while conducting an XFIL (exfiltration) operation. An XFIL operation is used to steal data from an organization. Additionally, the source port for this traffic is  <a href="https://www.speedguide.net/port.php?port=123" target="_blank">123</a>, which could indicate the possibility of an NTP attack. 
<p>
Gradient Cyber recommends blocking the above IP.

DYNAMIC_CONTENT
<p>
Recommendations:
1. Use only a few devices for external NTP
2. Point all external NTP requests to trusted external time servers, such as NIST time servers. List of NIST time servers can be found at: https://tf.nist.gov/tf-cgi/servers.cgi
3. Have all other assets point to trusted internal time servers. 
</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
Traffic was found with multiple IP addresses. The traffic was initiated from {{The_Internet_or_The_Local_Network}}. The Internet IPs are TOR IPs. TOR is a common anonymization service. Traffic like this is someone who wants to remain anonymous, which can be a threat as cybercriminals use these services for scouting particular targets. This type of communication can also be used to attempt to avoid detection while conducting an XFIL (exfiltration) operation. An XFIL operation is used to steal data from an organization. Additionally, the source port for this traffic is <a href="https://www.speedguide.net/port.php?port=123" target="_blank">123</a>, which could indicate the possibility of an NTP attack. 
<p>
Gradient Cyber recommends blocking the below IPs.

DYNAMIC_CONTENT
<p>
Recommendations:
1. Use only a few devices for external NTP
2. Point all external NTP requests to trusted external time servers, such as NIST time servers. List of NIST time servers can be found at: https://tf.nist.gov/tf-cgi/servers.cgi
3. Have all other assets point to trusted internal time servers.

</p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "event_description": """<div id="event-description">
Often, organizations that support free services on the internet (NTP, TOR, etc.) utilize a shared infrastructure for these services. At first look, it might be thought that utilizing a shared device for NTP that also hosts TOR would seem innocuous. However, NTP has been/continues to be harnessed for DDoS (NTP amplification), as well as exfiltration of data from organizations. As such @RISK is providing this guidance. 

Information Required: 
Determine whether the telecommunications/internet/data-center/cloud provider) has an upstream NTP server that is provided (not just a DNS CNAME pointer to another external service such as pool.ntp.org) free of additional charge 
Check default gateway on internet provider (often times this will also be an NTP service point as well. 

Guidance: 
Utilize telecommunications/internet/data-center/cloud provider free of additional charge NTP service, skip to guidance #3 if this is available 
ONLY utilize US-based NTP services like 0.us.pool.ntp.org, 1.us.pool.ntp.org, 2.us.pool.ntp.org, and 3.us.pool.ntp.org 
ONLY configure NTP on the private network with 2-3 nodes communicating with the outside world for NTP, configure ALL other machines/VMs/devices to query NTP from internal NTP servers 
DO NOT utilize TOR exit nodes for NTP - If NTP traffic (UDP port 123) is observed in exchange with a TOR node, then blocking that TOR node will cause NTP to switch to another NTP server.
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>"""
    },
"Malware IP": {
        "name": """<div id="name">
Traffic with {{threatType}} - {{country}} 
</div>""",

        "single_summary": """<div id="single-summary">
Traffic was found {{direction}} the above IP address in the Perimeter Threat (PCAP) application. This IP is hosting malware domain(s) identified by our IDS in your traffic (please check SNI column). Traffic like this is used to infect endpoints with Malware. 

The IP/domain(s) are part of the campaign: [add source and hyperlink]
PLEASE ADD THE MALWARE URL IF IT APPLIES <b>The VirusTotal link to the malware URL hosted by this IP is <b>link</b> </b>.
For a detailed overview of the IP, please check the VirusTotal <b>link</b>.
<p>
Gradient Cyber recommends taking the following steps to mitigate subsequent threats:
- Blocking the URI using a content filter or ACL;
- DNS blackhole;
- If there is no business requirement for traffic with the above IP, blacklist it;
- Investigate for any signs of infection in the local network.

Please access the Files tab for instruction sets on mitigation solutions for web application, Windows and Palo Alto firewalls, if applicable.

DYNAMIC_CONTENT
<p>
Recommendations:
1. Train users to validate attachments before opening them.
2. Keep updated patches on all critical and non-critical systems.
3. Use firewall, anti-malware, anti-ransomware, and anti-exploit technology.
4. Ensure anti-virus software and associated files are up to date.
5. Obtain applications only from their legitimate sources.
6. Use strong passwords and/or password managers.

</p></p></p><p>Gradient Cyber recommends applying the security measures across all your locations.</p></div>""",

        "plural_summary": """<div id="plural-summary">
[Full plural summary content as provided]</div>""",

        "bidirectional_single_summary": """<div id="bidirectional-single-summary">
[Full bidirectional single summary content as provided]</div>""",

        "bidirectional_plural_summary": """<div id="bidirectional-plural-summary">
[Full bidirectional plural summary content as provided]</div>""",

        "event_description": """<div id="event-description">
Malware, also known as "malicious software," can be classified in several ways in order to distinguish the unique types of malware from each other. Distinguishing and classifying different types of malware from each other is important to better understanding how they can infect computers and devices, the threat level they pose, and how to protect against them.

Individual malware programs often include several malicious functions and propagation routines – and, without some additional classification rules, this could lead to confusion.

For example, a specific malicious program may be capable of being spread via an email attachment and also as files via P2P networks. The program may also have the ability to harvest email addresses from an infected computer, without the consent of the user. With this range of functions, the program could be correctly classified as an Email-Worm, a P2P-Worm, or a Trojan-Mailfinder.

The following is a list of common types of malware, but it's hardly exhaustive:

Virus: Like their biological namesakes, viruses attach themselves to clean files and infect other clean files. They can spread uncontrollably, damaging a system's core functionality, and deleting or corrupting files. They usually appear as an executable file (.exe).

Trojans: This kind of malware disguises itself as legitimate software, or is hidden in legitimate software that has been tampered with. It tends to act discreetly and create backdoors in your security to let other malware in.

Spyware: Spyware is malware designed to spy on you. It hides in the background and takes notes on what you do online, including your passwords, credit card numbers, surfing habits, and more.

Worms: Worms infect entire networks of devices, either local or across the internet, by using network interfaces. It uses each consecutively infected machine to infect others.

Ransomware: This kind of malware typically locks down your computer and your files, and threatens to erase everything unless you pay a ransom.

Adware: Though not always malicious in nature, aggressive advertising software can undermine your security just to serve you ads — which can give other malware an easy way in. Plus, let's face it: pop-ups are really annoying.

Botnets: Botnets are networks of infected computers that are made to work together under the control of an attacker.

Some tips on how to protect against all classes of malware:

1. Always scan email attachments before you open them, even if they are presented as text files or if they came from people you know. Viruses often disguise themselves by forging e-mail addresses in order to trick users into opening infected attachments.
2. Keep your software updated. Most of nowadays threats are based on security vulnerabilities.
3. Install only original software
4. Perform antivirus scanning for every new program installed on your system
5. Be careful when using any type of shared folders. In case you want to share information on your computer, make sure not to allow full control permissions unless absolutely needed. Moreover, try to share only specific folders instead of large directories, such as full drives or entire data partitions.
6. Don't open files received via instant messengers without a virus check. Try to contact the sender to see if it was really his/her intention to send you this file. Many worms have the ability to send themselves without the knowledge of the user.

For more related information, please check these links:
https://www.kaspersky.com/resource-center/threats/malware-classifications
https://www.kaspersky.com/resource-center/preemptive-safety/what-is-malware-and-how-to-protect-against-it
https://blog.malwarebytes.com/101/2016/08/10-easy-ways-to-prevent-malware-infection/
https://www.avg.com/en/signal/what-is-malware
</div>""",

        "message_priority": """<div id="message-priority">
3: Medium
</div>""",

        "applications": """<div id="applications">
Perimeter (CM)
</div>"""
    }
}

class SitrepAnalyzer:
   def __init__(self):
       # Only use environment variable for API key
       self.openai_api_key = os.getenv("OPENAI_API_KEY") 
       if not self.openai_api_key:
           raise ValueError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY.")
           
       openai.api_key = self.openai_api_key
       self.llm = ChatOpenAI(
           model_name="gpt-4o-mini",
           temperature=0.1,
           openai_api_key=self.openai_api_key
       )
       
   def extract_client_metadata(self, query: str) -> Dict[str, str]:
    """
    Extract client metadata (name, timestamp) and clean query content
    """
    metadata_prompt = """
    Given this message, extract metadata and content.
    Rules:
    1. IF the message starts with a name followed by timestamp, extract them
    2. IF the message is just a question/comment, treat entire text as content
    3. Remove any metadata from content
if
    4. Check the content, if the content is just a information just say the greeting message like thank you...etc make by own llm
else

    Input: "{query}"

    Return exactly in this format (include all fields):
    {{"name": "extracted name or null",
    "timestamp": "extracted timestamp or null",
    "content": "cleaned message content"}}

    Examples:
    Input: "Wade Jones, Tue, 29 Oct 2024 15:34:26 GMT\nNot sure I understand what you are trying to tell me?"
    Output: {{"name": "Wade Jones", "timestamp": "Tue, 29 Oct 2024 15:34:26 GMT", "content": "Not sure I understand what you are trying to tell me?"}}

    Input: "Not sure I understand what you are trying to tell me?"
    Output: {{"name": null, "timestamp": null, "content": "Not sure I understand what you are trying to tell me?"}}
    """
    
    try:
        response = self.llm.predict(metadata_prompt.format(query=query))
        metadata = json.loads(response)
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {
            "name": None,
            "timestamp": None,
            "content": query
        }

   def find_matching_template(self, alert_summary: str) -> str:
       """Find matching template using LLM understanding"""
       template_matching_prompt = f"""
       Given a security alert summary, please analyze it against these 19 security templates and identify the single most appropriate matching template. Consider:

1. Core security threat/issue being described
2. Technical indicators and patterns
3. Type of traffic or activity involved
4. Associated risks and implications
5. Recommended mitigations

Based on this analysis, determine which ONE template best matches the scenario and explain why this template is the most relevant match compared to others.
For example, if you share this alert:
[Insert your security alert summary here]
I will analyze it and:

1. Identify the key elements of the alert
2. Compare it against all 19 templates
3. Select the single best matching template
4. Explain why this template is the most appropriate match
5. Note any important variations from the template

       Alert Summary: {alert_summary}

       Available Templates and their contexts:
       1. Anomalous Internal Traffic

Severity: Medium to High
Context: Detecting unusual traffic patterns between internal assets
Indicators:

Sudden increases in east-west traffic
Unusual port usage between internal systems
Traffic between segments that don't normally communicate


Response Actions:

Baseline comparison analysis
Asset relationship mapping
Network segmentation review
Process tree analysis on involved systems



2. Blacklisted IP

Severity: High
Context: Traffic involving known malicious IP addresses
Additional Monitoring:

Geographic location analysis
Historical communication patterns
Associated domain analysis
Type of blacklist (Command & Control, Malware, etc.)


Response Actions:

Immediate traffic blocking
System isolation if internal asset involved
Threat hunting on communicating assets
IOC sharing with security team



DNS & Domain Threats
3. DNS Queries to Bad Domains

Severity: High
Context: Detection of DNS queries to known malicious domains
Enhanced Detection:

DGA (Domain Generation Algorithm) pattern analysis
DNS tunneling detection
Newly registered domain checks
Domain reputation scoring


Response Actions:

DNS query blocking
Asset investigation
DNS log analysis
C2 communication pattern analysis



Volume-Based Anomalies
4. Anomalous Internet Traffic: Size

Severity: Medium to High
Context: Unusual increases in traffic volume
Thresholds:

Baseline deviation percentages
Time-based analysis (business hours vs. non-business hours)
Per-asset historical comparison


Additional Monitoring:

Data exfiltration patterns
Compression ratio analysis
Protocol analysis
Destination categorization



5. Anomalous Internet Traffic: Packets

Severity: Medium
Context: Unusual packet count increases
Analysis Factors:

Packet size distribution
Protocol distribution
Time-based patterns
Source/destination relationship analysis


Response Actions:

Traffic pattern analysis
Protocol behavior analysis
Network baseline comparison
Asset behavior profiling



6. Anomalous Internet Traffic: Sessions

Severity: Medium to High
Context: Unusual session count increases
Session Analysis:

Duration patterns
Connection frequency
Protocol distribution
Peer relationship analysis


Response Actions:

Session tracking
Connection analysis
User behavior analysis
Application usage review



Special Categories
7. Anonymization Services IP

Severity: Medium to High
Context: Traffic involving anonymization service IPs
Service Types:

VPN services
Proxy servers
Privacy networks
Residential proxy networks


Response Actions:

Policy compliance check
User validation
Traffic pattern analysis
Business justification review



8. Bots IP

Severity: Medium to High
Context: Known bot network traffic
Bot Categories:

Crawler bots
Scraping bots
DDoS bots
Spam bots


Response Actions:

Bot behavior analysis
Traffic pattern review
Rate limiting implementation
Access control review



Microsoft 365 Security
9. Gradient 365 Alert: Unusual Sign-in Activity

Severity: High
Context: Anomalous Microsoft 365 authentication
Detection Parameters:

Geographic location analysis
Device fingerprinting
Time-based patterns
Multi-factor authentication status


Response Actions:

Account security review
Authentication log analysis
Conditional access policy review
User notification and verification



10. Evaluated Addresses

Severity: Variable
Context: IP investigation tracking
Evaluation Criteria:

Historical behavior
Reputation scores
Associated incidents
Network relationship analysis


Response Actions:

Threat intelligence enrichment
Pattern analysis
Risk scoring
Watch list management



Scanning & Reconnaissance
11. Scanning with Automation Tools

Severity: Medium to High
Context: Automated scanning detection
Tool Categories:

Vulnerability scanners
Port scanners
Network mappers
Web application scanners


Response Actions:

Tool identification
Scan pattern analysis
Asset exposure review
Security control assessment



12. Scanning IP

Severity: Medium to High
Context: IP-based scanning behavior
Scanning Patterns:

Sequential port scanning
Distributed scanning
Service enumeration
Vulnerability probing


Response Actions:

IP blocking
Asset hardening
Exposure analysis
Security control review



Encryption & Protocol Threats
13. TLS Traffic to Bad Domains

Severity: High
Context: Encrypted traffic to malicious domains
Analysis Points:

Certificate analysis
TLS version monitoring
Cipher suite analysis
JA3 fingerprinting


Response Actions:

Traffic blocking
Certificate analysis
Domain reputation check
SSL/TLS inspection



Advanced Threat Categories
14. Gradient 365 Alert: Blacklisted IP Sign-in

Severity: Critical
Context: Microsoft 365 access from known bad IPs
Enhanced Monitoring:

Account privilege level
Access pattern analysis
Resource access tracking
Authentication method analysis


Response Actions:

Immediate account security review
Access termination
User notification
Security policy review



15. Social Engineering

Severity: High
Context: Social engineering threat detection
Attack Vectors:

Phishing attempts
Impersonation attacks
Pretexting scenarios
Business email compromise


Response Actions:

User awareness training
Communication pattern analysis
Security awareness campaigns
Policy enforcement review



16. Tor IP

Severity: High
Context: Tor network node traffic
Analysis Factors:

Node type (exit, relay, bridge)
Traffic patterns
Associated activities
Policy compliance


Response Actions:

Traffic blocking/monitoring
User investigation
Policy review
Security control assessment



17. Spam IP

Severity: Medium
Context: Known spam source traffic
Spam Categories:

Email spam
Web spam
Comment spam
Marketing abuse


Response Actions:

IP blocking
Content filtering
Rate limiting
Reputation monitoring



18. NTP TOR IP

Severity: Medium to High
Context: NTP requests via Tor
Detection Parameters:

NTP request patterns
Tor exit node verification
Time synchronization analysis
Protocol abuse patterns


Response Actions:

Traffic analysis
NTP configuration review
Time source validation
Network policy review



19. Sentinel Threat Alert

Severity: Medium
Context: An alert triggered by Sentinel indicating potential threat activity.

Threat Categories:

Unauthorized access attempts
Malware detection
Suspicious process executions
Network anomalies

Response Actions:

Device scanning for unwanted software
User and system activity review
Threat intelligence lookup
Quarantine affected systems
Incident response team notification



If none of the templates match well with the alert summary, return exactly "Unknown Template".
    Otherwise, return only the exact name of the best matching template.

Important: Double check your response:
    1. If it matches any template exactly, return just that template name
    2. If no good match exists, return exactly "Unknown Template"
    3. Do not add any explanation or additional text

       Return only the exact name of the best matching template. Return only the template name, nothing else.
       """

       try:
           matching_template = self.llm.predict(template_matching_prompt).strip()
           if matching_template in SITREP_TEMPLATES_DETAILED:
               return matching_template
           else:
               logger.warning(f"LLM returned unrecognized template: {matching_template}")
               return "Unknown Template"
       except Exception as e:
           logger.error(f"Error in template matching: {str(e)}")
           return "Unknown Template"

   def is_general_query(self, query: str) -> bool:
    """Determine if a query is general or specific using LLM"""
    # Extract only the content part of the query
    metadata = self.extract_client_metadata(query)
    query_content = metadata["content"]
    
    query_analysis_prompt = f"""
    Analyze if this query is general or specific to customer logs/systems:
    Query: {query_content}

    Guide:
    - General queries ask about understanding alerts, security concepts, or general procedures
    - Specific queries reference customer data, specific systems, or require log analysis
    
    Examples:
    General: "What does this alert mean?", "Not sure I understand what you are trying to tell me?"
    Specific: "Why did we see this traffic spike yesterday?"

    Return only 'general' or 'specific'.
    """

    try:
        response = self.llm.predict(query_analysis_prompt)
        return response.strip().lower() == "general"
    except Exception as e:
        logger.error(f"Error in query classification: {str(e)}")
        return False

   def is_acknowledgment(self, query: str) -> bool:
        """Determine if query is an acknowledgment rather than a question"""
        ack_prompt = f"""
        Analyze if this message is an acknowledgment/statement or a question:
        Message: {query}
        
        Examples of acknowledgments/statements:
        - "I received the documents"
        - "This traffic is expected"
        - "Thank you for the information"
        - "This is from our normal operations"
        
        Examples of questions:
        - "Why did this happen?"
        - "What does this mean?"
        - "Should I be concerned?"
        
        Return only 'acknowledgment' or 'question'.
        """
        
        try:
            response = self.llm.predict(ack_prompt).strip().lower()
            return response == 'acknowledgment'
        except Exception as e:
            logger.error(f"Error in acknowledgment detection: {str(e)}")
            return False

   def generate_json_path_filter(self, sitrep_data: Dict) -> Optional[Dict]:
       """Generate JSON path filters based on sitrep data"""
       try:
           filter_prompt = f"""
           Create a JSON path filter based on this security alert:
           Template: {sitrep_data.get('template', '')}
           Alert Summary: {sitrep_data.get('alert_summary', '')}
           Customer Query: {sitrep_data.get('feedback', '')}

           Generate a JSON filter that would help process similar alerts.
           Include:
           1. Key paths to monitor
           2. Conditions to match
           3. Thresholds or patterns to detect

           Return only valid JSON without explanation.
           """
           
           filter_response = self.llm.predict(filter_prompt)
           
           try:
               filter_data = json.loads(filter_response)
               filter_data["metadata"] = {
                   "template": sitrep_data.get('template', ''),
                   "generated_for": sitrep_data.get("alert_type", "unknown"),
                   "query_type": "general" if self.is_general_query(sitrep_data.get("feedback", "")) else "specific"
               }
               return filter_data
               
           except json.JSONDecodeError:
               logger.error("Failed to parse JSON filter response")
               return None
               
       except Exception as e:
           logger.error(f"Error generating JSON path filter: {str(e)}")
           return None

   def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
    """Main analysis method with enhanced metadata handling"""
    try:
        template_name = self.find_matching_template(alert_summary)
        template_details = SITREP_TEMPLATES_DETAILED.get(template_name, {})
        
        # Extract metadata and clean content first
        metadata = self.extract_client_metadata(client_query) if client_query else {
            "name": None,
            "timestamp": None,
            "content": ""
        }
        
        # Use cleaned content for general/specific classification
        is_general = True if not client_query else self.is_general_query(metadata["content"])
        
        result = {
            "template": template_name,
            "template_details": template_details,
            "is_general_query": is_general,
            "requires_manual_review": not is_general,
            "template_json": json.dumps(template_details, indent=2)
        }
        
        if client_query:
            json_filter = self.generate_json_path_filter({
                "template": template_name,
                "alert_summary": alert_summary,
                "feedback": metadata["content"]  # Use cleaned content
            })
            if json_filter:
                result["json_filter"] = json_filter
        
        if is_general:
            analysis_result = self.generate_analysis(
                template_name,
                template_details,
                alert_summary,
                client_query,
                is_general
            )
            result["analysis"] = analysis_result.get("analysis")
        
        return result

    except Exception as e:
        logger.error(f"Error in analyze_sitrep: {str(e)}")
        return {"error": str(e)}

    

   def generate_analysis(self, template_name: str, template_details: Dict, 
                         alert_summary: str, client_query: Optional[str], 
                         is_general: bool) -> Dict:
        """Generate analysis based on query type"""
        metadata = self.extract_client_metadata(client_query) if client_query else {
            "name": None,
            "timestamp": None,
            "content": client_query or ""
        }
        
        # Check if it's an acknowledgment
        if self.is_acknowledgment(metadata["content"]):
            greeting = f"Hey {metadata['name']}" if metadata['name'] else "Hey"
            response = f"{greeting}, thank you for letting us know. We've noted your response. - Gradient Cyber Team"
        else:
            # Original analysis logic for questions
            system_prompt = SystemMessagePromptTemplate.from_template(
                """You are a senior security analyst providing clear, accurate and concise responses.
                Rules:
                1. Start with "{greeting}"
                2. State the current security context and its implication
                3. Add one clear recommendation that should tell by "we" not "I"
                4. Use exactly 3-5 sentences maximum
                5. End with "We hope this answers your question. Thank you! Gradient Cyber Team"
                
                Template type: {template}
                Query Type: {query_type}"""
            )
            
            human_template = """
            Alert Summary: {alert_summary}
            Client Query: {query}
            
            Provide a clear, concise explanation of:
            1. What the alert means
            2. Why it matters
            3. What action is recommended (if any)
            
            Follow the exact format described in the system prompt.
            """
            
            greeting = f"Hey {metadata['name']}" if metadata['name'] else "Hey"
            
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_template])
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            
            response = chain.run(
                greeting=greeting,
                template=template_name,
                query_type="General" if is_general else "Specific",
                alert_summary=alert_summary,
                query=metadata["content"]
            )

        return {
            "template": template_name,
            "template_details": template_details,
            "analysis": response.strip(),
            "is_general_query": is_general,
            "requires_manual_review": not is_general,
            "template_json": json.dumps(template_details, indent=2)
        }

def main():
   st.set_page_config(page_title="Sitrep Analyzer", layout="wide")
   
   st.markdown("""
       <style>
       .json-box {
           background-color: #f8f9fa;
           padding: 15px;
           border-radius: 8px;
           border-left: 5px solid #28a745;
           font-family: monospace;
           white-space: pre-wrap;
       }
       .automation-box {
           background-color: #f8f9fa;
           padding: 15px;
           border-radius: 8px;
           border-left: 5px solid #17a2b8;
       }
       .manual-review-box {
           background-color: #fff3cd;
           padding: 15px;
           border-radius: 8px;
           border-left: 5px solid #ffc107;
       }
       </style>
   """, unsafe_allow_html=True)
   
   analyzer = SitrepAnalyzer()
   
   st.title("Sitrep Analysis System")
   
   col1, col2 = st.columns([2, 1])
   
   with col1:
       st.subheader("Alert Summary")
       alert_summary = st.text_area(
           "Paste your security alert details here",
           height=300
       )
       
   with col2:
       st.subheader("Client Query")
       client_query = st.text_area(
           "Enter client questions or feedback",
           height=150
       )
       
       st.subheader("JSON Display Options")
       show_template_json = st.checkbox("Show Template JSON", value=False)
       show_filter_json = st.checkbox("Show Generated Filters", value=True)
   
   if st.button("Analyze Alert", type="primary"):
       if not alert_summary:
           st.error("Please enter an alert summary to analyze.")
           return
       
       with st.spinner("Analyzing security alert..."):
           result = analyzer.analyze_sitrep(alert_summary, client_query)
           
           if "error" in result:
               st.error(result["error"])
           else:
               st.subheader("Template Match")
               st.json({
                   "Template": result["template"],
               })
               
               if show_template_json:
                   st.subheader("Template JSON")
                   st.markdown(f"""
                       <div class="json-box">
                       {result["template_json"]}
                       </div>
                   """, unsafe_allow_html=True)
               
               if show_filter_json and "json_filter" in result:
                   st.subheader("Generated JSON Filter")
                   st.json(result["json_filter"])
               
               if result.get("requires_manual_review"):
                   st.markdown("""
                       <div class="manual-review-box">
                       <h4>👥 Manual Review Required</h4>
                       <p>This query requires specific analysis of customer logs or systems.</p>
                       </div>
                   """, unsafe_allow_html=True)
               else:
                   st.markdown("""
                       <div class="automation-box">
                       <h4>🤖 Automated Processing</h4>
                       <p>This query has been identified as a general inquiry and can be handled automatically.</p>
                       </div>
                   """, unsafe_allow_html=True)
                   
                   if "analysis" in result:
                       st.subheader("Analysis")
                       st.markdown(result["analysis"])
               
               if show_filter_json and "json_filter" in result:
                   st.subheader("Generated JSON Filter")
                   st.json(result["json_filter"])

if __name__ == "__main__":
   main()
