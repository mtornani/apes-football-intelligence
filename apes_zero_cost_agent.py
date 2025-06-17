# apes_cognitive_scouting.py
"""
APES Cognitive Scouting Intelligence
Character-First Player Analysis Platform
Zero Cost, Maximum Impact
"""

import asyncio
import streamlit as st
from typing import Dict, Any, List, Tuple
import time
from datetime import datetime, timedelta
import json
import re
import random
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Free alternatives
from duckduckgo_search import DDGS
import feedparser
import requests
from bs4 import BeautifulSoup
import wikipedia
from pytrends.request import TrendReq

# Free LLM options
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

# Streamlit configuration
st.set_page_config(
    page_title="APES Cognitive Scouting Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for football-like styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #00ff88, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .ferguson-quote {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 1rem;
        font-style: italic;
        margin: 1rem 0;
    }
    
    .player-profile {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(0, 204, 255, 0.1);
        border: 1px solid #00ccff;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .red-flag {
        color: #ff4444;
        font-weight: bold;
    }
    
    .green-light {
        color: #00ff88;
        font-weight: bold;
    }
    
    .warning-flag {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "research_history" not in st.session_state:
    st.session_state.research_history = []

@dataclass
class PlayerIntelligence:
    """Complete player intelligence profile"""
    name: str
    technical_score: int
    family_stability: int
    mental_strength: int
    adaptability: int
    ferguson_factor: int
    growth_trajectory: str
    red_flags: List[str]
    green_lights: List[str]
    narrative: str
    raw_data: Dict

class FergusonTools:
    """Advanced tools for human-centric football intelligence"""
    
    def __init__(self):
        self.ddgs = DDGS()
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        except:
            self.pytrends = None
        
    def search_player_human_intel(self, player_name: str) -> Dict[str, List[Dict]]:
        """Search for human intelligence about player with fallback strategies"""
        
        # Extract just first name and last name for broader searches
        name_parts = player_name.split()
        first_name = name_parts[0] if name_parts else player_name
        last_name = name_parts[-1] if len(name_parts) > 1 else ""
        
        searches = {
            'family_background': [
                f'"{player_name}" family parents mother father',
                f'{first_name} {last_name} family background',
                f'{first_name} {last_name} parents interview'
            ],
            'psychological_profile': [
                f'"{player_name}" personality character mentality',
                f'{first_name} {last_name} mental strength pressure',
                f'{first_name} {last_name} difficult moments reaction'
            ],
            'cultural_adaptation': [
                f'"{player_name}" adaptation new club culture',
                f'{first_name} {last_name} integration team',
                f'{first_name} {last_name} cultural adjustment'
            ],
            'off_field_behavior': [
                f'"{player_name}" character personality behavior',
                f'{first_name} {last_name} off field lifestyle',
                f'{first_name} {last_name} professional attitude'
            ],
            'youth_development': [
                f'"{player_name}" youth academy development',
                f'{first_name} {last_name} academy coach opinion',
                f'{first_name} {last_name} youth career progression'
            ],
            'leadership_qualities': [
                f'"{player_name}" leadership captain qualities',
                f'{first_name} {last_name} team leader spirit',
                f'{first_name} {last_name} captain material'
            ],
            'crisis_management': [
                f'"{player_name}" injury comeback resilience',
                f'{first_name} {last_name} setback recovery',
                f'{first_name} {last_name} difficult period overcome'
            ],
            'social_media_pattern': [
                f'"{player_name}" social media presence',
                f'{first_name} {last_name} instagram twitter behavior',
                f'{first_name} {last_name} social media activity'
            ]
        }
        
        results = {}
        for category, query_list in searches.items():
            category_results = []
            
            # Try multiple query variations
            for query in query_list:
                try:
                    search_results = list(self.ddgs.text(query, max_results=3))
                    for r in search_results:
                        if r not in category_results:  # Avoid duplicates
                            category_results.append({
                                'title': r.get('title', ''),
                                'link': r.get('link', ''),
                                'snippet': r.get('body', ''),
                                'relevance': self._calculate_relevance(r.get('body', ''), player_name),
                                'query_used': query
                            })
                    
                    # If we got good results, don't need more queries for this category
                    if len(category_results) >= 3:
                        break
                        
                except Exception as e:
                    continue
            
            # Sort by relevance and take top results
            category_results.sort(key=lambda x: x['relevance'], reverse=True)
            results[category] = category_results[:5]
                
        return results
    
    def search_comparative_analysis(self, player_name: str) -> Dict[str, Any]:
        """Find similar player comparisons and success/failure patterns"""
        try:
            comparison_query = f'"{player_name}" similar player comparison style reminds'
            similar_players = list(self.ddgs.text(comparison_query, max_results=8))
            
            # Extract player names from comparisons
            mentioned_players = self._extract_player_names(similar_players)
            
            return {
                'similar_players': mentioned_players,
                'comparison_articles': similar_players,
                'success_patterns': self._analyze_success_patterns(similar_players),
                'failure_warnings': self._analyze_failure_patterns(similar_players)
            }
        except:
            return {}
    
    def get_contextual_news(self, player_name: str) -> List[Dict]:
        """Get recent news with human context"""
        feeds = [
            f"https://news.google.com/rss/search?q={player_name}+transfer+family",
            f"https://news.google.com/rss/search?q={player_name}+interview+personality",
            f"https://news.google.com/rss/search?q={player_name}+coach+opinion"
        ]
        
        news = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:3]:
                    news.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', '')[:200],
                        'human_relevance': self._assess_human_relevance(entry.title + ' ' + entry.get('summary', ''))
                    })
            except:
                continue
                
        return sorted(news, key=lambda x: x['human_relevance'], reverse=True)
    
    def _calculate_relevance(self, text: str, player_name: str) -> float:
        """Calculate how relevant the text is for human intelligence"""
        human_keywords = ['family', 'parents', 'mother', 'father', 'personality', 'character', 
                         'mental', 'pressure', 'adaptation', 'culture', 'leadership', 'captain',
                         'difficult', 'setback', 'comeback', 'interview', 'coach', 'opinion']
        
        text_lower = text.lower()
        player_mentions = text_lower.count(player_name.lower())
        human_mentions = sum(1 for keyword in human_keywords if keyword in text_lower)
        
        return (player_mentions * 2 + human_mentions) / max(len(text.split()), 1) * 100
    
    def _extract_player_names(self, articles: List[Dict]) -> List[str]:
        """Extract mentioned player names from articles"""
        # Simple extraction - could be enhanced with NER
        common_patterns = [r'like (\w+ \w+)', r'reminds of (\w+ \w+)', r'similar to (\w+ \w+)']
        players = set()
        
        for article in articles:
            text = article.get('body', '') + ' ' + article.get('title', '')
            for pattern in common_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                players.update(matches)
        
        return list(players)[:5]
    
    def _analyze_success_patterns(self, articles: List[Dict]) -> List[str]:
        """Identify success patterns from comparison articles"""
        success_indicators = [
            "strong family support", "mental resilience", "cultural adaptation",
            "leadership qualities", "work ethic", "humble personality",
            "pressure handling", "team spirit", "professional attitude"
        ]
        
        found_patterns = []
        for article in articles:
            text = (article.get('body', '') + ' ' + article.get('title', '')).lower()
            for indicator in success_indicators:
                if indicator in text:
                    found_patterns.append(indicator.title())
        
        return list(set(found_patterns))
    
    def _analyze_failure_patterns(self, articles: List[Dict]) -> List[str]:
        """Identify failure warning signs"""
        warning_signs = [
            "family pressure", "adaptation problems", "cultural shock",
            "ego issues", "discipline problems", "injury prone",
            "inconsistent", "attitude problems", "social media issues"
        ]
        
        found_warnings = []
        for article in articles:
            text = (article.get('body', '') + ' ' + article.get('title', '')).lower()
            for warning in warning_signs:
                if warning in text:
                    found_warnings.append(warning.title())
        
        return list(set(found_warnings))
    
    def _assess_human_relevance(self, text: str) -> float:
        """Assess how relevant news is for human intelligence"""
        human_keywords = ['interview', 'personality', 'family', 'character', 'mental', 
                         'adaptation', 'culture', 'leadership', 'coach opinion']
        
        score = sum(2 for keyword in human_keywords if keyword in text.lower())
        return score

class FergusonLLM:
    """LLM with Ferguson-style analysis"""
    
    def __init__(self, provider: str, api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "groq" and api_key and Groq:
            self.client = Groq(api_key=api_key)
        elif provider == "ollama" and Ollama:
            self.llm = Ollama(model="llama3")
        else:
            self.client = None
    
    def analyze_player_intelligence(self, player_name: str, data: Dict) -> PlayerIntelligence:
        """Generate Ferguson-style complete player analysis with structured output"""
        
        # Create structured prompt for consistent JSON output
        prompt = f"""
        You are Sir Alex Ferguson analyzing {player_name} for potential signing.
        
        Based on this intelligence data, provide a structured analysis:
        
        INTELLIGENCE DATA:
        {json.dumps(data, indent=2, default=str)}
        
        You must respond with a valid JSON object containing exactly these fields:
        {{
            "technical_score": (integer 0-10),
            "family_stability": (integer 0-10), 
            "mental_strength": (integer 0-10),
            "adaptability": (integer 0-10),
            "character_factor": (integer 0-10),
            "growth_trajectory": "(string describing 3-year development path)",
            "red_flags": ["flag1", "flag2", "flag3"],
            "green_lights": ["positive1", "positive2", "positive3"],
            "character_analysis": "(detailed character-focused narrative analysis)"
        }}
        
        SCORING CRITERIA:
        - Technical: Pure football ability and skill level
        - Family Stability: Support system, background, upbringing quality  
        - Mental Strength: Pressure handling, resilience, character
        - Adaptability: Cultural integration, flexibility, learning ability
        - Character Factor: Overall long-term success probability at top level
        
        RED FLAGS: Serious concerns that could derail career
        GREEN LIGHTS: Strong positives indicating success potential
        
        Base your analysis on character-first philosophy: "Character beats talent when talent doesn't have character."
        If data is limited, indicate this in your analysis but still provide reasonable estimates based on available information.
        
        IMPORTANT: Respond ONLY with the JSON object, no additional text.
        """
        
        if self.provider == "groq" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are Sir Alex Ferguson. Respond only with valid JSON as requested. No additional text or explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent output
                    max_tokens=2000
                )
                
                analysis_text = response.choices[0].message.content.strip()
                
                # Try to parse JSON response
                try:
                    # Clean the response - remove any non-JSON content
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = analysis_text[json_start:json_end]
                        analysis_json = json.loads(json_text)
                        return self._create_player_intelligence_from_json(player_name, analysis_json, data)
                    else:
                        raise ValueError("No valid JSON found")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    # Fallback to text parsing if JSON parsing fails
                    st.warning(f"JSON parsing failed, using text analysis fallback: {str(e)}")
                    return self._parse_ferguson_analysis_robust(player_name, analysis_text, data)
                    
            except Exception as e:
                return self._create_honest_analysis(player_name, data, f"Groq error: {str(e)}")
        
        elif self.provider == "ollama" and self.llm:
            try:
                analysis = self.llm.invoke(prompt)
                # Try JSON parsing first, then fallback
                try:
                    json_start = analysis.find('{')
                    json_end = analysis.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = analysis[json_start:json_end]
                        analysis_json = json.loads(json_text)
                        return self._create_player_intelligence_from_json(player_name, analysis_json, data)
                    else:
                        raise ValueError("No valid JSON found")
                except:
                    return self._parse_ferguson_analysis_robust(player_name, analysis, data)
            except:
                return self._create_honest_analysis(player_name, data, "Ollama not available")
        
        else:
            return self._create_honest_analysis(player_name, data, "Mock mode - No LLM")
    
    def _create_player_intelligence_from_json(self, player_name: str, analysis_json: Dict, data: Dict) -> PlayerIntelligence:
        """Create PlayerIntelligence from structured JSON response"""
        
        return PlayerIntelligence(
            name=player_name,
            technical_score=min(max(int(analysis_json.get('technical_score', 7)), 0), 10),
            family_stability=min(max(int(analysis_json.get('family_stability', 7)), 0), 10),
            mental_strength=min(max(int(analysis_json.get('mental_strength', 7)), 0), 10),
            adaptability=min(max(int(analysis_json.get('adaptability', 7)), 0), 10),
            ferguson_factor=min(max(int(analysis_json.get('character_factor', analysis_json.get('ferguson_factor', 7))), 0), 10),
            growth_trajectory=analysis_json.get('growth_trajectory', 'Steady development expected'),
            red_flags=analysis_json.get('red_flags', ['Limited data available'])[:3],
            green_lights=analysis_json.get('green_lights', ['Professional development path'])[:3],
            narrative=analysis_json.get('character_analysis', analysis_json.get('ferguson_analysis', 'Analysis based on available data')),
            raw_data=data
        )
    
    def _parse_ferguson_analysis_robust(self, player_name: str, analysis: str, data: Dict) -> PlayerIntelligence:
        """Robust text parsing with multiple fallback methods"""
        
        # Extract scores with multiple patterns  
        technical_score = self._extract_score_robust(analysis, ["technical", "skill", "ability"])
        family_stability = self._extract_score_robust(analysis, ["family", "background", "support"])
        mental_strength = self._extract_score_robust(analysis, ["mental", "character", "resilience", "pressure"])
        adaptability = self._extract_score_robust(analysis, ["adaptability", "cultural", "integration", "flexibility"])
        ferguson_factor = self._extract_score_robust(analysis, ["character_factor", "ferguson", "overall", "total", "final"])
        
        # Extract lists with robust patterns
        red_flags = self._extract_list_robust(analysis, ["red flag", "concern", "warning", "risk", "negative"])
        green_lights = self._extract_list_robust(analysis, ["green light", "positive", "strength", "asset", "advantage"])
        
        # Extract trajectory
        growth_trajectory = self._extract_trajectory_robust(analysis)
        
        # Clean up narrative - remove JSON artifacts
        clean_narrative = self._clean_narrative(analysis)
        
        return PlayerIntelligence(
            name=player_name,
            technical_score=technical_score,
            family_stability=family_stability,
            mental_strength=mental_strength,
            adaptability=adaptability,
            ferguson_factor=ferguson_factor,
            growth_trajectory=growth_trajectory,
            red_flags=red_flags,
            green_lights=green_lights,
            narrative=clean_narrative,
            raw_data=data
        )
    
    def _clean_narrative(self, text: str) -> str:
        """Clean narrative text from JSON artifacts and formatting issues"""
        
        # Remove JSON structure artifacts
        cleaned = re.sub(r'\{[^}]*\}', '', text)  # Remove JSON objects
        cleaned = re.sub(r'"[^"]*":', '', cleaned)  # Remove JSON keys
        cleaned = re.sub(r'[:,\[\]{}"]', '', cleaned)  # Remove JSON punctuation
        
        # Clean up whitespace and formatting
        cleaned = re.sub(r'\n+', '\n', cleaned)  # Multiple newlines to single
        cleaned = re.sub(r'\s+', ' ', cleaned)    # Multiple spaces to single
        cleaned = cleaned.strip()
        
        # If nothing meaningful left, create a basic narrative
        if len(cleaned) < 50:
            return """
CHARACTER ASSESSMENT: This player shows promising technical abilities with solid foundational development. 
Based on available intelligence, there are both positive indicators and areas requiring attention.

The analysis suggests a player with good potential but one who will need careful guidance and support 
to reach their full capabilities at the highest level.

Further direct scouting recommended to complete the character assessment.
            """.strip()
        
        return cleaned
    
    def _extract_score_robust(self, text: str, keywords: List[str]) -> int:
        """Extract numerical score with multiple fallback patterns"""
        
        # Pattern 1: Direct keyword + number
        for keyword in keywords:
            patterns = [
                rf"{keyword}[^0-9]*(\d+)(?:/10)?",
                rf"{keyword}[^0-9]*:[ ]*(\d+)",
                rf"{keyword}[^0-9]*score[^0-9]*(\d+)",
                rf"(\d+)/10.*{keyword}",
                rf"(\d+).*{keyword}"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    return min(max(score, 0), 10)
        
        # Pattern 2: Look for any score mentions
        score_patterns = [
            r"(\d+)/10",
            r"score.*?(\d+)",
            r"rating.*?(\d+)",
            r"(\d+) out of 10"
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                scores = [int(m) for m in matches if int(m) <= 10]
                if scores:
                    return scores[0]
        
        # Fallback: Conservative baseline
        return 6  # Conservative reasonable score
    
    def _extract_list_robust(self, text: str, keywords: List[str]) -> List[str]:
        """Extract list items with multiple extraction methods"""
        
        items = []
        
        # Method 1: Find sections with keywords
        for keyword in keywords:
            section_pattern = rf"{keyword}[^:]*:([^\\n]+(?:\\n[^\\n]*)*?)(?:\\n\\n|\n\n|$)"
            match = re.search(section_pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if match:
                section_text = match.group(1)
                # Extract bullet points or lines
                lines = section_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        items.append(line[1:].strip())
                    elif line and not line[0].isdigit():
                        items.append(line)
        
        # Method 2: Look for contextual phrases
        if not items:
            for keyword in keywords:
                # Find sentences containing keywords
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        # Extract meaningful phrases
                        phrase = sentence.strip()
                        if len(phrase) > 10 and len(phrase) < 100:
                            items.append(phrase)
        
        # Method 3: Fallback based on data quality
        if not items:
            if "limited" in text.lower() or "insufficient" in text.lower():
                items = ["Limited data available for assessment"]
            else:
                items = ["Requires further investigation"]
        
        return items[:3]  # Return top 3 items
    
    def _extract_trajectory_robust(self, text: str) -> str:
        """Extract growth trajectory with fallback methods"""
        
        trajectory_patterns = [
            r"trajectory[^.]*[.]",
            r"development[^.]*[.]",
            r"growth[^.]*[.]",
            r"potential[^.]*[.]",
            r"future[^.]*[.]"
        ]
        
        for pattern in trajectory_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        # Fallback based on scores
        return "Progressive development expected with proper guidance and opportunity"
    
    def _create_honest_analysis(self, player_name: str, data: Dict, reason: str) -> PlayerIntelligence:
        """Create realistic analysis when LLM unavailable - based on actual data only"""
        
        # Analyze ACTUAL available data only
        human_intel = data.get('human_intel', {})
        news = data.get('news', [])
        comparative = data.get('comparative', {})
        
        # Calculate data quality metrics
        total_intel_items = sum(len(results) for results in human_intel.values())
        high_relevance_items = sum(1 for results in human_intel.values() 
                                 for item in results if item.get('relevance', 0) > 3)
        
        # Conservative scoring based on data availability
        if total_intel_items > 10 and high_relevance_items > 3:
            base_score = 7  # Reasonable baseline when we have good data
        elif total_intel_items > 5:
            base_score = 6  # Lower when limited data
        else:
            base_score = 5  # Very conservative when minimal data
        
        # Extract REAL flags from actual data
        red_flags = []
        green_lights = []
        
        # Look for actual patterns in gathered data
        if comparative.get('failure_warnings'):
            red_flags.extend(comparative['failure_warnings'][:2])
        
        if comparative.get('success_patterns'):
            green_lights.extend(comparative['success_patterns'][:2])
        
        # Data quality flags (honest assessment)
        if total_intel_items < 3:
            red_flags.append("Insufficient public information for assessment")
        
        if high_relevance_items == 0:
            red_flags.append("No high-quality sources found")
        
        # Positive flags only if we actually found evidence
        if any('academy' in str(results).lower() for results in human_intel.values()):
            green_lights.append("Youth development background identified")
        
        # Default to honest assessment if no data
        if not red_flags:
            red_flags = ["Limited data available for comprehensive assessment"]
        
        if not green_lights:
            green_lights = ["Requires direct scouting for proper evaluation"]
        
        # Honest narrative about data limitations
        narrative = f"""
FERGUSON ASSESSMENT: {player_name}

ASSESSMENT STATUS: {reason}

DATA QUALITY REPORT:
- Intelligence items gathered: {total_intel_items}
- High-relevance sources: {high_relevance_items}
- Data coverage: {'Partial' if total_intel_items > 5 else 'Limited'}

PRELIMINARY ASSESSMENT:
Based on available public information, this represents a {self._get_honest_assessment_level(total_intel_items)} assessment.

{self._get_data_based_recommendation(total_intel_items, high_relevance_items)}

FERGUSON'S RULE: "Never sign a player you haven't properly scouted."

RECOMMENDATION: {self._get_honest_recommendation(total_intel_items)}

Note: This analysis is limited by available public data. Direct scouting required for investment decision.
        """
        
        return PlayerIntelligence(
            name=player_name,
            technical_score=base_score,
            family_stability=base_score,
            mental_strength=base_score,
            adaptability=base_score,
            ferguson_factor=max(base_score - 1, 1),  # Conservative Ferguson factor
            growth_trajectory=f"Assessment incomplete - requires direct evaluation",
            red_flags=red_flags[:3],
            green_lights=green_lights[:3],
            narrative=narrative,
            raw_data=data
        )
    
    def _get_honest_assessment_level(self, total_items: int) -> str:
        """Honest assessment of data quality"""
        if total_items >= 15:
            return "preliminary but substantive"
        elif total_items >= 8:
            return "limited but useful"
        elif total_items >= 3:
            return "minimal"
        else:
            return "insufficient for decision-making"
    
    def _get_data_based_recommendation(self, total_items: int, high_rel: int) -> str:
        """Recommendation based on actual data quality"""
        if total_items >= 10 and high_rel >= 3:
            return "Sufficient initial intelligence to warrant continued interest."
        elif total_items >= 5:
            return "Some intelligence gathered but gaps remain in character assessment."
        else:
            return "Insufficient intelligence for preliminary assessment."
    
    def _get_honest_recommendation(self, total_items: int) -> str:
        """Honest next steps based on data availability"""
        if total_items >= 10:
            return "Proceed to live scouting phase"
        elif total_items >= 5:
            return "Gather additional intelligence before field assessment"
        else:
            return "Insufficient data - focus resources on better-documented targets"

def create_player_dashboard(player_intel: PlayerIntelligence):
    """Create visual dashboard for player intelligence"""
    
    # Radar chart for core metrics
    categories = ['Technical', 'Family Stability', 'Mental Strength', 'Adaptability']
    values = [player_intel.technical_score, player_intel.family_stability, 
              player_intel.mental_strength, player_intel.adaptability]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Player Profile',
        line_color='#00ff88'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title="Ferguson Assessment Radar",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Ferguson Factor gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = player_intel.ferguson_factor,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Ferguson Factor"},
        delta = {'reference': 7},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#00ff88"},
            'steps': [
                {'range': [0, 5], 'color': "#ff4444"},
                {'range': [5, 7], 'color': "#ffaa00"},
                {'range': [7, 10], 'color': "#00ff88"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 8}}
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig_radar, fig_gauge

def create_timeline_analysis(player_intel: PlayerIntelligence):
    """Create development timeline"""
    
    months = ['Current', '6 months', '1 year', '2 years', '3 years']
    
    # Simulate development trajectory
    base_score = player_intel.ferguson_factor
    development = [base_score, base_score + 0.5, base_score + 1.2, base_score + 2.0, base_score + 2.8]
    development = [min(score, 10) for score in development]  # Cap at 10
    
    fig = go.Figure(data=go.Scatter(
        x=months,
        y=development,
        mode='lines+markers',
        line=dict(color='#00ccff', width=3),
        marker=dict(size=8, color='#00ff88'),
        name='Projected Development'
    ))
    
    fig.update_layout(
        title='Ferguson Development Projection',
        xaxis_title='Timeline',
        yaxis_title='Overall Rating',
        yaxis=dict(range=[0, 10]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )
    
    return fig

async def complete_ferguson_analysis(player_name: str, tools: FergusonTools, llm: FergusonLLM) -> PlayerIntelligence:
    """Complete Ferguson-style player analysis"""
    
    progress = st.progress(0)
    status = st.empty()
    
    # Gather intelligence
    status.text("🔍 Gathering human intelligence...")
    human_intel = tools.search_player_human_intel(player_name)
    progress.progress(25)
    
    status.text("📊 Analyzing comparative data...")
    comparative = tools.search_comparative_analysis(player_name)
    progress.progress(50)
    
    status.text("📰 Collecting contextual news...")
    news = tools.get_contextual_news(player_name)
    progress.progress(75)
    
    status.text("🧠 Generating Ferguson analysis...")
    data = {
        'human_intel': human_intel,
        'comparative': comparative,
        'news': news,
        'timestamp': datetime.now().isoformat()
    }
    
    player_intel = llm.analyze_player_intelligence(player_name, data)
    progress.progress(100)
    
    time.sleep(1)
    progress.empty()
    status.empty()
    
    return player_intel

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 APES Cognitive Scouting</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced Pattern Extraction System - Character-First Intelligence*")
    
    # Ferguson quote (manteniamo la filosofia ma rendiamo neutrale)
    st.markdown("""
    <div class="ferguson-quote">
    "The best scouts don't just analyze stats. They understand the person behind the player - 
    the family, the character, the resilience when things get tough. That's what separates 
    good talents from true champions." - Football Intelligence Philosophy
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Cognitive Intelligence Config")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "Intelligence Provider",
            ["Groq (Free)", "Ollama (Local)", "Mock Analysis"]
        )
        
        if llm_provider == "Groq (Free)":
            groq_api_key = st.text_input(
                "Groq API Key",
                value=st.session_state.groq_api_key,
                type="password",
                help="Get free API key at https://console.groq.com"
            )
            if groq_api_key:
                st.session_state.groq_api_key = groq_api_key
                st.session_state.llm_provider = "groq"
        elif llm_provider == "Ollama (Local)":
            st.info("Ensure Ollama is running: `ollama run llama3`")
            st.session_state.llm_provider = "ollama"
        else:
            st.session_state.llm_provider = "mock"
        
        st.divider()
        
        # Quick player searches
        st.markdown("### 🎯 Quick Analysis")
        quick_players = [
            "Pedri Barcelona",
            "Jude Bellingham",
            "Jamal Musiala Bayern",
            "Gavi Barcelona",
            "Eduardo Camavinga"
        ]
        
        for player in quick_players:
            if st.button(player, key=f"quick_{player}"):
                st.session_state.player_name = player
        
        st.divider()
        
        # Research history
        if st.session_state.research_history:
            st.markdown("### 📚 Recent Analysis")
            for i, research in enumerate(st.session_state.research_history[-3:]):
                if st.button(f"{research['player']} ({research['date']})", key=f"history_{i}"):
                    st.session_state.player_name = research['player']
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        player_name = st.text_input(
            "Enter Player Name:",
            value=st.session_state.get('player_name', ''),
            placeholder="e.g., Pedri Barcelona, Jude Bellingham, Jamal Musiala"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        analyze_button = st.button("🧠 Character Analysis", type="primary", use_container_width=True)
    
    if analyze_button:
        if not player_name:
            st.warning("Please enter a player name")
        else:
            # Initialize tools and LLM
            tools = FergusonTools()
            llm = FergusonLLM(
                st.session_state.llm_provider, 
                st.session_state.groq_api_key if st.session_state.llm_provider == "groq" else None
            )
            
            # Check LLM availability
            if st.session_state.llm_provider == "groq" and not st.session_state.groq_api_key:
                st.error("Please enter Groq API key in sidebar")
                st.stop()
            
            # Perform analysis
            with st.spinner("Conducting character-first analysis..."):
                player_intel = asyncio.run(complete_ferguson_analysis(player_name, tools, llm))
            
            # Add to history
            st.session_state.research_history.append({
                'player': player_name,
                'date': datetime.now().strftime('%m/%d'),
                'ferguson_factor': player_intel.ferguson_factor
            })
            
            # Display results
            st.success("Character analysis complete!")
            
            # Player profile header
            st.markdown(f"""
            <div class="player-profile">
                <h2>🎯 COGNITIVE SCOUTING REPORT: {player_intel.name.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚽ Technical</h3>
                    <h2>{player_intel.technical_score}/10</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>👨‍👩‍👧‍👦 Family</h3>
                    <h2>{player_intel.family_stability}/10</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🧠 Mental</h3>
                    <h2>{player_intel.mental_strength}/10</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌍 Adaptability</h3>
                    <h2>{player_intel.adaptability}/10</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[4]:
                color_class = "green-light" if player_intel.ferguson_factor >= 8 else "warning-flag" if player_intel.ferguson_factor >= 6 else "red-flag"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎖️ Character Factor</h3>
                    <h2 class="{color_class}">{player_intel.ferguson_factor}/10</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            chart_cols = st.columns(2)
            
            with chart_cols[0]:
                radar_fig, _ = create_player_dashboard(player_intel)
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with chart_cols[1]:
                _, gauge_fig = create_player_dashboard(player_intel)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Timeline
            timeline_fig = create_timeline_analysis(player_intel)
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Analysis sections
            analysis_tabs = st.tabs(["📋 Character Report", "🚨 Red Flags", "✅ Green Lights", "📈 Development", "🔍 Raw Intelligence"])
            
            with analysis_tabs[0]:
                st.markdown("### 🧠 Character Assessment")
                st.markdown(player_intel.narrative)
            
            with analysis_tabs[1]:
                st.markdown("### 🚨 Areas of Concern")
                for flag in player_intel.red_flags:
                    st.markdown(f"- <span class='red-flag'>{flag}</span>", unsafe_allow_html=True)
            
            with analysis_tabs[2]:
                st.markdown("### ✅ Positive Indicators")
                for light in player_intel.green_lights:
                    st.markdown(f"- <span class='green-light'>{light}</span>", unsafe_allow_html=True)
            
            with analysis_tabs[3]:
                st.markdown("### 📈 Development Trajectory")
                st.markdown(f"**Growth Path:** {player_intel.growth_trajectory}")
                
                # Create development recommendations
                recommendations = []
                if player_intel.technical_score < 7:
                    recommendations.append("Focus on technical skill development")
                if player_intel.family_stability < 7:
                    recommendations.append("Monitor family dynamics and provide support")
                if player_intel.mental_strength < 7:
                    recommendations.append("Implement mental conditioning program")
                if player_intel.adaptability < 7:
                    recommendations.append("Cultural integration support needed")
                
                if recommendations:
                    st.markdown("**Development Recommendations:**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.markdown("**Status:** Player shows strong development indicators across all areas")
            
            with analysis_tabs[4]:
                st.markdown("### 🔍 Raw Intelligence Data")
                
                # Data quality indicator
                total_intel = sum(len(results) for results in player_intel.raw_data.get('human_intel', {}).values())
                high_rel = sum(1 for results in player_intel.raw_data.get('human_intel', {}).values() 
                             for item in results if item.get('relevance', 0) > 3)
                
                # Data quality badge
                if total_intel >= 10 and high_rel >= 3:
                    quality_badge = "🟢 Good Data Coverage"
                    quality_color = "green-light"
                elif total_intel >= 5:
                    quality_badge = "🟡 Partial Data Coverage"
                    quality_color = "warning-flag"
                else:
                    quality_badge = "🔴 Limited Data Coverage"
                    quality_color = "red-flag"
                
                st.markdown(f"**Data Quality:** <span class='{quality_color}'>{quality_badge}</span>", unsafe_allow_html=True)
                st.markdown(f"*Total intelligence items: {total_intel} | High relevance: {high_rel}*")
                st.divider()
                
                # Human Intelligence
                if player_intel.raw_data.get('human_intel'):
                    st.markdown("#### Human Intelligence Gathering")
                    for category, results in player_intel.raw_data['human_intel'].items():
                        if results:
                            with st.expander(f"{category.replace('_', ' ').title()} ({len(results)} results)"):
                                for i, result in enumerate(results[:3]):
                                    st.markdown(f"**{i+1}. [{result['title']}]({result['link']})**")
                                    st.markdown(f"*Relevance: {result['relevance']:.1f}% | Query: {result.get('query_used', 'N/A')}*")
                                    st.markdown(result['snippet'][:200] + "...")
                                    st.divider()
                        else:
                            st.markdown(f"**{category.replace('_', ' ').title()}:** No relevant data found")
                
                # Comparative Analysis
                if player_intel.raw_data.get('comparative'):
                    st.markdown("#### Comparative Analysis")
                    comp_data = player_intel.raw_data['comparative']
                    
                    if comp_data.get('similar_players'):
                        st.markdown("**Similar Players Identified:**")
                        for player in comp_data['similar_players']:
                            st.markdown(f"- {player}")
                    else:
                        st.markdown("**Similar Players:** None identified")
                    
                    if comp_data.get('success_patterns'):
                        st.markdown("**Success Patterns Found:**")
                        for pattern in comp_data['success_patterns']:
                            st.markdown(f"- <span class='green-light'>{pattern}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Success Patterns:** None identified")
                    
                    if comp_data.get('failure_warnings'):
                        st.markdown("**Warning Signs Found:**")
                        for warning in comp_data['failure_warnings']:
                            st.markdown(f"- <span class='red-flag'>{warning}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Warning Signs:** None identified")
                else:
                    st.markdown("#### Comparative Analysis")
                    st.markdown("*No comparative data available*")
                
                # Contextual News
                if player_intel.raw_data.get('news'):
                    st.markdown("#### Recent Contextual News")
                    news_items = [item for item in player_intel.raw_data['news'] if item.get('title')]
                    if news_items:
                        for news_item in news_items[:5]:
                            st.markdown(f"**[{news_item['title']}]({news_item['link']})**")
                            st.markdown(f"*Published: {news_item.get('published', 'Unknown')} | Human Relevance: {news_item.get('human_relevance', 0)}/10*")
                            st.markdown(news_item.get('summary', 'No summary available'))
                            st.divider()
                    else:
                        st.markdown("*No relevant news items found*")
                else:
                    st.markdown("#### Recent Contextual News")
                    st.markdown("*No news data available*")
                
                # Search effectiveness analysis
                st.markdown("#### Search Analysis")
                st.markdown("**Search Strategy Effectiveness:**")
                
                effective_categories = [cat for cat, results in player_intel.raw_data.get('human_intel', {}).items() if results]
                failed_categories = [cat for cat, results in player_intel.raw_data.get('human_intel', {}).items() if not results]
                
                if effective_categories:
                    st.markdown("✅ **Successful searches:**")
                    for cat in effective_categories:
                        st.markdown(f"- {cat.replace('_', ' ').title()}")
                
                if failed_categories:
                    st.markdown("❌ **Failed searches:**")
                    for cat in failed_categories:
                        st.markdown(f"- {cat.replace('_', ' ').title()}")
                
                # Recommendations for data improvement
                if total_intel < 5:
                    st.warning("⚠️ **Data Quality Warning:** Limited intelligence gathered. Consider:")
                    st.markdown("- More specific player name variations")
                    st.markdown("- Alternative search terms")
                    st.markdown("- Direct club/academy sources")
                    st.markdown("- Social media analysis")
                elif high_rel == 0:
                    st.warning("⚠️ **Relevance Warning:** No high-quality sources found. May indicate:")
                    st.markdown("- Emerging talent with limited coverage")
                    st.markdown("- Name spelling variations needed")
                    st.markdown("- Private/academy player requiring direct contact")
            
            # Download section
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Character report download
                character_report = f"""
COGNITIVE SCOUTING REPORT
Player: {player_intel.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

ASSESSMENT SCORES:
- Technical: {player_intel.technical_score}/10
- Family Stability: {player_intel.family_stability}/10
- Mental Strength: {player_intel.mental_strength}/10
- Adaptability: {player_intel.adaptability}/10
- Character Factor: {player_intel.ferguson_factor}/10

DEVELOPMENT TRAJECTORY:
{player_intel.growth_trajectory}

RED FLAGS:
{chr(10).join(f"- {flag}" for flag in player_intel.red_flags)}

GREEN LIGHTS:
{chr(10).join(f"- {light}" for light in player_intel.green_lights)}

CHARACTER ANALYSIS:
{player_intel.narrative}

---
Generated by APES Cognitive Scouting Intelligence
Character-First Football Intelligence
                """
                
                st.download_button(
                    "📄 Download Character Report",
                    character_report,
                    file_name=f"character_report_{player_name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Raw data download
                st.download_button(
                    "📊 Download Raw Data",
                    json.dumps(player_intel.raw_data, indent=2, default=str),
                    file_name=f"raw_intelligence_{player_name.replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col3:
                # Executive summary
                executive_summary = f"""
EXECUTIVE SUMMARY - {player_intel.name}

RECOMMENDATION: {"STRONG BUY" if player_intel.ferguson_factor >= 8 else "MONITOR" if player_intel.ferguson_factor >= 6 else "PASS"}

KEY METRICS:
Character Factor: {player_intel.ferguson_factor}/10
Technical: {player_intel.technical_score}/10
Character: {(player_intel.family_stability + player_intel.mental_strength) // 2}/10

TIMELINE: {player_intel.growth_trajectory}

RISK LEVEL: {"LOW" if player_intel.ferguson_factor >= 8 else "MEDIUM" if player_intel.ferguson_factor >= 6 else "HIGH"}

Intelligence Quote: "Character determines career length, talent determines peak performance."
                """
                
                st.download_button(
                    "📋 Executive Summary",
                    executive_summary,
                    file_name=f"executive_{player_name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

# Comparison tool
def show_comparison_tool():
    """Show player comparison interface"""
    st.markdown("### ⚖️ Character Comparison Tool")
    st.markdown("*Compare multiple players using character-first criteria*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Player 1", placeholder="e.g., Pedri")
    
    with col2:
        player2 = st.text_input("Player 2", placeholder="e.g., Gavi")
    
    if st.button("🔄 Compare Players") and player1 and player2:
        st.info("Comparison feature coming soon! This will analyze both players and provide character-first head-to-head comparison.")

# Settings and about
def show_about():
    """Show about section"""
    st.markdown("""
    ### 🧠 About APES Cognitive Scouting Intelligence
    
    **Character-First Football Intelligence Platform**
    
    This system revolutionizes football scouting by combining:
    - **Technical Analysis:** Traditional performance metrics
    - **Human Intelligence:** Family, character, psychology analysis  
    - **Cultural Assessment:** Adaptation and integration potential
    - **Character Factor:** Overall long-term success probability
    
    #### What Makes This Different?
    
    While others focus on xG and pass completion rates, APES Cognitive analyzes what truly matters:
    - Family stability and support system
    - Mental resilience under pressure
    - Character and leadership qualities
    - Cultural adaptability
    - Long-term development potential
    
    #### The Character-First Philosophy
    
    *"The best scouts don't just analyze stats. They understand the person behind the player - 
    the family, the character, the resilience when things get tough. That's what separates 
    good talents from true champions."*
    
    #### Technology Stack
    
    - **Intelligence Gathering:** DuckDuckGo search, RSS feeds, Wikipedia
    - **Analysis Engine:** Groq/Ollama LLM with character-focused prompting
    - **Visualization:** Plotly for interactive charts and dashboards
    - **Zero Cost:** No expensive APIs required
    
    #### Success Stories
    
    The underlying APES system has already identified:
    - **Justin Lerma (2008)** - Flagged soon after Borussia Dortmund acquisition, with no knowledge about that
    - **Bence Dárdai (2006)** - Identified when market value was €9M
    
    #### Future Evolution: Sócrates
    
    APES Cognitive is the bridge toward **Sócrates**, an autonomous football intelligence 
    that will understand the poetry of football itself - not just analyzing the game, 
    but comprehending the human stories that determine success.
    
    ---
    
    **Built by:** APES Development Team  
    **Version:** Cognitive 1.0  
    **License:** Zero Cost, Maximum Impact  
    """)

# Navigation
def main_navigation():
    """Main navigation and page routing"""
    
    # Navigation tabs
    nav_tabs = st.tabs(["🧠 Character Analysis", "⚖️ Compare Players", "ℹ️ About"])
    
    with nav_tabs[0]:
        main()  # Main analysis tool
    
    with nav_tabs[1]:
        show_comparison_tool()
    
    with nav_tabs[2]:
        show_about()

if __name__ == "__main__":
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        🧠 <strong>APES Cognitive Scouting Intelligence</strong> - Character-First Football Intelligence<br>
        <em>Beyond Statistics - Understanding the Human Factor</em><br><br>
        <small>
        "Football is about character. When you have character, you can play anywhere, 
        adapt to anything, and overcome everything." - Football Intelligence Philosophy
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if running as main app or in navigation
    try:
        # If we're in a navigation context, show full navigation
        main_navigation()
    except:
        # If standalone, just run main analysis
        main()
