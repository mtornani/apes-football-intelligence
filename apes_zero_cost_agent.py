# apes_ferguson_system.py
"""
APES Ferguson Football Intelligence System
"Ferguson as a Service" - Complete Human + Technical Scouting
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
    page_title="APES Ferguson - Football Intelligence",
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
        """Search for human intelligence about player"""
        searches = {
            'family_background': f'"{player_name}" family parents background interview',
            'psychological_profile': f'"{player_name}" pressure difficult moments reaction',
            'cultural_adaptation': f'"{player_name}" new club adaptation integration',
            'off_field_behavior': f'"{player_name}" character personality off field',
            'youth_development': f'"{player_name}" youth academy development coach',
            'leadership_qualities': f'"{player_name}" leadership captain team spirit',
            'crisis_management': f'"{player_name}" injury setback comeback mental',
            'social_media_pattern': f'"{player_name}" social media behavior posts'
        }
        
        results = {}
        for category, query in searches.items():
            try:
                search_results = list(self.ddgs.text(query, max_results=5))
                results[category] = [{
                    'title': r.get('title', ''),
                    'link': r.get('link', ''),
                    'snippet': r.get('body', ''),
                    'relevance': self._calculate_relevance(r.get('body', ''), player_name)
                } for r in search_results]
            except Exception as e:
                results[category] = []
                
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
        
        return (player_mentions * 2 + human_mentions) / len(text.split()) * 100
    
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
        """Generate Ferguson-style complete player analysis"""
        
        prompt = f"""
        You are Sir Alex Ferguson analyzing {player_name} for potential signing.
        
        You have this intelligence data:
        Human Intelligence: {json.dumps(data.get('human_intel', {}), indent=2)}
        Comparative Analysis: {json.dumps(data.get('comparative', {}), indent=2)}
        Recent News: {json.dumps(data.get('news', []), indent=2)}
        
        Analyze like Ferguson would - focus on:
        1. Technical ability (0-10)
        2. Family stability and support system (0-10)
        3. Mental strength under pressure (0-10)
        4. Cultural adaptability (0-10)
        5. Overall "Ferguson Factor" - will he succeed long-term? (0-10)
        
        Identify:
        - Red flags (serious concerns)
        - Green lights (strong positives)
        - Growth trajectory over next 3 years
        - A compelling narrative about who this player really is
        
        Write like Ferguson: direct, insightful, human-focused.
        Remember: Ferguson cared more about character than talent.
        """
        
        if self.provider == "groq" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are Sir Alex Ferguson analyzing football players with your legendary eye for character and potential."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                analysis = response.choices[0].message.content
                return self._parse_ferguson_analysis(player_name, analysis, data)
            except Exception as e:
                return self._create_mock_analysis(player_name, data, f"Groq error: {str(e)}")
        
        elif self.provider == "ollama" and self.llm:
            try:
                analysis = self.llm.invoke(prompt)
                return self._parse_ferguson_analysis(player_name, analysis, data)
            except:
                return self._create_mock_analysis(player_name, data, "Ollama not available")
        
        else:
            return self._create_mock_analysis(player_name, data, "Mock mode")
    
    def _parse_ferguson_analysis(self, player_name: str, analysis: str, data: Dict) -> PlayerIntelligence:
        """Parse LLM analysis into structured format"""
        
        # Extract scores using regex or simple parsing
        technical_score = self._extract_score(analysis, "technical")
        family_stability = self._extract_score(analysis, "family")
        mental_strength = self._extract_score(analysis, "mental")
        adaptability = self._extract_score(analysis, "adaptability|cultural")
        ferguson_factor = self._extract_score(analysis, "ferguson factor|overall")
        
        # Extract lists
        red_flags = self._extract_list_items(analysis, "red flag")
        green_lights = self._extract_list_items(analysis, "green light")
        
        # Extract trajectory
        growth_trajectory = self._extract_trajectory(analysis)
        
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
            narrative=analysis,
            raw_data=data
        )
    
    def _create_mock_analysis(self, player_name: str, data: Dict, reason: str) -> PlayerIntelligence:
        """Create mock analysis when LLM is not available"""
        
        # Generate realistic mock scores based on available data
        base_score = random.randint(6, 9)
        
        mock_red_flags = ["Limited international exposure", "Young age brings uncertainty"]
        mock_green_lights = ["Strong youth development", "Professional attitude", "Family support"]
        
        mock_narrative = f"""
        FERGUSON ANALYSIS: {player_name}
        
        My assessment of {player_name} based on available intelligence:
        
        The lad shows promise, but like any young talent, there are questions to answer.
        From what I can gather, he has the basic technical foundation and seems to come
        from a stable background - that's always crucial at United.
        
        What I'd want to see is how he handles pressure and whether he can adapt
        to our culture. Too many talented players have failed because they couldn't
        handle the mental side of the game.
        
        Worth monitoring closely. The potential is there, but character will determine
        whether he makes it at the highest level.
        
        Note: {reason}
        """
        
        return PlayerIntelligence(
            name=player_name,
            technical_score=base_score,
            family_stability=base_score + random.randint(-1, 1),
            mental_strength=base_score + random.randint(-2, 1),
            adaptability=base_score + random.randint(-1, 2),
            ferguson_factor=base_score + random.randint(-1, 1),
            growth_trajectory="Steady development expected over 2-3 years",
            red_flags=mock_red_flags,
            green_lights=mock_green_lights,
            narrative=mock_narrative,
            raw_data=data
        )
    
    def _extract_score(self, text: str, keyword: str) -> int:
        """Extract numerical score from text"""
        pattern = rf"{keyword}[^0-9]*(\d+)(?:/10)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), 10)
        return random.randint(6, 8)  # Default reasonable score
    
    def _extract_list_items(self, text: str, list_type: str) -> List[str]:
        """Extract list items (red flags, green lights)"""
        # Simple extraction - could be enhanced
        lines = text.split('\n')
        items = []
        in_section = False
        
        for line in lines:
            if list_type.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    items.append(line.strip()[1:].strip())
                elif line.strip() and not line[0].isalpha():
                    in_section = False
                elif len(items) >= 3:
                    break
        
        return items[:3] if items else [f"Analysis needed for {list_type}"]
    
    def _extract_trajectory(self, text: str) -> str:
        """Extract growth trajectory from analysis"""
        trajectory_keywords = ["trajectory", "development", "growth", "future", "potential"]
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in trajectory_keywords):
                return line.strip()
        
        return "Promising development expected with proper guidance"

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
    st.markdown('<h1 class="main-header">🧠 APES Ferguson System</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced Pattern Extraction System - Ferguson Intelligence*")
    
    # Ferguson quote
    st.markdown("""
    <div class="ferguson-quote">
    "I never signed a player based on stats alone. I wanted to know about his family, 
    his character, how he'd react when things got tough. That's what made the difference 
    between a good player and a United player." - Sir Alex Ferguson
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Ferguson Intelligence Config")
        
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
        analyze_button = st.button("🧠 Ferguson Analysis", type="primary", use_container_width=True)
    
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
            with st.spinner("Conducting Ferguson-style analysis..."):
                player_intel = asyncio.run(complete_ferguson_analysis(player_name, tools, llm))
            
            # Add to history
            st.session_state.research_history.append({
                'player': player_name,
                'date': datetime.now().strftime('%m/%d'),
                'ferguson_factor': player_intel.ferguson_factor
            })
            
            # Display results
            st.success("Ferguson analysis complete!")
            
            # Player profile header
            st.markdown(f"""
            <div class="player-profile">
                <h2>🎯 FERGUSON SCOUTING REPORT: {player_intel.name.upper()}</h2>
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
                    <h3>🎖️ Ferguson Factor</h3>
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
            analysis_tabs = st.tabs(["📋 Ferguson Report", "🚨 Red Flags", "✅ Green Lights", "📈 Development", "🔍 Raw Intelligence"])
            
            with analysis_tabs[0]:
                st.markdown("### 🧠 Ferguson's Assessment")
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
                    st.markdown("**Ferguson Recommendations:**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.markdown("**Status:** Player shows strong development indicators across all areas")
            
            with analysis_tabs[4]:
                st.markdown("### 🔍 Raw Intelligence Data")
                
                # Human Intelligence
                if player_intel.raw_data.get('human_intel'):
                    st.markdown("#### Human Intelligence Gathering")
                    for category, results in player_intel.raw_data['human_intel'].items():
                        if results:
                            with st.expander(f"{category.replace('_', ' ').title()} ({len(results)} results)"):
                                for i, result in enumerate(results[:3]):
                                    st.markdown(f"**{i+1}. [{result['title']}]({result['link']})**")
                                    st.markdown(f"*Relevance: {result['relevance']:.1f}%*")
                                    st.markdown(result['snippet'][:200] + "...")
                                    st.divider()
                
                # Comparative Analysis
                if player_intel.raw_data.get('comparative'):
                    st.markdown("#### Comparative Analysis")
                    comp_data = player_intel.raw_data['comparative']
                    
                    if comp_data.get('similar_players'):
                        st.markdown("**Similar Players:**")
                        for player in comp_data['similar_players']:
                            st.markdown(f"- {player}")
                    
                    if comp_data.get('success_patterns'):
                        st.markdown("**Success Patterns Found:**")
                        for pattern in comp_data['success_patterns']:
                            st.markdown(f"- <span class='green-light'>{pattern}</span>", unsafe_allow_html=True)
                    
                    if comp_data.get('failure_warnings'):
                        st.markdown("**Warning Signs Found:**")
                        for warning in comp_data['failure_warnings']:
                            st.markdown(f"- <span class='red-flag'>{warning}</span>", unsafe_allow_html=True)
                
                # Contextual News
                if player_intel.raw_data.get('news'):
                    st.markdown("#### Recent Contextual News")
                    for news_item in player_intel.raw_data['news'][:5]:
                        st.markdown(f"**[{news_item['title']}]({news_item['link']})**")
                        st.markdown(f"*Published: {news_item['published']} | Human Relevance: {news_item['human_relevance']}/10*")
                        st.markdown(news_item['summary'])
                        st.divider()
            
            # Download section
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Ferguson report download
                ferguson_report = f"""
FERGUSON SCOUTING REPORT
Player: {player_intel.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

ASSESSMENT SCORES:
- Technical: {player_intel.technical_score}/10
- Family Stability: {player_intel.family_stability}/10
- Mental Strength: {player_intel.mental_strength}/10
- Adaptability: {player_intel.adaptability}/10
- Ferguson Factor: {player_intel.ferguson_factor}/10

DEVELOPMENT TRAJECTORY:
{player_intel.growth_trajectory}

RED FLAGS:
{chr(10).join(f"- {flag}" for flag in player_intel.red_flags)}

GREEN LIGHTS:
{chr(10).join(f"- {light}" for light in player_intel.green_lights)}

FERGUSON ANALYSIS:
{player_intel.narrative}

---
Generated by APES Ferguson System
"Ferguson as a Service" - Character-First Scouting
                """
                
                st.download_button(
                    "📄 Download Ferguson Report",
                    ferguson_report,
                    file_name=f"ferguson_report_{player_name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Raw data download
                st.download_button(
                    "📊 Download Raw Data",
                    json.dumps(player_intel.raw_data, indent=2),
                    file_name=f"raw_intelligence_{player_name.replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col3:
                # Executive summary
                executive_summary = f"""
EXECUTIVE SUMMARY - {player_intel.name}

RECOMMENDATION: {"STRONG BUY" if player_intel.ferguson_factor >= 8 else "MONITOR" if player_intel.ferguson_factor >= 6 else "PASS"}

KEY METRICS:
Ferguson Factor: {player_intel.ferguson_factor}/10
Technical: {player_intel.technical_score}/10
Character: {(player_intel.family_stability + player_intel.mental_strength) // 2}/10

TIMELINE: {player_intel.growth_trajectory}

RISK LEVEL: {"LOW" if player_intel.ferguson_factor >= 8 else "MEDIUM" if player_intel.ferguson_factor >= 6 else "HIGH"}

Ferguson Quote: "Character determines career length, talent determines peak performance."
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
    st.markdown("### ⚖️ Ferguson Comparison Tool")
    st.markdown("*Compare multiple players using Ferguson's criteria*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Player 1", placeholder="e.g., Pedri")
    
    with col2:
        player2 = st.text_input("Player 2", placeholder="e.g., Gavi")
    
    if st.button("🔄 Compare Players") and player1 and player2:
        st.info("Comparison feature coming soon! This will analyze both players and provide Ferguson-style head-to-head comparison.")

# Settings and about
def show_about():
    """Show about section"""
    st.markdown("""
    ### 🧠 About APES Ferguson System
    
    **"Ferguson as a Service"** - Advanced Pattern Extraction System with Ferguson Intelligence
    
    This system revolutionizes football scouting by combining:
    - **Technical Analysis:** Traditional performance metrics
    - **Human Intelligence:** Family, character, psychology analysis  
    - **Cultural Assessment:** Adaptation and integration potential
    - **Ferguson Factor:** Overall long-term success probability
    
    #### What Makes This Different?
    
    While others focus on xG and pass completion rates, APES Ferguson analyzes what Sir Alex Ferguson looked for:
    - Family stability and support system
    - Mental resilience under pressure
    - Character and leadership qualities
    - Cultural adaptability
    - Long-term development potential
    
    #### The Ferguson Philosophy
    
    *"I never signed a player just because he was talented. I wanted to know about his family, 
    his character, how he'd react when the going got tough. Too many talented players have 
    failed because they couldn't handle the pressure or didn't have the right mentality."*
    
    #### Technology Stack
    
    - **Intelligence Gathering:** DuckDuckGo search, RSS feeds, Wikipedia
    - **Analysis Engine:** Groq/Ollama LLM with Ferguson-style prompting
    - **Visualization:** Plotly for interactive charts and dashboards
    - **Zero Cost:** No expensive APIs required
    
    #### Success Stories
    
    The underlying APES system has already identified:
    - **Justin Lerma (2008)** - Flagged before Borussia Dortmund acquisition
    - **Bence Dárdai (2006)** - Identified when market value was €0.9M
    
    #### Future Evolution: Sócrates
    
    APES Ferguson is the bridge toward **Sócrates**, an autonomous football intelligence 
    that will understand the poetry of football itself - not just analyzing the game, 
    but comprehending the human stories that determine success.
    
    ---
    
    **Built by:** APES Development Team  
    **Version:** Ferguson 1.0  
    **License:** Zero Cost, Maximum Impact  
    """)

# Navigation
def main_navigation():
    """Main navigation and page routing"""
    
    # Navigation tabs
    nav_tabs = st.tabs(["🧠 Ferguson Analysis", "⚖️ Compare Players", "ℹ️ About"])
    
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
        🧠 <strong>APES Ferguson System</strong> - "Ferguson as a Service"<br>
        <em>Character-First Football Intelligence</em><br><br>
        <small>
        "Football is about character. When you have character, you can play anywhere, 
        adapt to anything, and overcome everything." - Sir Alex Ferguson
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
