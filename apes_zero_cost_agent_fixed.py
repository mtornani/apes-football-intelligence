# apes_cognitive_scouting_enhanced.py
"""
APES Cognitive Scouting Intelligence - Enhanced with Google CSE
Character-First Player Analysis Platform with Better Search
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
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin, urlparse

# Free alternatives - manteniamo DuckDuckGo come fallback
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

import feedparser
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
if "google_cse_api_key" not in st.session_state:
    st.session_state.google_cse_api_key = st.secrets.get("GOOGLE_CSE_API_KEY", "")
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

class EnhancedSearchEngine:
    """Enhanced search engine integrating Google CSE from app.py"""
    
    def __init__(self):
        # Google CSE configuration
        self.google_api_key = st.session_state.get("google_cse_api_key", "")
        self.cse_id = "c12f53951c8884cfd"  # Same as app.py
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize DuckDuckGo as fallback
        self.ddgs = DDGS() if DDGS else None
        
        # Try pytrends
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        except:
            self.pytrends = None
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Primary search method - tries Google CSE first, falls back to DuckDuckGo"""
        
        # Try Google CSE first if API key available
        if self.google_api_key:
            results = self._google_cse_search(query, max_results)
            if results:
                return results
        
        # Fallback to DuckDuckGo
        if self.ddgs:
            try:
                results = list(self.ddgs.text(query, max_results=max_results))
                return [self._format_ddg_result(r) for r in results]
            except:
                pass
        
        return []
    
    def _google_cse_search(self, query: str, max_results: int) -> List[Dict]:
        """Google Custom Search Engine search (from app.py)"""
        
        if not self.google_api_key:
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'key': self.google_api_key,
                'cx': self.cse_id,
                'q': query,
                'num': min(max_results, 10),
                'fields': 'items(title,snippet,link,pagemap)'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_google_results(data)
            elif response.status_code == 429:
                st.warning("⚠️ Google search quota exceeded. Switching to DuckDuckGo...")
                return []
            else:
                return []
                
        except Exception as e:
            return []
    
    def _parse_google_results(self, data: dict) -> List[Dict]:
        """Parse Google CSE response"""
        
        if 'items' not in data:
            return []
        
        results = []
        
        for item in data['items']:
            result = {
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'body': item.get('snippet', ''),  # For compatibility
                'source': self._get_source_name(item.get('link', ''))
            }
            results.append(result)
        
        return results
    
    def _format_ddg_result(self, result: dict) -> dict:
        """Format DuckDuckGo result to match our structure"""
        return {
            'title': result.get('title', ''),
            'link': result.get('link', ''),
            'snippet': result.get('body', ''),
            'body': result.get('body', ''),
            'source': self._get_source_name(result.get('link', ''))
        }
    
    def _get_source_name(self, url: str) -> str:
        """Get readable source name from URL"""
        
        if not url:
            return 'Unknown'
            
        domain_map = {
            'transfermarkt': 'Transfermarkt',
            'whoscored': 'WhoScored',
            'espn': 'ESPN',
            'goal.com': 'Goal.com',
            'tuttomercatoweb': 'TuttoMercatoWeb',
            'calciomercato': 'Calciomercato',
            'fbref': 'FBref',
            'sofascore': 'SofaScore'
        }
        
        url_lower = url.lower()
        for key, name in domain_map.items():
            if key in url_lower:
                return name
        
        try:
            domain = urlparse(url).netloc
            return domain.replace('www.', '').split('.')[0].title()
        except:
            return 'Web Source'
    
    def scrape_page_content(self, url: str) -> Dict:
        """Enhanced scraping with patterns from app.py"""
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            page_text = soup.get_text()
            
            # Extract structured data using patterns from app.py
            extracted_data = self._extract_player_data(page_text)
            
            return {
                'scraped_data': extracted_data,
                'scraping_success': True,
                'page_text': page_text[:1000]  # First 1000 chars
            }
            
        except Exception as e:
            return {
                'scraping_success': False,
                'scraping_error': str(e)
            }
    
    def _extract_player_data(self, text: str) -> Dict:
        """Extract player data using patterns from app.py"""
        
        data = {}
        
        # Age patterns
        age_patterns = [
            r'Age:\s*(\d{1,2})', 
            r'(\d{1,2})\s*years old', 
            r'Born:.*\((\d{1,2})\)',
            r'(\d{1,2})\s*(?:years old|age|anni)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['age'] = int(match.group(1))
                break
        
        # Position patterns
        position_patterns = [
            r'Position:\s*([^,\n]+)', 
            r'Main position:\s*([^,\n]+)',
            r'Plays as:\s*([^,\n]+)'
        ]
        
        for pattern in position_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['position'] = match.group(1).strip()
                break
        
        # Goals and assists
        goals_match = re.search(r'Goals:\s*(\d+)|(\d+)\s*goals?', text, re.IGNORECASE)
        if goals_match:
            data['goals'] = int(goals_match.group(1) or goals_match.group(2))
        
        assists_match = re.search(r'Assists:\s*(\d+)|(\d+)\s*assists?', text, re.IGNORECASE)
        if assists_match:
            data['assists'] = int(assists_match.group(1) or assists_match.group(2))
        
        return data

class FergusonTools:
    """Advanced tools for human-centric football intelligence with Google CSE"""
    
    def __init__(self):
        self.search_engine = EnhancedSearchEngine()
    
    def search_player_human_intel(self, player_name: str) -> Dict[str, List[Dict]]:
        """Search for human intelligence about player using enhanced search"""
        
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
                    search_results = self.search_engine.search(query, max_results=3)
                    
                    for r in search_results:
                        if r not in category_results:  # Avoid duplicates
                            # Enhanced scraping for better data
                            if self.search_engine.google_api_key and r.get('link'):
                                scraped = self.search_engine.scrape_page_content(r['link'])
                                if scraped.get('scraped_data'):
                                    r['scraped_data'] = scraped['scraped_data']
                            
                            r['relevance'] = self._calculate_relevance(r.get('body', ''), player_name)
                            r['query_used'] = query
                            category_results.append(r)
                    
                    # If we got good results, don't need more queries for this category
                    if len(category_results) >= 3:
                        break
                    
                    time.sleep(0.3)  # Rate limiting
                        
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
            similar_players = self.search_engine.search(comparison_query, max_results=8)
            
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
        # Use Google CSE for news searches if available
        if self.search_engine.google_api_key:
            news_queries = [
                f"{player_name} transfer family news 2024",
                f"{player_name} interview personality recent",
                f"{player_name} coach opinion latest"
            ]
            
            news = []
            for query in news_queries:
                try:
                    results = self.search_engine.search(query, max_results=3)
                    for result in results:
                        news.append({
                            'title': result.get('title', ''),
                            'link': result.get('link', ''),
                            'published': 'Recent',
                            'summary': result.get('snippet', '')[:200],
                            'human_relevance': self._assess_human_relevance(
                                result.get('title', '') + ' ' + result.get('snippet', '')
                            )
                        })
                except:
                    continue
            
            return sorted(news, key=lambda x: x['human_relevance'], reverse=True)[:5]
        
        # Fallback to RSS feeds
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
    
    def discover_unknown_talents(self, age_range: str, league_region: str, position: str, additional_terms: str = "") -> List[Dict]:
        """Discover unknown talents using enhanced search"""
        
        # Build discovery search queries
        age_queries = {
            "15-17 (Academy)": ["wonderkid 2007 2008", "young talent 16 17", "academy prospect"],
            "18-20 (Breakthrough)": ["breakthrough 2004 2005", "young player 18 19", "debut 2024"],
            "21-23 (Emerging)": ["emerging talent 2001 2002", "rising star 21 22", "transfer 2024"],
            "Any Age": ["wonderkid", "young talent", "breakthrough"]
        }
        
        region_queries = {
            "South America": ["Brazil wonderkid", "Argentina talent", "Colombian player"],
            "Eastern Europe": ["Polish talent", "Czech player", "Serbian wonderkid"],
            "Africa": ["Nigerian wonderkid", "Senegalese talent", "Ghanaian player"],
            "Asia": ["Japanese talent", "Korean player", "Asian wonderkid"],
            "Lower European Leagues": ["League Two talent", "lower division wonderkid"],
            "Any Region": ["football wonderkid", "soccer talent", "young prospect"]
        }
        
        position_queries = {
            "Midfielder": ["midfielder wonderkid", "midfield talent"],
            "Forward": ["striker wonderkid", "forward talent"],
            "Defender": ["defender wonderkid", "centre back talent"],
            "Goalkeeper": ["goalkeeper wonderkid", "keeper talent"],
            "Any Position": ["football wonderkid", "soccer talent"]
        }
        
        # Combine search terms
        discovery_queries = []
        
        age_terms = age_queries.get(age_range, ["young talent"])
        region_terms = region_queries.get(league_region, ["wonderkid"])
        pos_terms = position_queries.get(position, ["prospect"])
        
        # Create focused queries
        for age_term in age_terms:
            for region_term in region_terms:
                query = f"{age_term} {region_term}"
                if additional_terms:
                    query += f" {additional_terms}"
                discovery_queries.append(query)
        
        # Execute discovery searches
        discovered_players = []
        
        for query in discovery_queries[:8]:  # Limit queries
            try:
                search_results = self.search_engine.search(query, max_results=6)
                
                for result in search_results:
                    # Enhanced scraping if using Google CSE
                    if self.search_engine.google_api_key and result.get('link'):
                        scraped = self.search_engine.scrape_page_content(result['link'])
                        if scraped.get('scraped_data'):
                            result['scraped_data'] = scraped['scraped_data']
                    
                    # Extract potential player names
                    potential_players = self._extract_player_names_from_discovery(result)
                    for player_info in potential_players:
                        if player_info and isinstance(player_info, dict) and player_info.get('name'):
                            existing_names = [p.get('name', '') for p in discovered_players]
                            if player_info['name'] not in existing_names:
                                relevance = self._calculate_discovery_relevance(result, age_range, position)
                                
                                if relevance > 0.5:
                                    player_entry = {
                                        **player_info,
                                        'discovery_query': query,
                                        'source_relevance': relevance
                                    }
                                    discovered_players.append(player_entry)
                
                time.sleep(0.3)  # Rate limiting
                    
            except Exception as e:
                continue
        
        # Sort by relevance
        discovered_players.sort(key=lambda x: x.get('source_relevance', 0), reverse=True)
        return discovered_players[:10]
    
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
    
    def _extract_player_names_from_discovery(self, article: Dict) -> List[Dict]:
        """Extract player names from discovery search results"""
        if not article or not isinstance(article, dict):
            return []
            
        text = str(article.get('body', '')) + ' ' + str(article.get('title', ''))
        
        # Enhanced patterns for player extraction
        patterns = [
            r'(\w+ \w+)\s*\((\d{1,2})\)[^.]*(?:€[\d.]+[MK]|market value|valuation)',
            r'(\d+)\.\s+(\w+ \w+)[^.]*(?:years old|age \d+|\(\d{4}\))',
            r'(\w+ \w+)[^.]*(?:rated|prospect|potential|breakthrough|debut)',
            r'(\w+ \w+),?\s*(?:aged?\s*)?(\d{1,2})[^.]*(?:signed|transfer|move)',
            r'Profile:\s*(\w+ \w+)',
            r'(\w+ \w+)\s*-\s*(?:midfielder|forward|striker|defender|goalkeeper)',
        ]
        
        found_players = []
        try:
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        name = str(match[1] if match[0].isdigit() else match[0])
                        age_info = str(match[0] if match[0].isdigit() else (match[1] if len(match) > 1 else ""))
                    else:
                        name = str(match)
                        age_info = ""
                    
                    # Quality filtering
                    if (len(name.split()) == 2 and 
                        not any(skip in name.lower() for skip in ['the', 'and', 'or', 'but']) and
                        len(name) > 4 and
                        name[0].isupper()):
                        
                        # Check if scraped data has additional info
                        scraped_data = article.get('scraped_data', {})
                        
                        player_info = {
                            'name': name.title(),
                            'age_info': age_info or scraped_data.get('age', ''),
                            'source_title': str(article.get('title', '')),
                            'source_link': str(article.get('link', '')),
                            'context': self._extract_context(text, name),
                            'position': scraped_data.get('position', ''),
                            'goals': scraped_data.get('goals', ''),
                            'assists': scraped_data.get('assists', '')
                        }
                        
                        if player_info['name'] not in [p['name'] for p in found_players]:
                            found_players.append(player_info)
        except Exception:
            return []
        
        return found_players[:5]
    
    def _extract_context(self, text: str, player_name: str) -> str:
        """Extract relevant context around player name"""
        try:
            text = str(text) if text is not None else ""
            player_name = str(player_name) if player_name is not None else ""
            
            if not text or not player_name:
                return ""
            
            sentences = re.split(r'[.!?]', text)
            relevant_sentences = [s.strip() for s in sentences if player_name.lower() in s.lower()]
            
            if relevant_sentences:
                return relevant_sentences[0][:150] + "..."
            return ""
        except Exception:
            return ""
    
    def _calculate_discovery_relevance(self, article: Dict, age_range: str, position: str) -> float:
        """Calculate relevance for talent discovery"""
        text = (article.get('body', '') + ' ' + article.get('title', '')).lower()
        
        relevance_keywords = [
            'academy', 'youth', 'prospect', 'talent', 'wonderkid', 'breakthrough', 
            'debut', 'signing', 'transfer', 'scout', 'potential', 'rising', 'emerging'
        ]
        
        age_keywords = ['young', '16', '17', '18', '19', '20', 'teenage', 'academy']
        position_keywords = position.lower().split() if position != "Any Position" else []
        
        score = 0
        score += sum(2 for keyword in relevance_keywords if keyword in text)
        score += sum(1 for keyword in age_keywords if keyword in text)
        score += sum(3 for keyword in position_keywords if keyword in text)
        
        # Bonus for scraped data
        if article.get('scraped_data'):
            score += 5
        
        return score
    
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
        
        # Check if we used Google CSE (better data quality)
        used_google_cse = any(item.get('source') != 'DuckDuckGo' for results in human_intel.values() for item in results)
        
        # Conservative scoring based on data availability
        if total_intel_items > 10 and high_relevance_items > 3:
            base_score = 7  # Reasonable baseline when we have good data
        elif total_intel_items > 5:
            base_score = 6  # Lower when limited data
        else:
            base_score = 5  # Very conservative when minimal data
        
        # Bonus for Google CSE data
        if used_google_cse and total_intel_items > 5:
            base_score = min(base_score + 1, 8)
        
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
        
        if used_google_cse:
            green_lights.append("Enhanced data quality from verified sources")
        
        # Default to honest assessment if no data
        if not red_flags:
            red_flags = ["Limited data available for comprehensive assessment"]
        
        if not green_lights:
            green_lights = ["Requires direct scouting for proper evaluation"]
        
        # Honest narrative about data limitations
        data_source = "Google CSE + Web Intelligence" if used_google_cse else "Web Intelligence"
        
        narrative = f"""
FERGUSON ASSESSMENT: {player_name}

ASSESSMENT STATUS: {reason}

DATA QUALITY REPORT:
- Intelligence items gathered: {total_intel_items}
- High-relevance sources: {high_relevance_items}
- Data source: {data_source}
- Data coverage: {'Good' if total_intel_items > 10 else 'Partial' if total_intel_items > 5 else 'Limited'}

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

def main(tab_context="main"):
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 APES Cognitive Scouting</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced Pattern Extraction System - Character-First Intelligence*")
    
    # Check if Google CSE is configured
    if st.session_state.google_cse_api_key:
        st.success("✅ Google CSE API Configured - Enhanced search enabled")
    else:
        st.info("💡 Using DuckDuckGo search. Add Google CSE API key for better results.")
    
    # Ferguson quote
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
        
        # Search Engine Config
        st.markdown("### 🔍 Search Engine")
        if st.session_state.google_cse_api_key:
            st.success("Google CSE Active")
        else:
            google_key = st.text_input(
                "Google CSE API Key (Optional)",
                type="password",
                help="Get free API key at https://developers.google.com/custom-search"
            )
            if google_key:
                st.session_state.google_cse_api_key = google_key
                st.rerun()
        
        st.divider()
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "Intelligence Provider",
            ["Groq (Free)", "Ollama (Local)", "Mock Analysis"],
            key=f"llm_provider_{tab_context}"
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
            if st.button(player, key=f"quick_{player}_{tab_context}"):
                st.session_state.player_name = player
        
        st.divider()
        
        # Quick Discovery Searches
        st.markdown("### 🌟 Quick Discovery")
        discovery_presets = [
            ("South American Midfielders", "18-20 (Breakthrough)", "South America", "Midfielder"),
            ("Eastern European Defenders", "21-23 (Emerging)", "Eastern Europe", "Defender"),
            ("African Wingers", "15-17 (Academy)", "Africa", "Forward"),
            ("Asian Prospects", "Any Age", "Asia", "Any Position")
        ]
        
        for preset_name, age, region, pos in discovery_presets:
            if st.button(preset_name, key=f"discover_{preset_name}_{tab_context}"):
                st.session_state.discovery_preset = {
                    'age_range': age,
                    'league_region': region, 
                    'position': pos,
                    'terms': ''
                }
        
        st.divider=st.divider()
        
        # Research history
        if st.session_state.research_history:
            st.markdown("### 📚 Recent Analysis")
            for i, research in enumerate(st.session_state.research_history[-3:]):
                if st.button(f"{research['player']} ({research['date']})", key=f"history_{i}_{tab_context}"):
                    st.session_state.player_name = research['player']
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    # Add mode selection
    analysis_mode = st.radio(
        "Analysis Mode:",
        ["🔍 Known Player Analysis", "🌟 Discovery Mode - Find Unknown Talents"],
        horizontal=True,
        key=f"analysis_mode_{tab_context}",
        index=0 if st.session_state.get('switch_to_known_analysis', False) else (1 if st.session_state.get('stay_in_discovery', False) else 0)
    )
    
    # Clear the switch flag after using it
    if st.session_state.get('switch_to_known_analysis', False):
        st.session_state.switch_to_known_analysis = False
        analysis_mode = "🔍 Known Player Analysis"
    
    if analysis_mode == "🔍 Known Player Analysis":
        with col1:
            player_name = st.text_input(
                "Enter Player Name:",
                value=st.session_state.get('player_name', ''),
                placeholder="e.g., Pedri Barcelona, Jude Bellingham, Jamal Musiala"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            analyze_button = st.button("🧠 Character Analysis", type="primary", use_container_width=True)
    
    else:  # Discovery Mode
        st.markdown("### 🌟 Discovery Mode - Find Hidden Talents")
        
        # Check for preset discovery parameters
        preset = st.session_state.get('discovery_preset', None)
        
        disco1, disco2, disco3 = st.columns(3)
        with disco1:
            age_range = st.selectbox(
                "Age Range",
                ["15-17 (Academy)", "18-20 (Breakthrough)", "21-23 (Emerging)", "Any Age"],
                index=["15-17 (Academy)", "18-20 (Breakthrough)", "21-23 (Emerging)", "Any Age"].index(preset['age_range']) if preset else 0,
                key=f"discovery_age_{tab_context}"
            )
        
        with disco2:
            league_region = st.selectbox(
                "League/Region",
                ["South America", "Eastern Europe", "Africa", "Asia", "Lower European Leagues", "Any Region"],
                index=["South America", "Eastern Europe", "Africa", "Asia", "Lower European Leagues", "Any Region"].index(preset['league_region']) if preset else 0,
                key=f"discovery_region_{tab_context}"
            )
        
        with disco3:
            position = st.selectbox(
                "Position",
                ["Midfielder", "Forward", "Defender", "Goalkeeper", "Any Position"],
                index=["Midfielder", "Forward", "Defender", "Goalkeeper", "Any Position"].index(preset['position']) if preset else 0,
                key=f"discovery_pos_{tab_context}"
            )
        
        # Clear preset after use
        if preset:
            del st.session_state.discovery_preset
        
        # Discovery search terms
        col_search1, col_search2 = st.columns([3, 1])
        with col_search1:
            discovery_terms = st.text_input(
                "Additional Search Terms (optional):",
                placeholder="e.g., wonderkid, academy graduate, breakthrough, rising star",
                key=f"discovery_terms_{tab_context}"
            )
        
        with col_search2:
            st.markdown("<br>", unsafe_allow_html=True)
            discover_button = st.button("🔍 Discover Talents", type="primary", use_container_width=True, key=f"discover_btn_{tab_context}")
        
        player_name = None  # No specific name in discovery mode
        analyze_button = discover_button
    
    if analyze_button:
        if analysis_mode == "🔍 Known Player Analysis" and not player_name:
            st.warning("Please enter a player name")
        elif analysis_mode == "🌟 Discovery Mode - Find Unknown Talents":
            # Discovery Mode Flow
            tools = FergusonTools()
            
            with st.spinner("🔍 Discovering hidden talents..."):
                discovered_players = tools.discover_unknown_talents(
                    age_range, league_region, position, discovery_terms
                )
            
            if discovered_players:
                st.success(f"🌟 Discovered {len(discovered_players)} potential talents!")
                
                # Show discovered players
                st.markdown("### 🎯 Discovered Talents")
                
                for i, player_info in enumerate(discovered_players):
                    relevance_color = "🔥" if player_info['source_relevance'] >= 10 else "⭐" if player_info['source_relevance'] >= 5 else "💡"
                    
                    with st.expander(f"{relevance_color} {player_info['name']} - Relevance Score: {player_info['source_relevance']:.1f}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Name:** {player_info['name']}")
                            st.markdown(f"**Age Info:** {player_info['age_info'] if player_info['age_info'] else 'Unknown'}")
                            st.markdown(f"**Context:** {player_info['context']}")
                            st.markdown(f"**Source:** [{player_info['source_title']}]({player_info['source_link']})")
                            st.markdown(f"**Discovery Query:** {player_info['discovery_query']}")
                            
                            # Show additional scraped data if available
                            if player_info.get('position'):
                                st.markdown(f"**Position:** {player_info['position']}")
                            if player_info.get('goals'):
                                st.markdown(f"**Goals:** {player_info['goals']}")
                            if player_info.get('assists'):
                                st.markdown(f"**Assists:** {player_info['assists']}")
                            
                            # Show relevance breakdown for high scores
                            if player_info['source_relevance'] >= 5:
                                st.markdown("**🎯 High Relevance Match!**")
                        
                        with col2:
                            # Use a form to prevent immediate rerun
                            with st.form(key=f"analyze_form_{i}_{tab_context}"):
                                analyze_clicked = st.form_submit_button(f"🧠 Analyze {player_info['name']}", use_container_width=True)
                                
                                if analyze_clicked:
                                    # Set the discovered player for analysis and switch mode
                                    st.session_state.player_name = player_info['name']
                                    st.session_state.discovery_context = player_info
                                    st.session_state.switch_to_known_analysis = True
                                    st.rerun()
                
                st.info("💡 **Tip:** Click 'Analyze' on any discovered player to run full character assessment")
            else:
                st.warning("🔍 No talents discovered with current criteria. Try different search parameters.")
            
            return  # Exit early for discovery mode
        
        else:
            # Standard Known Player Analysis
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
            
            # Add discovery context if available
            discovery_context = st.session_state.get('discovery_context', None)
            if discovery_context:
                st.info(f"🌟 **Discovered Player:** {discovery_context['name']} - {discovery_context['context']}")
            
            # Perform analysis
            with st.spinner("Conducting character-first analysis..."):
                player_intel = asyncio.run(complete_ferguson_analysis(player_name, tools, llm))
            
            # Add discovery info to analysis if available
            if discovery_context:
                player_intel.raw_data['discovery_info'] = discovery_context
                # Clear discovery context
                del st.session_state.discovery_context
            
            # Add to history
            st.session_state.research_history.append({
                'player': player_name,
                'date': datetime.now().strftime('%m/%d'),
                'ferguson_factor': player_intel.ferguson_factor
            })
            
            # Display results
            st.success("Character analysis complete!")

# Show discovery badge if this was a discovered player
            if 'discovery_info' in player_intel.raw_data:
                st.markdown("""
                <div style="background: linear-gradient(90deg, #FFD700, #FFA500); color: black; padding: 0.5rem; border-radius: 5px; text-align: center; margin: 1rem 0;">
                    🌟 <strong>DISCOVERY:</strong> This player was found through APES Discovery Mode
                </div>
                """, unsafe_allow_html=True)
            
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
                
                # Check if Google CSE was used
                used_google = any(item.get('source') != 'DuckDuckGo' 
                                for results in player_intel.raw_data.get('human_intel', {}).values() 
                                for item in results)
                
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
                if used_google:
                    st.markdown("*Using Google CSE for enhanced results*")
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
                                    st.markdown(f"*Source: {result.get('source', 'Unknown')}*")
                                    st.markdown(result['snippet'][:200] + "...")
                                    
                                    # Show scraped data if available
                                    if result.get('scraped_data'):
                                        scraped = result['scraped_data']
                                        if scraped.get('age'):
                                            st.markdown(f"*Extracted Age: {scraped['age']}*")
                                        if scraped.get('position'):
                                            st.markdown(f"*Extracted Position: {scraped['position']}*")
                                    
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
        player1 = st.text_input("Player 1", placeholder="e.g., Pedri", key="comparison_player1")
    
    with col2:
        player2 = st.text_input("Player 2", placeholder="e.g., Gavi", key="comparison_player2")
    
    if st.button("🔄 Compare Players", key="compare_players_button") and player1 and player2:
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
    
    - **Intelligence Gathering:** Google CSE + DuckDuckGo fallback
    - **Enhanced Scraping:** BeautifulSoup with pattern extraction
    - **Analysis Engine:** Groq/Ollama LLM with character-focused prompting
    - **Visualization:** Plotly for interactive charts and dashboards
    - **Zero Cost Options:** Works with free APIs and local models
    
    #### Enhanced with Google CSE
    
    This version integrates Google Custom Search Engine for:
    - Better search quality and relevance
    - Structured data extraction
    - Enhanced scraping capabilities
    - Fallback to DuckDuckGo when quota exceeded
    
    #### Success Stories
    
    The underlying APES system has already identified:
    - **Justin Lerma (2008)** - Flagged before Borussia Dortmund acquisition
    - **Bence Dárdai (2006)** - Identified when market value was €0.9M
    
    #### Future Evolution: Sócrates
    
    APES Cognitive is the bridge toward **Sócrates**, an autonomous football intelligence 
    that will understand the poetry of football itself - not just analyzing the game, 
    but comprehending the human stories that determine success.
    
    ---
    
    **Built by:** APES Development Team  
    **Version:** Cognitive 2.0 - Enhanced with Google CSE  
    **License:** Zero Cost, Maximum Impact  
    """)

# Navigation
def main_navigation():
    """Main navigation and page routing"""
    
    # Navigation tabs
    nav_tabs = st.tabs(["🧠 Character Analysis", "⚖️ Compare Players", "ℹ️ About"])
    
    with nav_tabs[0]:
        try:
            main("tab1")  # Main analysis tool
        except Exception as e:
            st.error(f"Error in main analysis: {str(e)}")
            st.info("Please try refreshing the page or contact support if the problem persists.")
    
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
    
    # Run main navigation directly
    main_navigation()
