# apes_zero_cost_agent.py
"""
APES Football Intelligence - Zero Cost Version
No expensive APIs required!
"""

import asyncio
import streamlit as st
from typing import Dict, Any, List
import time
from datetime import datetime
import json

# Free alternatives
from duckduckgo_search import DDGS
import feedparser
import requests
from bs4 import BeautifulSoup
import wikipedia
from pytrends.request import TrendReq
import pandas as pd

# Free LLM options
try:
    from groq import Groq  # Option 1: Groq (free tier)
except:
    pass

try:
    from langchain_community.llms import Ollama  # Option 2: Local LLM
except:
    pass

# Set page configuration
st.set_page_config(
    page_title="APES Football Intelligence - Zero Cost",
    page_icon="⚽",
    layout="wide"
)

# Initialize session state
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

# Sidebar configuration
with st.sidebar:
    st.title("🎯 APES Configuration")
    st.markdown("### Zero Cost Intelligence")
    
    # LLM Selection
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["Groq (Free Tier)", "Ollama (Local)", "Mock Mode (No LLM)"]
    )
    
    if llm_provider == "Groq (Free Tier)":
        groq_api_key = st.text_input(
            "Groq API Key (Free at groq.com)",
            value=st.session_state.groq_api_key,
            type="password",
            help="Get free API key at https://console.groq.com"
        )
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            st.session_state.llm_provider = "groq"
    
    elif llm_provider == "Ollama (Local)":
        st.info("Make sure Ollama is running locally with a model installed")
        st.code("ollama run llama3", language="bash")
        st.session_state.llm_provider = "ollama"
    
    else:
        st.session_state.llm_provider = "mock"
    
    st.divider()
    
    # Quick searches
    st.markdown("### 🚀 Quick Searches")
    quick_searches = [
        "Liverpool youth academy transfers",
        "Barcelona La Masia graduates 2025",
        "African wonderkids Europe",
        "Premier League academy signings"
    ]
    
    for search in quick_searches:
        if st.button(search, key=f"quick_{search}"):
            st.session_state.research_topic = search

# Free Research Tools
class APESFreeTools:
    """All free tools for APES intelligence gathering"""
    
    def __init__(self):
        self.ddgs = DDGS()
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_web(self, query: str, max_results: int = 10) -> List[Dict]:
        """Free web search using DuckDuckGo"""
        try:
            results = []
            for r in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'link': r.get('link', ''),
                    'snippet': r.get('body', '')
                })
            return results
        except Exception as e:
            st.warning(f"Search error: {e}")
            return []
    
    def get_news_feeds(self, topic: str) -> List[Dict]:
        """Get news from RSS feeds - completely free"""
        feeds = [
            f"https://news.google.com/rss/search?q={topic}+football+transfer",
            "https://www.theguardian.com/football/rss",
            "https://www.goal.com/feeds",
            "https://www.football-italia.net/rss"
        ]
        
        all_news = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    all_news.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', '')[:200]
                    })
            except:
                continue
        
        return all_news
    
    def check_wikipedia(self, query: str) -> Dict:
        """Wikipedia check - free API"""
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=3)
            
            if search_results:
                # Get first result
                page = wikipedia.page(search_results[0])
                
                # Check recent changes (simplified)
                return {
                    'found': True,
                    'title': page.title,
                    'url': page.url,
                    'summary': page.summary[:300],
                    'last_check': datetime.now().isoformat()
                }
            else:
                return {'found': False, 'query': query}
                
        except Exception as e:
            return {'error': str(e), 'found': False}
    
    def analyze_trends(self, queries: List[str]) -> Dict:
        """Google Trends analysis - unofficial but free"""
        try:
            # Build payload
            self.pytrends.build_payload(queries, timeframe='now 7-d')
            
            # Get interest over time
            interest = self.pytrends.interest_over_time()
            
            if not interest.empty:
                # Analyze for spikes
                results = {}
                for query in queries:
                    if query in interest.columns:
                        data = interest[query]
                        latest = data.iloc[-1]
                        avg = data.mean()
                        spike = latest > avg * 1.5
                        
                        results[query] = {
                            'latest_interest': int(latest),
                            'average': int(avg),
                            'spike_detected': spike,
                            'trend': 'UP' if spike else 'NORMAL'
                        }
                
                return results
            else:
                return {'error': 'No trend data available'}
                
        except Exception as e:
            return {'error': f'Trend analysis failed: {str(e)}'}
    
    def scrape_transfermarkt_basic(self, player_name: str) -> Dict:
        """Basic Transfermarkt info - no API needed"""
        try:
            # Simple search URL
            search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name}"
            
            # Note: In production, use cloudscraper or selenium for better results
            return {
                'search_url': search_url,
                'note': 'Direct scraping requires more setup',
                'alternative': 'Use news feeds for transfer info'
            }
        except:
            return {'error': 'Scraping failed'}

# Free LLM Analysis
class APESFreeLLM:
    """Handle LLM analysis with free options"""
    
    def __init__(self, provider: str, api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "groq" and api_key:
            self.client = Groq(api_key=api_key)
        elif provider == "ollama":
            self.llm = Ollama(model="llama3")  # or mistral, phi3
        else:
            self.client = None
    
    def analyze(self, data: Dict, topic: str) -> str:
        """Analyze gathered data using free LLM"""
        
        prompt = f"""
        You are APES (Advanced Pattern Extraction System), a football intelligence analyst.
        
        Research Topic: {topic}
        
        Analyze this data for football transfer patterns:
        
        Web Search Results: {json.dumps(data.get('search_results', []), indent=2)}
        
        News Feed Data: {json.dumps(data.get('news_feeds', []), indent=2)}
        
        Wikipedia Info: {json.dumps(data.get('wikipedia', {}), indent=2)}
        
        Trend Analysis: {json.dumps(data.get('trends', {}), indent=2)}
        
        Provide a structured intelligence report with:
        1. Executive Summary
        2. Key Signals Detected
        3. Transfer Probability (if applicable)
        4. Timeline Assessment
        5. Recommended Actions
        
        Focus on patterns that indicate player movements, academy graduations, or transfer activity.
        """
        
        if self.provider == "groq" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",  # Free!
                    messages=[
                        {"role": "system", "content": "You are APES Football Intelligence Analyst"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Groq API error: {str(e)}"
        
        elif self.provider == "ollama":
            try:
                return self.llm.invoke(prompt)
            except:
                return "Ollama not running. Start with: ollama run llama3"
        
        else:
            # Mock mode - return structured analysis
            return f"""
# APES Intelligence Report

## Executive Summary
Analysis completed for: {topic}

## Key Findings
- Web search returned {len(data.get('search_results', []))} results
- News feeds provided {len(data.get('news_feeds', []))} articles
- Wikipedia status: {data.get('wikipedia', {}).get('found', False)}
- Trend analysis: {data.get('trends', {}).get('trend', 'No data')}

## Recommendations
1. Monitor these sources daily
2. Set up alerts for key terms
3. Track Wikipedia page changes
4. Analyze social media patterns

*Note: Running in mock mode - install Groq or Ollama for AI analysis*
"""

# Main Research Function
async def deep_research_free(topic: str, tools: APESFreeTools, llm: APESFreeLLM) -> Dict[str, Any]:
    """Perform deep research using only free tools"""
    
    research_data = {
        'topic': topic,
        'timestamp': datetime.now().isoformat(),
        'sources': {}
    }
    
    # Progress tracking
    progress = st.progress(0)
    status = st.empty()
    
    # Step 1: Web Search
    status.text("🔍 Searching the web...")
    search_results = tools.search_web(f"{topic} transfer news 2025")
    research_data['search_results'] = search_results
    progress.progress(25)
    
    # Step 2: News Feeds
    status.text("📰 Gathering news feeds...")
    news_feeds = tools.get_news_feeds(topic)
    research_data['news_feeds'] = news_feeds
    progress.progress(50)
    
    # Step 3: Wikipedia Check
    status.text("📚 Checking Wikipedia...")
    wiki_data = tools.check_wikipedia(topic)
    research_data['wikipedia'] = wiki_data
    progress.progress(75)
    
    # Step 4: Trend Analysis
    status.text("📈 Analyzing trends...")
    # Extract key terms for trend analysis
    terms = topic.split()[:3]  # Simple extraction
    trends = tools.analyze_trends(terms)
    research_data['trends'] = trends
    progress.progress(90)
    
    # Step 5: LLM Analysis
    status.text("🧠 Generating intelligence report...")
    report = llm.analyze(research_data, topic)
    research_data['analysis'] = report
    progress.progress(100)
    
    status.text("✅ Research complete!")
    time.sleep(1)
    progress.empty()
    status.empty()
    
    return research_data

# Main Interface
st.title("⚽ APES Football Intelligence - Zero Cost Edition")
st.markdown("*Advanced Pattern Extraction System - No expensive APIs required!*")

# Research input
research_topic = st.text_input(
    "Enter research topic:",
    value=st.session_state.get('research_topic', ''),
    placeholder="e.g., Liverpool youth academy signings"
)

# Research button
if st.button("🚀 Start Intelligence Gathering", type="primary"):
    if not research_topic:
        st.warning("Please enter a research topic")
    else:
        # Initialize tools
        tools = APESFreeTools()
        
        # Initialize LLM based on selection
        if st.session_state.llm_provider == "groq":
            if not st.session_state.groq_api_key:
                st.error("Please enter Groq API key in sidebar (it's free!)")
                st.stop()
            llm = APESFreeLLM("groq", st.session_state.groq_api_key)
        elif st.session_state.llm_provider == "ollama":
            llm = APESFreeLLM("ollama")
        else:
            llm = APESFreeLLM("mock")
        
        # Run research
        research_data = asyncio.run(deep_research_free(research_topic, tools, llm))
        
        # Display results
        st.success("Intelligence gathering complete!")
        
        # Main report
        st.markdown("## 📋 Intelligence Report")
        st.markdown(research_data['analysis'])
        
        # Additional data in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Web Search", "📰 News", "📚 Wikipedia", "📈 Trends"])
        
        with tab1:
            st.markdown("### Web Search Results")
            for idx, result in enumerate(research_data['search_results'][:5]):
                st.markdown(f"**{idx+1}. [{result['title']}]({result['link']})**")
                st.markdown(f"*{result['snippet']}*")
                st.divider()
        
        with tab2:
            st.markdown("### Latest News")
            for news in research_data['news_feeds'][:5]:
                st.markdown(f"**[{news['title']}]({news['link']})**")
                st.markdown(f"*Published: {news['published']}*")
                st.markdown(news['summary'])
                st.divider()
        
        with tab3:
            st.markdown("### Wikipedia Information")
            wiki = research_data['wikipedia']
            if wiki.get('found'):
                st.markdown(f"**Page:** [{wiki['title']}]({wiki['url']})")
                st.markdown(wiki['summary'])
            else:
                st.info("No Wikipedia page found")
        
        with tab4:
            st.markdown("### Trend Analysis")
            st.json(research_data['trends'])
        
        # Download report
        st.download_button(
            "📥 Download Full Report",
            json.dumps(research_data, indent=2),
            file_name=f"apes_report_{research_topic.replace(' ', '_')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("🚀 **APES Zero Cost** - No expensive APIs required!")
st.markdown("Using: DuckDuckGo (search) • RSS Feeds (news) • Wikipedia (data) • Groq/Ollama (analysis)")