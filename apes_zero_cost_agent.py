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
    from groq import Groq
except:
    pass

try:
    from langchain_community.llms import Ollama
except:
    pass

st.set_page_config(
    page_title="APES Football Intelligence - Zero Cost",
    page_icon="⚽",
    layout="wide"
)

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

with st.sidebar:
    st.title("🎯 APES Configuration")
    st.markdown("### Zero Cost Intelligence")

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

class APESFreeTools:
    def __init__(self):
        self.ddgs = DDGS()
        self.pytrends = TrendReq(hl='en-US', tz=360)

    def search_web(self, query: str, max_results: int = 10) -> List[Dict]:
        try:
            return [{
                'title': r.get('title', ''),
                'link': r.get('link', ''),
                'snippet': r.get('body', '')
            } for r in self.ddgs.text(query, max_results=max_results)]
        except Exception as e:
            st.warning(f"Search error: {e}")
            return []

    def get_news_feeds(self, topic: str) -> List[Dict]:
        feeds = [
            f"https://news.google.com/rss/search?q={topic}+football+transfer",
            "https://www.theguardian.com/football/rss",
            "https://www.goal.com/feeds",
            "https://www.football-italia.net/rss"
        ]
        news = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    news.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', '')[:200]
                    })
            except:
                continue
        return news

    def check_wikipedia(self, query: str) -> Dict:
        try:
            results = wikipedia.search(query, results=3)
            if results:
                page = wikipedia.page(results[0])
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
        try:
            self.pytrends.build_payload(queries, timeframe='now 7-d')
            interest = self.pytrends.interest_over_time()
            if not interest.empty:
                return {
                    query: {
                        'latest_interest': int(interest[query].iloc[-1]),
                        'average': int(interest[query].mean()),
                        'spike_detected': interest[query].iloc[-1] > interest[query].mean() * 1.5,
                        'trend': 'UP' if interest[query].iloc[-1] > interest[query].mean() * 1.5 else 'NORMAL'
                    } for query in queries if query in interest.columns
                }
            return {'error': 'No trend data available'}
        except Exception as e:
            return {'error': f'Trend analysis failed: {str(e)}'}

class APESFreeLLM:
    def __init__(self, provider: str, api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        if provider == "groq" and api_key:
            self.client = Groq(api_key=api_key)
        elif provider == "ollama":
            self.llm = Ollama(model="llama3")
        else:
            self.client = None

    def analyze(self, data: Dict, topic: str) -> str:
        try:
            trends_json = json.dumps(data.get('trends', {}), indent=2)
        except TypeError:
            trends = data.get('trends', {})
            trends_json = json.dumps({k: str(v) for k, v in trends.items()}, indent=2)

        prompt = f"""
        You are APES (Advanced Pattern Extraction System), a football intelligence analyst.

        Research Topic: {topic}

        Analyze this data:
        Web: {json.dumps(data.get('search_results', []), indent=2)}
        News: {json.dumps(data.get('news_feeds', []), indent=2)}
        Wikipedia: {json.dumps(data.get('wikipedia', {}), indent=2)}
        Trends: {trends_json}

        Output a structured intelligence report:
        - Executive Summary
        - Key Signals
        - Transfer Probability
        - Timeline
        - Recommended Actions
        """

        if self.provider == "groq" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
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
            return f"""
# APES Intelligence Report

## Executive Summary
Topic: {topic}

## Key Findings
- Web: {len(data.get('search_results', []))} results
- News: {len(data.get('news_feeds', []))} articles
- Wikipedia: {data.get('wikipedia', {}).get('found', False)}
- Trends: {data.get('trends', {}).get('trend', 'No data')}

## Recommendations
- Monitor sources
- Track page changes
- Watch social spikes

*Mock Mode Active*
"""

async def deep_research_free(topic: str, tools: APESFreeTools, llm: APESFreeLLM) -> Dict[str, Any]:
    data = {'topic': topic, 'timestamp': datetime.now().isoformat()}
    progress = st.progress(0)
    status = st.empty()

    status.text("🔍 Searching the web...")
    data['search_results'] = tools.search_web(f"{topic} transfer news 2025")
    progress.progress(25)

    status.text("📰 Gathering news feeds...")
    data['news_feeds'] = tools.get_news_feeds(topic)
    progress.progress(50)

    status.text("📚 Checking Wikipedia...")
    data['wikipedia'] = tools.check_wikipedia(topic)
    progress.progress(75)

    status.text("📈 Analyzing trends...")
    data['trends'] = tools.analyze_trends(topic.split()[:3])
    progress.progress(90)

    status.text("🧠 Generating report...")
    data['analysis'] = llm.analyze(data, topic)
    progress.progress(100)

    time.sleep(1)
    progress.empty()
    status.empty()
    return data

def make_serializable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        else:
            return str(obj)

st.title("⚽ APES Football Intelligence - Zero Cost Edition")
st.markdown("*Advanced Pattern Extraction System – No expensive APIs required!*")

research_topic = st.text_input(
    "Enter research topic:",
    value=st.session_state.get('research_topic', ''),
    placeholder="e.g., Liverpool youth academy signings"
)

if st.button("🚀 Start Intelligence Gathering", type="primary"):
    if not research_topic:
        st.warning("Please enter a research topic")
    else:
        tools = APESFreeTools()
        provider = st.session_state.llm_provider
        llm = APESFreeLLM(provider, st.session_state.groq_api_key if provider == "groq" else None)
        if provider == "groq" and not st.session_state.groq_api_key:
            st.error("Please enter Groq API key")
            st.stop()

        data = asyncio.run(deep_research_free(research_topic, tools, llm))
        st.success("Intelligence gathering complete!")

        st.markdown("## 📋 Intelligence Report")
        st.markdown(data['analysis'])

        tabs = st.tabs(["🔍 Web", "📰 News", "📚 Wiki", "📈 Trends"])
        with tabs[0]:
            for idx, r in enumerate(data['search_results'][:5]):
                st.markdown(f"**{idx+1}. [{r['title']}]({r['link']})**")
                st.markdown(f"*{r['snippet']}*")
                st.divider()
        with tabs[1]:
            for news in data['news_feeds'][:5]:
                st.markdown(f"**[{news['title']}]({news['link']})**")
                st.markdown(f"*Published: {news['published']}*")
                st.markdown(news['summary'])
                st.divider()
        with tabs[2]:
            wiki = data['wikipedia']
            if wiki.get('found'):
                st.markdown(f"**Page:** [{wiki['title']}]({wiki['url']})")
                st.markdown(wiki['summary'])
            else:
                st.info("No Wikipedia page found")
        with tabs[3]:
            st.json(data['trends'])

        serializable_data = make_serializable(data)
        st.download_button(
            "📅 Download Full Report",
            json.dumps(serializable_data, indent=2),
            file_name=f"apes_report_{research_topic.replace(' ', '_')}.json",
            mime="application/json"
        )

st.markdown("---")
st.markdown("APES Zero Cost - Built with Open Tools")
