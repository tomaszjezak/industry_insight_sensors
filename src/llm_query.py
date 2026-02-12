"""
LLM interface for natural language queries.
Converts natural language questions to structured queries.
"""

import re
from typing import Dict, Optional, Tuple
import os

from .query_engine import QueryEngine


class LLMQueryInterface:
    """
    Natural language query interface.
    Uses LLM (OpenAI or Llama) or pattern matching to convert NL to structured queries.
    """
    
    def __init__(self, query_engine: QueryEngine = None, use_openai: bool = True):
        """
        Initialize LLM query interface.
        
        Args:
            query_engine: QueryEngine instance
            use_openai: Whether to try OpenAI API (requires OPENAI_API_KEY)
        """
        self.query_engine = query_engine or QueryEngine()
        self.use_openai = use_openai and os.getenv('OPENAI_API_KEY') is not None
        
        if self.use_openai:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.llm_available = True
            except ImportError:
                self.llm_available = False
        else:
            self.llm_available = False
    
    def query(self, question: str) -> Dict:
        """
        Process natural language query.
        
        Args:
            question: Natural language question
        
        Returns:
            Query results with interpretation
        """
        # Try LLM first if available
        if self.llm_available:
            structured = self._llm_parse(question)
        else:
            structured = self._pattern_parse(question)
        
        if structured:
            # Execute structured query
            result = self.query_engine.query(**structured)
            return {
                'question': question,
                'interpretation': structured,
                'result': result,
            }
        else:
            return {
                'question': question,
                'error': 'Could not parse query',
                'suggestions': [
                    'How many containers between 2019 and 2021?',
                    'What changed from February 2019 to December 2021?',
                    'Estimate revenue for 2021-12-01',
                    'Show inventory timeline',
                ]
            }
    
    def _llm_parse(self, question: str) -> Optional[Dict]:
        """
        Use LLM to parse natural language query.
        
        Args:
            question: Natural language question
        
        Returns:
            Structured query dict or None
        """
        if not self.llm_available:
            return None
        
        try:
            prompt = f"""Convert this question about recycling yard timelapse data into a structured query.

Question: {question}

Available query types:
- inventory: Get inventory levels (requires date or date_range)
- containers: Get container counts (requires date_range)
- changes: Get changes between dates (requires date_range)
- revenue: Get revenue estimates (requires date or date_range)
- timeline: Get timeline data (optional date_range, optional metric: volume/area/tonnage/containers)

Date format: YYYY-MM-DD or YYYY-MM or just YYYY

Respond in JSON format:
{{"query_type": "...", "date": "...", "date_range": ["...", "..."], "metric": "..."}}

If dates are mentioned, extract them. If no dates, use null.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query parser. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"[!] LLM parse error: {e}")
            return None
    
    def _pattern_parse(self, question: str) -> Optional[Dict]:
        """
        Use pattern matching to parse natural language query.
        
        Args:
            question: Natural language question
        
        Returns:
            Structured query dict or None
        """
        question_lower = question.lower()
        
        # Extract dates
        date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{4})-(\d{2})',          # YYYY-MM
            r'(\d{4})',                  # YYYY
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, question_lower)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 3:  # YYYY-MM-DD
                        dates.append(f"{match[0]}-{match[1]}-{match[2]}")
                    elif len(match) == 2:
                        if match[0].isdigit():  # YYYY-MM
                            dates.append(f"{match[0]}-{match[1]}-01")
                        else:  # Month name
                            month_map = {
                                'january': '01', 'february': '02', 'march': '03',
                                'april': '04', 'may': '05', 'june': '06',
                                'july': '07', 'august': '08', 'september': '09',
                                'october': '10', 'november': '11', 'december': '12',
                            }
                            month = month_map.get(match[0].lower(), '01')
                            dates.append(f"{match[1]}-{month}-01")
                else:  # Single year
                    dates.append(f"{match}-01-01")
        
        # Determine query type
        query_type = None
        if 'container' in question_lower:
            query_type = 'containers'
        elif 'change' in question_lower or 'arrival' in question_lower or 'departure' in question_lower:
            query_type = 'changes'
        elif 'revenue' in question_lower or 'value' in question_lower or 'worth' in question_lower:
            query_type = 'revenue'
        elif 'timeline' in question_lower or 'over time' in question_lower or 'trend' in question_lower:
            query_type = 'timeline'
        elif 'inventory' in question_lower or 'how many' in question_lower or 'what' in question_lower:
            query_type = 'inventory'
        
        if not query_type:
            return None
        
        # Build structured query
        structured = {'query_type': query_type}
        
        if len(dates) == 1:
            structured['date'] = dates[0]
        elif len(dates) >= 2:
            # Sort dates to ensure start < end
            sorted_dates = sorted(dates)
            structured['date_range'] = (sorted_dates[0], sorted_dates[-1])
        
        # Extract metric for timeline
        if query_type == 'timeline':
            if 'volume' in question_lower:
                structured['metric'] = 'volume'
            elif 'area' in question_lower:
                structured['metric'] = 'area'
            elif 'tonnage' in question_lower or 'ton' in question_lower:
                structured['metric'] = 'tonnage'
            elif 'container' in question_lower:
                structured['metric'] = 'containers'
        
        return structured


if __name__ == "__main__":
    # Test LLM query interface
    interface = LLMQueryInterface()
    
    test_queries = [
        "How many containers between 2019 and 2021?",
        "What changed from February 2019 to December 2021?",
        "Estimate revenue for 2021-12-01",
        "Show inventory timeline",
        "How many pallets came in from 2019 to 2021?",
    ]
    
    print("[*] Testing natural language queries...")
    
    for query in test_queries:
        print(f"\n[Q] {query}")
        result = interface.query(query)
        
        if 'error' in result:
            print(f"    [!] {result['error']}")
        else:
            print(f"    [Interpretation] {result['interpretation']}")
            if 'result' in result and 'summary' in result['result']:
                summary = result['result']['summary']
                print(f"    [Result] {summary}")

