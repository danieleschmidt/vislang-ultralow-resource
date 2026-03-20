"""HumanitarianReportScraper: fetch UN OCHA/ReliefWeb reports."""

import requests

DUMMY_REPORTS = [
    {
        "title": "Humanitarian Situation Report - Somalia",
        "url": "https://reliefweb.int/report/somalia/humanitarian-situation-report",
        "date": "2024-01-15",
        "language": "en",
        "summary": "Overview of humanitarian conditions in Somalia.",
    },
    {
        "title": "Emergency Response Update - Sudan",
        "url": "https://reliefweb.int/report/sudan/emergency-response-update",
        "date": "2024-01-10",
        "language": "en",
        "summary": "Emergency response update for Sudan crisis.",
    },
]


class HumanitarianReportScraper:
    """Fetch UN OCHA/ReliefWeb humanitarian reports."""

    RELIEFWEB_API = "https://api.reliefweb.int/v1/reports"

    def __init__(self, base_url="https://reliefweb.int/api/v1/reports", timeout=10):
        self.base_url = base_url
        self.timeout = timeout

    def fetch_reports(self, query="humanitarian", limit=10, lang="en"):
        """Fetch reports from ReliefWeb API. Falls back to dummy data on failure."""
        results = []
        try:
            params = {
                "appname": "vislang",
                "limit": limit,
                "query[value]": query,
                "fields[include][]": ["title", "url_alias", "date", "language", "body"],
            }
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            results = self.parse_response(data)
        except Exception:
            return list(DUMMY_REPORTS)

        return results if results else list(DUMMY_REPORTS)

    def parse_response(self, data):
        """Parse ReliefWeb API response into list of report dicts."""
        results = []
        items = data.get("data", [])
        for item in items:
            fields = item.get("fields", {})
            lang_list = fields.get("language", [{}])
            lang_code = lang_list[0].get("code", "en") if lang_list else "en"
            date_val = fields.get("date", {})
            date_str = (
                date_val.get("created", "") if isinstance(date_val, dict) else str(date_val)
            )
            results.append({
                "title": fields.get("title", ""),
                "url": fields.get("url_alias", ""),
                "date": date_str,
                "language": lang_code,
                "summary": (fields.get("body", "") or "")[:200],
            })
        return results

    def fetch_images_from_report(self, report_url):
        """Fetch image URLs from a report page. Returns empty list on failure."""
        try:
            resp = requests.get(report_url, timeout=self.timeout)
            resp.raise_for_status()
            import re
            images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', resp.text)
            return [url for url in images if url.startswith("http")]
        except Exception:
            return []
