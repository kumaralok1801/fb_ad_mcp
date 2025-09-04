"""
fb_api_mcp_server.py
- Facebook Graph API (v22.0) MCP server, exposed over HTTP + SSE.
- Works with Claude/other MCP hosts as a remote HTTP server.
- Requires env FB_ACCESS_TOKEN or --fb-token CLI arg.

Run:
    export FB_ACCESS_TOKEN="YOUR_TOKEN"
    python fb_api_mcp_server.py
    # SSE endpoint at http://0.0.0.0:8000/sse
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# If your FastMCP comes from a different package path, adjust this import.
from fastmcp import FastMCP

# --- Constants ---
FB_API_VERSION = "v22.0"
FB_GRAPH_URL = f"https://graph.facebook.com/{FB_API_VERSION}"
DEFAULT_AD_ACCOUNT_FIELDS = [
    'name', 'business_name', 'age', 'account_status', 'balance',
    'amount_spent', 'attribution_spec', 'account_id', 'business',
    'business_city', 'brand_safety_content_filter_levels', 'currency',
    'created_time', 'id'
]

# Create an MCP server
mcp = FastMCP("fb-api-mcp-server")

# Global token cache
FB_ACCESS_TOKEN: Optional[str] = None

# Reusable HTTP session
_SESSION = requests.Session()
# Tune timeouts as needed (connect, read)
_DEFAULT_TIMEOUT = (10, 60)

# --- Utilities & Core HTTP helpers ---

def _get_fb_access_token() -> str:
    """
    Resolve Facebook access token from:
      1) in-memory cache
      2) CLI: --fb-token <token>
      3) ENV: FB_ACCESS_TOKEN
    """
    global FB_ACCESS_TOKEN
    if FB_ACCESS_TOKEN:
        return FB_ACCESS_TOKEN

    token: Optional[str] = None

    # CLI flag
    if "--fb-token" in sys.argv:
        idx = sys.argv.index("--fb-token") + 1
        if idx < len(sys.argv):
            token = sys.argv[idx]
        else:
            raise Exception("--fb-token argument provided but no token value followed it")

    # ENV fallback
    if not token:
        token = os.getenv("FB_ACCESS_TOKEN")

    if not token:
        raise Exception("Facebook token must be provided via '--fb-token' or FB_ACCESS_TOKEN env var")

    FB_ACCESS_TOKEN = token
    print("Using Facebook token from configuration")
    return FB_ACCESS_TOKEN


def _redact_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied dict with access_token redacted for logging."""
    if not params:
        return {}
    redacted = dict(params)
    if 'access_token' in redacted:
        redacted['access_token'] = '***REDACTED***'
    return redacted


def _request_with_retries(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 4,
    backoff_factor: float = 0.7,
    timeout: Tuple[int, int] = _DEFAULT_TIMEOUT,
) -> requests.Response:
    """
    HTTP request with retries on 429 and 5xx.
    - Exponential backoff with jitter.
    """
    attempt = 0
    while True:
        try:
            resp = _SESSION.request(method, url, params=params, timeout=timeout)
            # Retry on 429 and 5xx
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                if attempt >= max_retries:
                    resp.raise_for_status()
                    return resp
                sleep_for = (backoff_factor * (2 ** attempt)) + (0.05 * attempt)
                print(f"[Retryable] {resp.status_code} on {url} | sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)
                attempt += 1
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                print(f"[Error] {method} {url} params={_redact_query_params(params or {})} -> {e}")
                raise
            sleep_for = (backoff_factor * (2 ** attempt)) + (0.05 * attempt)
            print(f"[Network Err] retry {attempt+1}/{max_retries} in {sleep_for:.2f}s: {e}")
            time.sleep(sleep_for)
            attempt += 1


def _make_graph_api_call(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET helper with retry & JSON decode."""
    resp = _request_with_retries("GET", url, params=params)
    return resp.json()


def _prepare_params(base_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adds optional parameters to a dictionary if they are not None. Handles JSON encoding & CSV joins."""
    params = base_params.copy()
    for key, value in kwargs.items():
        if value is None:
            continue

        # JSON-encoded params
        if key in [
            'filtering', 'time_range', 'time_ranges', 'effective_status',
            'special_ad_categories', 'objective', 'buyer_guarantee_agreement_status'
        ] and isinstance(value, (list, dict)):
            params[key] = json.dumps(value)

        # CSV joins for lists
        elif key in ['fields', 'action_attribution_windows', 'action_breakdowns', 'breakdowns'] and isinstance(value, list):
            params[key] = ','.join(value)

        else:
            params[key] = value
    return params


def _fetch_node(node_id: str, **kwargs) -> Dict[str, Any]:
    """Fetch a single object (node) by its ID."""
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{node_id}"
    params = _prepare_params({'access_token': access_token}, **kwargs)
    return _make_graph_api_call(url, params)


def _fetch_edge(parent_id: str, edge_name: str, **kwargs) -> Dict[str, Any]:
    """Fetch a collection (edge) for a parent object. Supports time params for 'activities'."""
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{parent_id}/{edge_name}"

    # Handle activities time params (special rule precedence)
    time_params: Dict[str, Any] = {}
    if edge_name == 'activities':
        time_range = kwargs.pop('time_range', None)
        since = kwargs.pop('since', None)
        until = kwargs.pop('until', None)
        if time_range:
            time_params['time_range'] = time_range
        else:
            if since: time_params['since'] = since
            if until: time_params['until'] = until

    base_params = {'access_token': access_token}
    params = _prepare_params(base_params, **kwargs)
    params.update(_prepare_params({}, **time_params))
    return _make_graph_api_call(url, params)


def _build_insights_params(
    params: Dict[str, Any],
    fields: Optional[List[str]] = None,
    date_preset: Optional[str] = None,
    time_range: Optional[Dict[str, str]] = None,
    time_ranges: Optional[List[Dict[str, str]]] = None,
    time_increment: Optional[str] = None,
    level: Optional[str] = None,
    action_attribution_windows: Optional[List[str]] = None,
    action_breakdowns: Optional[List[str]] = None,
    action_report_time: Optional[str] = None,
    breakdowns: Optional[List[str]] = None,
    default_summary: bool = False,
    use_account_attribution_setting: bool = False,
    use_unified_attribution_setting: bool = True,
    filtering: Optional[List[dict]] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    offset: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    locale: Optional[str] = None
) -> Dict[str, Any]:
    """Build common params for insights endpoints."""
    params = _prepare_params(
        params,
        fields=fields,
        level=level,
        action_attribution_windows=action_attribution_windows,
        action_breakdowns=action_breakdowns,
        action_report_time=action_report_time,
        breakdowns=breakdowns,
        filtering=filtering,
        sort=sort,
        limit=limit,
        after=after,
        before=before,
        offset=offset,
        locale=locale
    )

    # Time logic
    time_params_provided = time_range or time_ranges or since or until
    if not time_params_provided and date_preset:
        params['date_preset'] = date_preset
    if time_range:
        params['time_range'] = json.dumps(time_range)
    if time_ranges:
        params['time_ranges'] = json.dumps(time_ranges)
    if time_increment and time_increment != 'all_days':
        params['time_increment'] = time_increment

    # Only use since/until if not using time_range(s)
    if not time_range and not time_ranges:
        if since:
            params['since'] = since
        if until:
            params['until'] = until

    # Boolean flags (Graph often accepts bools; strings also historically accepted)
    if default_summary:
        params['default_summary'] = True
    if use_account_attribution_setting:
        params['use_account_attribution_setting'] = True
    if use_unified_attribution_setting:
        params['use_unified_attribution_setting'] = True

    return params


def _accumulate_pages(initial: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accumulate 'data' across pages using 'paging.next' URLs.
    Returns a consolidated list of rows.
    """
    data = list(initial.get("data", []))
    next_url = initial.get("paging", {}).get("next")
    while next_url:
        page = fetch_pagination_url(next_url)
        data.extend(page.get("data", []))
        next_url = page.get("paging", {}).get("next")
    return data


# --- Pagination Tool ---

@mcp.tool()
def fetch_pagination_url(url: str) -> Dict[str, Any]:
    """
    Fetch data from a Facebook Graph API pagination URL.
    """
    resp = _request_with_retries("GET", url)
    return resp.json()


# --- Account Tools ---

@mcp.tool()
def list_ad_accounts() -> Dict[str, Any]:
    """
    List ad accounts and names for the current user.
    MUST auto-paginate the 'adaccounts' edge until exhausted.
    """
    access_token = _get_fb_access_token()
    # We request the adaccounts edge nested under /me
    url = f"{FB_GRAPH_URL}/me"
    params = {
        'access_token': access_token,
        'fields': 'adaccounts{name}'
    }
    root = _make_graph_api_call(url, params)
    adacc = root.get("adaccounts", {})
    data = adacc.get("data", [])
    next_url = adacc.get("paging", {}).get("next")
    while next_url:
        page = fetch_pagination_url(next_url)
        data.extend(page.get("data", []))
        next_url = page.get("paging", {}).get("next")
    return {"data": data}


@mcp.tool()
def get_details_of_ad_account(act_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get details of a specific ad account.
    """
    effective_fields = fields if fields is not None else DEFAULT_AD_ACCOUNT_FIELDS
    return _fetch_node(node_id=act_id, fields=effective_fields)


# --- Insights Tools (auto-paginate ALL) ---

def _insights_auto_paginate(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    first = _make_graph_api_call(url, params)
    combined = _accumulate_pages(first)
    return {"data": combined}

@mcp.tool()
def get_adaccount_insights(
    act_id: str,
    fields: Optional[List[str]] = None,
    date_preset: str = 'last_30d',
    time_range: Optional[Dict[str, str]] = None,
    time_ranges: Optional[List[Dict[str, str]]] = None,
    time_increment: str = 'all_days',
    level: str = 'account',
    action_attribution_windows: Optional[List[str]] = None,
    action_breakdowns: Optional[List[str]] = None,
    action_report_time: Optional[str] = None,
    breakdowns: Optional[List[str]] = None,
    default_summary: bool = False,
    use_account_attribution_setting: bool = False,
    use_unified_attribution_setting: bool = True,
    filtering: Optional[List[dict]] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    offset: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    locale: Optional[str] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{act_id}/insights"
    params = _build_insights_params(
        params={'access_token': access_token},
        fields=fields,
        date_preset=date_preset,
        time_range=time_range,
        time_ranges=time_ranges,
        time_increment=time_increment,
        level=level,
        action_attribution_windows=action_attribution_windows,
        action_breakdowns=action_breakdowns,
        action_report_time=action_report_time,
        breakdowns=breakdowns,
        default_summary=default_summary,
        use_account_attribution_setting=use_account_attribution_setting,
        use_unified_attribution_setting=use_unified_attribution_setting,
        filtering=filtering,
        sort=sort,
        limit=limit,
        after=after,
        before=before,
        offset=offset,
        since=since,
        until=until,
        locale=locale
    )
    return _insights_auto_paginate(url, params)


@mcp.tool()
def get_campaign_insights(
    campaign_id: str,
    fields: Optional[List[str]] = None,
    date_preset: str = 'last_30d',
    time_range: Optional[Dict[str, str]] = None,
    time_ranges: Optional[List[Dict[str, str]]] = None,
    time_increment: str = 'all_days',
    action_attribution_windows: Optional[List[str]] = None,
    action_breakdowns: Optional[List[str]] = None,
    action_report_time: Optional[str] = None,
    breakdowns: Optional[List[str]] = None,
    default_summary: bool = False,
    use_account_attribution_setting: bool = False,
    use_unified_attribution_setting: bool = True,
    level: Optional[str] = None,
    filtering: Optional[List[dict]] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    offset: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    locale: Optional[str] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{campaign_id}/insights"
    effective_level = level if level else 'campaign'
    params = _build_insights_params(
        params={'access_token': access_token},
        fields=fields,
        date_preset=date_preset,
        time_range=time_range,
        time_ranges=time_ranges,
        time_increment=time_increment,
        level=effective_level,
        action_attribution_windows=action_attribution_windows,
        action_breakdowns=action_breakdowns,
        action_report_time=action_report_time,
        breakdowns=breakdowns,
        default_summary=default_summary,
        use_account_attribution_setting=use_account_attribution_setting,
        use_unified_attribution_setting=use_unified_attribution_setting,
        filtering=filtering,
        sort=sort,
        limit=limit,
        after=after,
        before=before,
        offset=offset,
        since=since,
        until=until,
        locale=locale
    )
    return _insights_auto_paginate(url, params)


@mcp.tool()
def get_adset_insights(
    adset_id: str,
    fields: Optional[List[str]] = None,
    date_preset: str = 'last_30d',
    time_range: Optional[Dict[str, str]] = None,
    time_ranges: Optional[List[Dict[str, str]]] = None,
    time_increment: str = 'all_days',
    action_attribution_windows: Optional[List[str]] = None,
    action_breakdowns: Optional[List[str]] = None,
    action_report_time: Optional[str] = None,
    breakdowns: Optional[List[str]] = None,
    default_summary: bool = False,
    use_account_attribution_setting: bool = False,
    use_unified_attribution_setting: bool = True,
    level: Optional[str] = None,
    filtering: Optional[List[dict]] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    offset: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    locale: Optional[str] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{adset_id}/insights"
    effective_level = level if level else 'adset'
    params = _build_insights_params(
        params={'access_token': access_token},
        fields=fields,
        date_preset=date_preset,
        time_range=time_range,
        time_ranges=time_ranges,
        time_increment=time_increment,
        level=effective_level,
        action_attribution_windows=action_attribution_windows,
        action_breakdowns=action_breakdowns,
        action_report_time=action_report_time,
        breakdowns=breakdowns,
        default_summary=default_summary,
        use_account_attribution_setting=use_account_attribution_setting,
        use_unified_attribution_setting=use_unified_attribution_setting,
        filtering=filtering,
        sort=sort,
        limit=limit,
        after=after,
        before=before,
        offset=offset,
        since=since,
        until=until,
        locale=locale
    )
    return _insights_auto_paginate(url, params)


@mcp.tool()
def get_ad_insights(
    ad_id: str,
    fields: Optional[List[str]] = None,
    date_preset: str = 'last_30d',
    time_range: Optional[Dict[str, str]] = None,
    time_ranges: Optional[List[Dict[str, str]]] = None,
    time_increment: str = 'all_days',
    action_attribution_windows: Optional[List[str]] = None,
    action_breakdowns: Optional[List[str]] = None,
    action_report_time: Optional[str] = None,
    breakdowns: Optional[List[str]] = None,
    default_summary: bool = False,
    use_account_attribution_setting: bool = False,
    use_unified_attribution_setting: bool = True,
    level: Optional[str] = None,
    filtering: Optional[List[dict]] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    offset: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    locale: Optional[str] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{ad_id}/insights"
    effective_level = level if level else 'ad'
    params = _build_insights_params(
        params={'access_token': access_token},
        fields=fields,
        date_preset=date_preset,
        time_range=time_range,
        time_ranges=time_ranges,
        time_increment=time_increment,
        level=effective_level,
        action_attribution_windows=action_attribution_windows,
        action_breakdowns=action_breakdowns,
        action_report_time=action_report_time,
        breakdowns=breakdowns,
        default_summary=default_summary,
        use_account_attribution_setting=use_account_attribution_setting,
        use_unified_attribution_setting=use_unified_attribution_setting,
        filtering=filtering,
        sort=sort,
        limit=limit,
        after=after,
        before=before,
        offset=offset,
        since=since,
        until=until,
        locale=locale
    )
    return _insights_auto_paginate(url, params)


# --- Ad Creative Tools ---

@mcp.tool()
def get_ad_creative_by_id(
    creative_id: str,
    fields: Optional[List[str]] = None,
    thumbnail_width: Optional[int] = None,
    thumbnail_height: Optional[int] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{creative_id}"
    params = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if thumbnail_width:
        params['thumbnail_width'] = thumbnail_width
    if thumbnail_height:
        params['thumbnail_height'] = thumbnail_height
    return _make_graph_api_call(url, params)


@mcp.tool()
def get_ad_creatives_by_ad_id(
    ad_id: str,
    fields: Optional[List[str]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    date_format: Optional[str] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{ad_id}/adcreatives"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if date_format:
        params['date_format'] = date_format

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


# --- Ad Tools ---

@mcp.tool()
def get_ad_by_id(ad_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{ad_id}"
    params = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    return _make_graph_api_call(url, params)


@mcp.tool()
def get_ads_by_adaccount(
    act_id: str,
    fields: Optional[List[str]] = None,
    filtering: Optional[List[dict]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    date_preset: Optional[str] = None,
    time_range: Optional[Dict[str, str]] = None,
    updated_since: Optional[int] = None,
    effective_status: Optional[List[str]] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{act_id}/ads"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if filtering:
        params['filtering'] = json.dumps(filtering)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if date_preset:
        params['date_preset'] = date_preset
    if time_range:
        params['time_range'] = json.dumps(time_range)
    if updated_since:
        params['updated_since'] = updated_since
    if effective_status:
        params['effective_status'] = json.dumps(effective_status)

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


@mcp.tool()
def get_ads_by_campaign(
    campaign_id: str,
    fields: Optional[List[str]] = None,
    filtering: Optional[List[dict]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    effective_status: Optional[List[str]] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{campaign_id}/ads"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if filtering:
        params['filtering'] = json.dumps(filtering)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if effective_status:
        params['effective_status'] = json.dumps(effective_status)

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


@mcp.tool()
def get_ads_by_adset(
    adset_id: str,
    fields: Optional[List[str]] = None,
    filtering: Optional[List[dict]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    effective_status: Optional[List[str]] = None,
    date_format: Optional[str] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{adset_id}/ads"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if filtering:
        params['filtering'] = json.dumps(filtering)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if effective_status:
        params['effective_status'] = json.dumps(effective_status)
    if date_format:
        params['date_format'] = date_format

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


# --- Ad Set Tools ---

@mcp.tool()
def get_adset_by_id(adset_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{adset_id}"
    params = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    return _make_graph_api_call(url, params)


@mcp.tool()
def get_adsets_by_ids(
    adset_ids: List[str],
    fields: Optional[List[str]] = None,
    date_format: Optional[str] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/"
    params: Dict[str, Any] = {
        'access_token': access_token,
        'ids': ','.join(adset_ids)
    }
    if fields:
        params['fields'] = ','.join(fields)
    if date_format:
        params['date_format'] = date_format
    return _make_graph_api_call(url, params)


@mcp.tool()
def get_adsets_by_adaccount(
    act_id: str,
    fields: Optional[List[str]] = None,
    filtering: Optional[List[dict]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    date_preset: Optional[str] = None,
    time_range: Optional[Dict[str, str]] = None,
    updated_since: Optional[int] = None,
    effective_status: Optional[List[str]] = None,
    date_format: Optional[str] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{act_id}/adsets"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if filtering:
        params['filtering'] = json.dumps(filtering)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if date_preset:
        params['date_preset'] = date_preset
    if time_range:
        params['time_range'] = json.dumps(time_range)
    if updated_since:
        params['updated_since'] = updated_since
    if effective_status:
        params['effective_status'] = json.dumps(effective_status)
    if date_format:
        params['date_format'] = date_format

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


@mcp.tool()
def get_adsets_by_campaign(
    campaign_id: str,
    fields: Optional[List[str]] = None,
    filtering: Optional[List[dict]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    effective_status: Optional[List[str]] = None,
    date_format: Optional[str] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{campaign_id}/adsets"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if filtering:
        params['filtering'] = json.dumps(filtering)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if effective_status:
        params['effective_status'] = json.dumps(effective_status)
    if date_format:
        params['date_format'] = date_format

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


# --- Campaign Tools ---

@mcp.tool()
def get_campaign_by_id(
    campaign_id: str,
    fields: Optional[List[str]] = None,
    date_format: Optional[str] = None
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{campaign_id}"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if date_format:
        params['date_format'] = date_format
    return _make_graph_api_call(url, params)


@mcp.tool()
def get_campaigns_by_adaccount(
    act_id: str,
    fields: Optional[List[str]] = None,
    filtering: Optional[List[dict]] = None,
    limit: Optional[int] = 25,
    after: Optional[str] = None,
    before: Optional[str] = None,
    date_preset: Optional[str] = None,
    time_range: Optional[Dict[str, str]] = None,
    updated_since: Optional[int] = None,
    effective_status: Optional[List[str]] = None,
    is_completed: Optional[bool] = None,
    special_ad_categories: Optional[List[str]] = None,
    objective: Optional[List[str]] = None,
    date_format: Optional[str] = None,
    include_drafts: Optional[bool] = None,
    auto_paginate: bool = True
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{act_id}/campaigns"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if filtering:
        params['filtering'] = json.dumps(filtering)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if date_preset:
        params['date_preset'] = date_preset
    if time_range:
        params['time_range'] = json.dumps(time_range)
    if updated_since:
        params['updated_since'] = updated_since
    if effective_status:
        params['effective_status'] = json.dumps(effective_status)
    if is_completed is not None:
        params['is_completed'] = is_completed
    if special_ad_categories:
        params['special_ad_categories'] = json.dumps(special_ad_categories)
    if objective:
        params['objective'] = json.dumps(objective)
    if date_format:
        params['date_format'] = date_format
    if include_drafts is not None:
        params['include_drafts'] = include_drafts

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


# --- Activity Tools ---

@mcp.tool()
def get_activities_by_adaccount(
    act_id: str,
    fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    time_range: Optional[Dict[str, str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    auto_paginate: bool = False
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{act_id}/activities"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if time_range:
        params['time_range'] = json.dumps(time_range)
    else:
        if since:
            params['since'] = since
        if until:
            params['until'] = until

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


@mcp.tool()
def get_activities_by_adset(
    adset_id: str,
    fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    time_range: Optional[Dict[str, str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    auto_paginate: bool = False
) -> Dict[str, Any]:
    access_token = _get_fb_access_token()
    url = f"{FB_GRAPH_URL}/{adset_id}/activities"
    params: Dict[str, Any] = {'access_token': access_token}
    if fields:
        params['fields'] = ','.join(fields)
    if limit is not None:
        params['limit'] = limit
    if after:
        params['after'] = after
    if before:
        params['before'] = before
    if time_range:
        params['time_range'] = json.dumps(time_range)
    else:
        if since:
            params['since'] = since
        if until:
            params['until'] = until

    first = _make_graph_api_call(url, params)
    if not auto_paginate:
        return first
    return {"data": _accumulate_pages(first)}


# --- Main (serve over HTTP + SSE) ---

if __name__ == "__main__":
    _get_fb_access_token()
    # Serve as an HTTP SSE server so Claude/other hosts can connect via URL
    mcp.run(
        transport="sse",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
        path=os.getenv("MCP_SSE_PATH", "/sse")
    )
