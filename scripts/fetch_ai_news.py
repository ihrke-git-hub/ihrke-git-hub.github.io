#!/usr/bin/env python3
"""AI Topics: ニュース収集・カテゴリ分類・HTML生成スクリプト"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import feedparser
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = BASE_DIR / "data" / "sources.json"
ARTICLES_DIR = BASE_DIR / "data" / "articles"
OUTPUT_PATH = BASE_DIR / "ai-topics" / "index.html"

JST = timezone(timedelta(hours=9))
TODAY = datetime.now(JST).strftime("%Y-%m-%d")
KEEP_DAYS = 7

CATEGORIES = [
    "LLM・チャットAI",
    "画像・動画生成",
    "音声・音楽AI",
    "ロボティクス・自動運転",
    "AI規制・政策",
    "AI倫理・安全性",
    "研究・論文",
    "AI製品・ツール",
    "企業・資金調達",
    "医療・科学応用",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AITopicsBot/1.0; +https://ihrke-git-hub.github.io/ai-topics/)"
}


# ── 1. ソース読み込み ──

def load_sources():
    with open(SOURCES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 2. RSS取得 ──

def fetch_rss(source):
    """RSSフィードから記事を取得する"""
    articles = []
    try:
        resp = requests.get(source["url"], headers=HEADERS, timeout=20)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        for entry in feed.entries[:30]:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = time.strftime("%Y-%m-%d", entry.published_parsed)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = time.strftime("%Y-%m-%d", entry.updated_parsed)

            if not published:
                published = TODAY

            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            if not title or not link:
                continue

            articles.append({
                "title": title,
                "url": link,
                "source": source["name"],
                "lang": source["lang"],
                "date": published,
            })
    except Exception as e:
        print(f"  [WARN] {source['name']}: {e}")
    return articles


# ── 3. 記事収集 ──

def collect_articles(sources):
    """全ソースから記事を収集し、当日分を抽出・重複排除する"""
    all_articles = []
    for source in sources:
        print(f"  取得中: {source['name']}...")
        articles = fetch_rss(source)
        print(f"    → {len(articles)}件")
        all_articles.extend(articles)

    # 当日の記事のみ抽出（直近2日分を許容。時差考慮）
    yesterday = (datetime.now(JST) - timedelta(days=1)).strftime("%Y-%m-%d")
    recent = [a for a in all_articles if a["date"] in (TODAY, yesterday)]

    # 当日分が少なすぎる場合は直近3日に広げる
    if len(recent) < 5:
        three_days_ago = (datetime.now(JST) - timedelta(days=3)).strftime("%Y-%m-%d")
        cutoff = datetime.strptime(three_days_ago, "%Y-%m-%d")
        recent = [a for a in all_articles if datetime.strptime(a["date"], "%Y-%m-%d") >= cutoff]

    # URL重複排除
    seen_urls = set()
    unique = []
    for a in recent:
        if a["url"] not in seen_urls:
            seen_urls.add(a["url"])
            unique.append(a)

    return unique


# ── 4. Claude APIでカテゴリ分類 ──

def classify_articles(articles):
    """Claude APIで記事をカテゴリ分類する"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [WARN] ANTHROPIC_API_KEY未設定。カテゴリは「AI製品・ツール」で統一します。")
        for a in articles:
            a["category"] = "AI製品・ツール"
        return articles

    client = anthropic.Anthropic(api_key=api_key)
    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)

    # バッチ処理（20件ずつ）
    batch_size = 20
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        articles_list = "\n".join(
            f'{idx+1}. [{a["lang"].upper()}] {a["title"]}'
            for idx, a in enumerate(batch)
        )

        prompt = f"""以下の記事タイトルをそれぞれ1つのカテゴリに分類してください。

## カテゴリ一覧
{categories_str}

## 記事一覧
{articles_list}

## 出力形式
JSON配列で、各要素は記事番号（1始まり）に対応するカテゴリ名の文字列です。
カテゴリ名は上記一覧と完全一致させてください。
JSON配列のみを出力し、他のテキストは含めないでください。

出力例: ["LLM・チャットAI", "画像・動画生成", ...]"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # JSON部分を抽出
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                cats = json.loads(match.group())
                for idx, a in enumerate(batch):
                    if idx < len(cats) and cats[idx] in CATEGORIES:
                        a["category"] = cats[idx]
                    else:
                        a["category"] = "AI製品・ツール"
            else:
                for a in batch:
                    a["category"] = "AI製品・ツール"
        except Exception as e:
            print(f"  [WARN] Claude API エラー: {e}")
            for a in batch:
                a["category"] = "AI製品・ツール"

    return articles


# ── 5. 記事の選定（10-15件） ──

def select_top_articles(articles, max_count=15):
    """カテゴリのバランスを考慮して10-15件を選定する"""
    if len(articles) <= max_count:
        return articles

    # カテゴリごとにグループ化
    by_cat = {}
    for a in articles:
        cat = a.get("category", "AI製品・ツール")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(a)

    # ラウンドロビンで各カテゴリから均等に選出
    selected = []
    cat_keys = list(by_cat.keys())
    idx_per_cat = {k: 0 for k in cat_keys}

    while len(selected) < max_count:
        added = False
        for cat in cat_keys:
            if idx_per_cat[cat] < len(by_cat[cat]) and len(selected) < max_count:
                selected.append(by_cat[cat][idx_per_cat[cat]])
                idx_per_cat[cat] += 1
                added = True
        if not added:
            break

    return selected


# ── 6. データ保存・読み込み ──

def save_articles(articles, date_str):
    """当日分の記事をJSONに保存する"""
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTICLES_DIR / f"{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"  保存: {path} ({len(articles)}件)")


def load_recent_articles():
    """直近7日分の記事を読み込む"""
    all_articles = {}
    cutoff = datetime.now(JST) - timedelta(days=KEEP_DAYS)

    for path in sorted(ARTICLES_DIR.glob("*.json")):
        date_str = path.stem
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except ValueError:
            continue

        if file_date < cutoff:
            # 古いファイルを削除
            path.unlink()
            print(f"  削除（古い）: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            articles = json.load(f)
        all_articles[date_str] = articles

    return all_articles


# ── 7. HTML生成 ──

CATEGORY_COLORS = {
    "LLM・チャットAI": "#8B5CF6",
    "画像・動画生成": "#EC4899",
    "音声・音楽AI": "#F59E0B",
    "ロボティクス・自動運転": "#10B981",
    "AI規制・政策": "#EF4444",
    "AI倫理・安全性": "#F97316",
    "研究・論文": "#3B82F6",
    "AI製品・ツール": "#06B6D4",
    "企業・資金調達": "#84CC16",
    "医療・科学応用": "#14B8A6",
}


def generate_html(all_articles, updated_at):
    """ポータルサイトのHTMLを生成する"""
    dates = sorted(all_articles.keys(), reverse=True)

    # 日付タブHTML
    date_tabs = ""
    for i, d in enumerate(dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        label = f'{dt.month}/{dt.day}({weekdays[dt.weekday()]})'
        active = " active" if i == 0 else ""
        date_tabs += f'<button class="tab{active}" onclick="showDate(\'{d}\')">{label}</button>\n'

    # カテゴリフィルタHTML
    cat_filters = '<button class="cat-btn active" onclick="filterCat(\'all\')">すべて</button>\n'
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#6B7280")
        cat_filters += f'<button class="cat-btn" onclick="filterCat(\'{cat}\')" style="--cat-color:{color}">{cat}</button>\n'

    # 日付ごとの記事セクション
    date_sections = ""
    for i, d in enumerate(dates):
        display = "block" if i == 0 else "none"
        articles = all_articles[d]
        cards = ""
        for a in articles:
            cat = a.get("category", "AI製品・ツール")
            color = CATEGORY_COLORS.get(cat, "#6B7280")
            lang_badge = f'<span class="lang-badge lang-{a.get("lang", "en")}">{a.get("lang", "en").upper()}</span>'
            cards += f'''<a href="{a["url"]}" target="_blank" rel="noopener" class="card" data-category="{cat}">
                <div class="card-header">
                    <span class="cat-tag" style="background:{color}">{cat}</span>
                    {lang_badge}
                </div>
                <h3 class="card-title">{a["title"]}</h3>
                <div class="card-meta">{a["source"]}</div>
            </a>
'''
        if not cards:
            cards = '<p class="no-articles">この日の記事はありません</p>'
        date_sections += f'<div class="date-section" id="date-{d}" style="display:{display}">\n{cards}</div>\n'

    # JSON data for JS
    json_data = json.dumps(all_articles, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Topics - 最新AIニュース</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial,
                         "Hiragino Sans", "Hiragino Kaku Gothic ProN", Meiryo, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 20px 16px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 1.8rem;
            color: #f8fafc;
            margin-bottom: 4px;
            letter-spacing: 0.02em;
        }}
        .header h1 span {{
            background: linear-gradient(135deg, #8B5CF6, #3B82F6, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .header .subtitle {{
            font-size: 0.85rem;
            color: #94a3b8;
        }}
        .header .updated {{
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 4px;
        }}

        /* Date tabs */
        .date-tabs {{
            display: flex;
            gap: 6px;
            margin-bottom: 16px;
            overflow-x: auto;
            padding-bottom: 4px;
            -webkit-overflow-scrolling: touch;
        }}
        .tab {{
            padding: 8px 14px;
            border: 1px solid #334155;
            background: #1e293b;
            color: #94a3b8;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            white-space: nowrap;
            transition: all 0.2s;
        }}
        .tab:hover {{ background: #334155; color: #e2e8f0; }}
        .tab.active {{
            background: #3b82f6;
            border-color: #3b82f6;
            color: #fff;
        }}

        /* Category filters */
        .cat-filters {{
            display: flex;
            gap: 6px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .cat-btn {{
            padding: 5px 12px;
            border: 1px solid #334155;
            background: #1e293b;
            color: #94a3b8;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
        }}
        .cat-btn:hover {{
            border-color: var(--cat-color, #3b82f6);
            color: var(--cat-color, #3b82f6);
        }}
        .cat-btn.active {{
            background: var(--cat-color, #3b82f6);
            border-color: var(--cat-color, #3b82f6);
            color: #fff;
        }}

        /* Article cards */
        .card {{
            display: block;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 10px;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
        }}
        .card:hover {{
            border-color: #475569;
            background: #253348;
            transform: translateY(-1px);
        }}
        .card-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .cat-tag {{
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 12px;
            color: #fff;
            font-weight: 600;
        }}
        .lang-badge {{
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
        }}
        .lang-en {{ background: #1d4ed8; color: #dbeafe; }}
        .lang-ja {{ background: #b91c1c; color: #fecaca; }}
        .card-title {{
            font-size: 0.95rem;
            line-height: 1.5;
            color: #f1f5f9;
            margin-bottom: 6px;
        }}
        .card-meta {{
            font-size: 0.75rem;
            color: #64748b;
        }}
        .no-articles {{
            text-align: center;
            color: #64748b;
            padding: 40px 0;
        }}

        /* Stats bar */
        .stats {{
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-bottom: 20px;
            font-size: 0.8rem;
            color: #64748b;
        }}
        .stats span {{
            color: #3b82f6;
            font-weight: 700;
        }}

        @media (max-width: 600px) {{
            .container {{ padding: 12px 10px; }}
            .header h1 {{ font-size: 1.4rem; }}
            .card {{ padding: 12px; }}
            .card-title {{ font-size: 0.85rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><span>AI Topics</span></h1>
            <div class="subtitle">最新AIニュース デイリーダイジェスト</div>
            <div class="updated">最終更新: {updated_at}</div>
        </div>

        <div class="date-tabs">
            {date_tabs}
        </div>

        <div class="cat-filters">
            {cat_filters}
        </div>

        <div id="content">
            {date_sections}
        </div>
    </div>

    <script>
    let currentDate = '{dates[0] if dates else TODAY}';
    let currentCat = 'all';

    function showDate(d) {{
        currentDate = d;
        document.querySelectorAll('.date-section').forEach(el => el.style.display = 'none');
        const target = document.getElementById('date-' + d);
        if (target) target.style.display = 'block';
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');
        applyFilter();
    }}

    function filterCat(cat) {{
        currentCat = cat;
        document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');
        applyFilter();
    }}

    function applyFilter() {{
        const section = document.getElementById('date-' + currentDate);
        if (!section) return;
        const cards = section.querySelectorAll('.card');
        let visible = 0;
        cards.forEach(card => {{
            if (currentCat === 'all' || card.dataset.category === currentCat) {{
                card.style.display = 'block';
                visible++;
            }} else {{
                card.style.display = 'none';
            }}
        }});
        // Show/hide no-articles message
        let noMsg = section.querySelector('.no-articles-filter');
        if (visible === 0) {{
            if (!noMsg) {{
                noMsg = document.createElement('p');
                noMsg.className = 'no-articles no-articles-filter';
                noMsg.textContent = 'このカテゴリの記事はありません';
                section.appendChild(noMsg);
            }}
            noMsg.style.display = 'block';
        }} else if (noMsg) {{
            noMsg.style.display = 'none';
        }}
    }}
    </script>
</body>
</html>"""
    return html


# ── メイン処理 ──

def main():
    print("=== AI Topics 更新開始 ===")

    print("\n[1/5] ソース読み込み...")
    sources = load_sources()
    print(f"  {len(sources)}サイト")

    print("\n[2/5] 記事収集...")
    articles = collect_articles(sources)
    print(f"  収集合計: {len(articles)}件")

    if not articles:
        print("  記事が0件のため終了します")
        return

    print("\n[3/5] カテゴリ分類...")
    articles = classify_articles(articles)

    print("\n[4/5] 記事選定...")
    articles = select_top_articles(articles, max_count=15)
    print(f"  選定: {len(articles)}件")

    # 保存
    save_articles(articles, TODAY)

    print("\n[5/5] HTML生成...")
    all_articles = load_recent_articles()
    total = sum(len(v) for v in all_articles.values())
    print(f"  表示対象: {len(all_articles)}日分 / {total}件")

    updated_at = datetime.now(JST).strftime("%Y年%m月%d日 %H:%M JST")
    html = generate_html(all_articles, updated_at)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  出力: {OUTPUT_PATH}")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
