import argparse
import datetime as dt
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def safe_int(txt):
    """Return the first digit string in txt as int, or None."""
    m = re.search(r"\d[\d,]*", txt or "")
    return int(m.group(0).replace(",", "")) if m else None


def get_comment_depth(div):
    depth = 1
    parent = div.find_parent("div", class_="comment")
    while parent:
        depth += 1
        parent = parent.find_parent("div", class_="comment")
    return depth


def parse_html_file(fp_str: str) -> dict:
    fp = Path(fp_str)
    try:
        soup = BeautifulSoup(Path(fp_str).read_bytes(), "lxml")
    except Exception as e:
        return {"reddit_id": fp.stem, "parse_error": str(e)}

    title_tag = soup.select_one("a.title")
    score_tag = soup.select_one("div.score.unvoted")
    domain_tag = soup.select_one("span.domain")
    flair_tag = soup.select_one("span.linkflairlabel")
    comm_link = soup.select_one("a.comments")
    comments = soup.select("div.comment")

    sidebox = soup.select_one("div.titlebox, div.side")

    subreddit_created = soup.select_one(
        "div.titlebox time, div.side time, span.age time"
    )
    subs_created_at = subreddit_created["datetime"] if subreddit_created else None

    subscriber_count = (
        safe_int(sidebox.select_one(".subscribers").get_text(strip=True))
        if sidebox and sidebox.select_one(".subscribers")
        else None
    )

    active_users = (
        safe_int(sidebox.select_one(".users-online").get_text(strip=True))
        if sidebox and sidebox.select_one(".users-online")
        else None
    )

    sidebar_len = len(sidebox.get_text(" ", strip=True).split()) if sidebox else None

    nsfw_flag = bool(
        sidebox and re.search(r"\bnsfw\b", sidebox.get_text(" ", strip=True), re.I)
    )

    num_mods = len(soup.select("span.author.moderator"))

    comment_scores = [
        safe_int(c.select_one("span.score.unvoted").get_text(strip=True))
        for c in comments
        if c.select_one("span.score.unvoted")
    ]
    top_comment_score = max(comment_scores) if comment_scores else None
    comment_depth_mean = (
        sum(get_comment_depth(c) for c in comments) / len(comments)
        if comments
        else None
    )

    first_comment_tag = soup.select_one("div.comment time")
    try:
        first_comm_ts = (
            dt.datetime.fromisoformat(
                first_comment_tag["datetime"].replace("Z", "+00:00")
            )
            if first_comment_tag and first_comment_tag.has_attr("datetime")
            else None
        )
    except Exception:
        first_comm_ts = None

    return {
        "reddit_id": fp.stem,
        "title_html": title_tag.get_text(" ", strip=True) if title_tag else None,
        "score_html": int(score_tag["title"])
        if score_tag and score_tag.has_attr("title")
        else None,
        "num_comments_html": safe_int(comm_link.get_text()) if comm_link else None,
        "domain_html": domain_tag.get_text(strip=True).strip("()")
        if domain_tag
        else None,
        "flair": flair_tag.get_text(strip=True) if flair_tag else None,
        "comment_count": len(comments) or None,
        "comment_depth_max": max(
            (get_comment_depth(c) for c in comments), default=None
        ),
        "subreddit_created_at": subs_created_at,
        "subscriber_count": subscriber_count,
        "active_users": active_users,
        "sidebar_len": sidebar_len,
        "nsfw_flag": nsfw_flag,
        "num_mods": num_mods,
        "top_comment_score": top_comment_score,
        "comment_depth_mean": comment_depth_mean,
        "first_comment_delay_sec": (
            (
                first_comm_ts
                - dt.datetime.fromtimestamp(int(fp.stem, 36), tz=dt.timezone.utc)
            ).total_seconds()
            if first_comm_ts
            else None
        ),
    }


# ---------- main -------------------------------------------------------------
def main(html_dir: Path):
    html_paths = [str(p) for p in sorted(html_dir.glob("*.html"))]
    n = len(html_paths)
    print(f"Found {n:,} HTML files in {html_dir}")
    pool_size = cpu_count()
    print(f"Launching Pool  {pool_size} processes…")

    # imap_unordered parses and yields as soon as each file is done
    with Pool(pool_size) as pool:
        results = []
        for res in tqdm(
            pool.imap_unordered(parse_html_file, html_paths),
            total=n,
            desc="Parsing HTML",
        ):
            results.append(res)

    df_html = pd.DataFrame(results)
    out_path = (
        html_dir.parent.parent / "processed" / "html_parsed_lookup.parquet"
    ).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving to:", out_path)
    df_html.to_parquet(out_path, compression="zstd")
    print("Done! Rows written:", len(df_html))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_dir", type=Path, default=Path("../data/redditHtmlData"))
    args = parser.parse_args()
    main(args.html_dir)
