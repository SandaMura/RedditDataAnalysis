# Reddit Resurrections (2008 – 2013)

**Idea in one sentence**

> _How do clever title rewrites and community hops keep the **same image** alive on Reddit for years?_

## Dataset

| File                | Rows              | Description                                                                 |
| ------------------- | ----------------- | --------------------------------------------------------------------------- |
| `redditSubmissions` |  ~132 k           | 2008‑2013 reposts dump from SNAP – one row per submission                   |
| `redditHtmlData`    |  132 k HTML files | full markup for every submission (title, score, comments, sidebar metadata) |

## Why it’s interesting

- The SNAP set tracks **the same `image_id` across all reposts**, letting us measure lifecycles.
- By scraping comment trees & sidebar data we can enrich the plain CSV with _conversation depth_ and _subreddit age_ – aspects untouched in earlier studies (e.g., Lakkaraju et al., ICWSM 2013).

## Planned analyses (3‑D lens)

| Track           | Possible Questions                                                                         |
| --------------- | ------------------------------------------------------------------------------------------ |
| **1. Static**   | How many times is a typical image reposted? Does a bigger title change = higher new score? |
| **2. Network**  | Which subreddits act as “revival hubs” that re‑export old images to the rest of Reddit?    |
| **3. Temporal** | Did the half‑life of viral images shrink from 2008 to 2013 as audience attention sped up?  |

_No hypotheses baked in yet – we’ll let the data tell us whether title rewrites, community hops or time windows drive resurgence._
