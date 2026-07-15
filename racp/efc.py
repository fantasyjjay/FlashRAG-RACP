import hashlib
import re
from collections import Counter


SOURCE_RRF_WEIGHTS = {
    "original": 1.0,
    "qd": 1.05,
    "generation_guided": 1.15,
}
ROLE_NAMES = (
    "anchor",
    "bridge",
    "answer",
    "comparison_left",
    "comparison_right",
    "constraint",
)
ENTITY_STOP_WORDS = {
    "according",
    "answer",
    "based",
    "doc",
    "documents",
    "evidence",
    "final",
    "question",
    "so",
    "so the",
    "the",
    "they",
    "therefore",
    "wikipedia",
    "yes",
    "no",
    "american",
    "british",
    "english",
    "french",
    "german",
    "however",
    "there",
    "from",
    "key",
}
RELATION_TERMS = (
    "born",
    "directed",
    "written",
    "published",
    "founded",
    "located",
    "member",
    "spouse",
    "wife",
    "husband",
    "father",
    "mother",
    "worked with",
)
PROBE_UNCERTAINTY_MARKERS = (
    "no information",
    "not enough information",
    "does not mention",
    "do not mention",
    "doesn't mention",
    "unknown",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not provided",
    "not specified",
    "not available",
)
QUERY_FOCUS_PATTERNS = (
    (r"\bgovernment position\b", "government position"),
    (r"\bfight song\b", "fight song"),
    (r"\bbirth name\b", "birth name"),
    (r"\b(?:other )?occupation\b", "occupation"),
    (r"\bnationality\b", "nationality"),
    (r"\bformer name\b", "former name"),
    (r"\breal name\b", "real name"),
    (r"\bmaiden name\b", "maiden name"),
    (r"\bdate of birth\b", "date of birth"),
    (r"\bwhen was\b", "date year"),
    (r"\bwhere was\b", "location"),
    (r"\bseating capacity\b|\bcan seat\b", "seating capacity"),
    (r"\bhow many\b", "number"),
    (r"\bwhat album\b", "album"),
    (r"\bwhat university\b|\bwhich university\b", "university"),
    (r"\bwhat school\b|\bwhich school\b", "school"),
    (r"\bwhat country\b|\bwhich country\b", "country"),
    (r"\bwhat city\b|\bwhich city\b", "city"),
    (r"\bwhat year\b|\bwhich year\b", "year"),
    (r"\bwho directed\b|\bdirector\b", "director"),
    (r"\bwho wrote\b|\bauthor\b|\bwriter\b", "author"),
    (r"\bspouse\b|\bwife\b|\bhusband\b", "spouse"),
)


def doc_contents(doc):
    if isinstance(doc, str):
        return doc
    for key in ("contents", "text", "content"):
        value = doc.get(key)
        if value:
            return str(value)
    return ""


def doc_title(doc):
    if isinstance(doc, dict):
        title = str(doc.get("title") or "").strip().strip("\"'")
        if title:
            return title
    first_line = doc_contents(doc).partition("\n")[0].strip().strip("\"'")
    return first_line or doc_contents(doc)[:80]


def doc_uid(doc):
    if isinstance(doc, dict):
        value = doc.get("doc_uid") or doc.get("doc_id") or doc.get("id")
        if value is not None:
            return str(value)
    payload = f"{doc_title(doc)}\n{doc_contents(doc)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def normalize_doc(doc, score, rank, source, query, query_id):
    normalized = dict(doc) if isinstance(doc, dict) else {"contents": str(doc)}
    contents = doc_contents(normalized)
    title = doc_title(normalized)
    normalized.update(
        {
            "doc_uid": doc_uid(normalized),
            "doc_id": normalized.get("doc_id", normalized.get("id")),
            "title": title,
            "contents": contents,
            "text": str(normalized.get("text") or contents),
            "sources": [source],
            "source_queries": [query],
            "ranks": {query_id: int(rank)},
            "retriever_scores": {query_id: float(score)},
        }
    )
    return normalized


def merge_docs(doc_groups):
    pool = {}
    for docs in doc_groups:
        for doc in docs:
            uid = doc["doc_uid"]
            if uid not in pool:
                pool[uid] = dict(doc)
                pool[uid]["sources"] = list(doc.get("sources", []))
                pool[uid]["source_queries"] = list(doc.get("source_queries", []))
                pool[uid]["ranks"] = dict(doc.get("ranks", {}))
                pool[uid]["retriever_scores"] = dict(doc.get("retriever_scores", {}))
                continue
            current = pool[uid]
            current["sources"] = list(
                dict.fromkeys(current.get("sources", []) + doc.get("sources", []))
            )
            current["source_queries"] = list(
                dict.fromkeys(
                    current.get("source_queries", []) + doc.get("source_queries", [])
                )
            )
            for key, rank in doc.get("ranks", {}).items():
                current["ranks"][key] = min(rank, current["ranks"].get(key, rank))
            current["retriever_scores"].update(doc.get("retriever_scores", {}))
    return list(pool.values())


def classify_question(question, assume_multihop=False):
    lowered = " ".join(question.lower().split())
    comparison_markers = (
        " between ",
        "which came first",
        "who had more",
        "which is larger",
        "which is smaller",
        " earlier",
        " later",
        " more ",
        " less ",
        " same ",
        " or ",
    )
    bridge_markers = (
        "the director of",
        "the author of",
        "the wife of",
        "the husband of",
        "the father of",
        "the mother of",
        "where was",
        "where the",
        "arena where",
        "what is the",
        "who wrote",
        "who directed",
        "capital of",
        "spouse of",
    )
    constraint_markers = (
        "born in",
        "located in",
        "worked with",
        "directed by",
        "written by",
        "published by",
        "founded by",
        "member of",
    )
    if any(marker in f" {lowered} " for marker in comparison_markers):
        return "comparison"
    if any(marker in lowered for marker in bridge_markers):
        return "bridge"
    if any(marker in lowered for marker in constraint_markers):
        return "constraint"
    if assume_multihop:
        return "bridge"
    return "generic"


def infer_answer_type(question):
    lowered = question.lower().strip()
    if any(term in lowered for term in ("how many", "number", "capacity", "population")):
        return "number"
    if any(term in lowered for term in ("when", "what year", "which year", " date")):
        return "date"
    if lowered.startswith("who"):
        return "person"
    if lowered.startswith("where"):
        return "location"
    if lowered.startswith(("is ", "are ", "was ", "were ", "can ", "did ", "does ")):
        return "yesno"
    return "entity"


def answer_type_hit(text, answer_type):
    lowered = text.lower()
    if answer_type == "number":
        return bool(re.search(r"\b\d[\d,.]*\b", text)) or any(
            term in lowered for term in ("capacity", "seats", "population", "height", "score")
        )
    if answer_type == "date":
        return bool(re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text)) or any(
            term in lowered for term in ("born", "date", "founded", "released", "published")
        )
    if answer_type == "location":
        return any(
            term in lowered
            for term in ("born in", "located in", " city", " country", " state", " place")
        )
    if answer_type == "person":
        return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
    if answer_type == "yesno":
        return any(term in lowered for term in (" is ", " are ", " was ", " were "))
    return bool(text.strip())


def probe_answer_is_uncertain(answer):
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in PROBE_UNCERTAINTY_MARKERS)


def extract_probe_final_answer(answer):
    matches = re.findall(
        r"(?:so\s+the\s+answer\s+is|final\s+answer\s*:?)\s*"
        r"(yes|no|[^\n.]+)",
        answer or "",
        re.I,
    )
    return matches[-1].strip().lower() if matches else ""


def probe_answer_is_contradictory(question, answer):
    if infer_answer_type(question) != "yesno":
        return False
    lowered = (answer or "").lower()
    final_answer = extract_probe_final_answer(answer)
    if final_answer.startswith("no") and re.search(
        r"\bboth\b.{0,100}\b(?:same|american|british|english|french|german)\b",
        lowered,
        re.S,
    ):
        return True
    return final_answer.startswith("yes") and bool(
        re.search(r"\b(?:different|not the same|differ)\b", lowered)
    )


def probe_answer_is_bad(question, answer):
    answer = (answer or "").strip()
    lowered = answer.lower()
    if len(answer.split()) < 2:
        return True
    if any(
        marker in lowered
        for marker in ("i don't know", "i do not know", "cannot answer", "can't answer")
    ):
        return True
    if probe_answer_is_uncertain(answer):
        return True
    trailing_word = re.search(r"([a-z]+)[^a-z0-9]*$", lowered)
    if trailing_word and trailing_word.group(1) in {
        "a",
        "an",
        "and",
        "are",
        "is",
        "of",
        "the",
        "to",
        "was",
        "were",
    }:
        return True
    question_ratio = len(set(question.lower().split()) & set(lowered.split())) / max(
        1, len(set(lowered.split()))
    )
    if question_ratio > 0.9 and len(answer.split()) <= len(question.split()) + 2:
        return True

    return probe_answer_is_contradictory(question, answer)


def extract_new_entities(question, answer, r0_titles):
    question_lower = question.lower()
    candidates = []
    for title in r0_titles:
        title = title.strip()
        if len(title) > 2 and title.lower() in answer.lower() and title.lower() not in question_lower:
            candidates.append(title)
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"\b(?:[A-Z][A-Za-z0-9'&-]*)(?:\s+(?:[A-Z][A-Za-z0-9'&-]*|of|the|and|de|van)){0,5}",
            answer,
        )
    )
    entities = []
    seen = set()
    for entity in candidates:
        entity = entity.strip(" .,:;!?()[]\"'")
        normalized = entity.lower()
        if (
            len(entity) < 3
            or normalized in question_lower
            or normalized in ENTITY_STOP_WORDS
            or normalized in seen
            or entity.isdigit()
        ):
            continue
        seen.add(normalized)
        entities.append(entity)
    return entities


def compute_router_features(
    question,
    r0_docs,
    probe_answer,
    centroid_features=None,
    assume_multihop=False,
):
    top5 = r0_docs[:5]
    scores = [
        float(doc.get("retriever_scores", {}).get("original", 0.0)) for doc in r0_docs
    ]
    titles = [doc_title(doc) for doc in top5]
    question_type = classify_question(question, assume_multihop=assume_multihop)
    answer_type = infer_answer_type(question)
    new_entities = extract_new_entities(question, probe_answer, titles)
    comparison_left, comparison_right = extract_comparison_sides(question)
    top5_text = "\n".join(
        f"{doc_title(doc)}\n{doc_contents(doc)}".lower() for doc in top5
    )
    comparison_side_coverage = (
        int(bool(comparison_left) and comparison_left.lower() in top5_text)
        + int(bool(comparison_right) and comparison_right.lower() in top5_text)
    )
    centroid_features = centroid_features or {}
    return {
        "s1": scores[0] if scores else 0.0,
        "s5": scores[min(4, len(scores) - 1)] if scores else 0.0,
        "avg_top5_score": sum(scores[:5]) / max(1, min(5, len(scores))),
        "gap_1_5": scores[0] - scores[min(4, len(scores) - 1)] if scores else 0.0,
        "title_unique_ratio": len({title.lower() for title in titles}) / max(1, len(titles)),
        "q_centroid_sim": centroid_features.get("q_centroid_sim"),
        "top5_cohesion": centroid_features.get("top5_cohesion"),
        "centroid_shift": centroid_features.get("centroid_shift"),
        "question_is_multihop": question_type != "generic",
        "question_type": question_type,
        "comparison_side_coverage_top5": comparison_side_coverage,
        "y1_bad": probe_answer_is_bad(question, probe_answer),
        "y1_uncertain": probe_answer_is_uncertain(probe_answer),
        "y1_contradictory": probe_answer_is_contradictory(question, probe_answer),
        "y1_final_answer": extract_probe_final_answer(probe_answer),
        "y1_has_new_entity": bool(new_entities),
        "y1_new_entities": new_entities,
        "y1_new_entity_count": len(new_entities),
        "answer_type": answer_type,
        "answer_type_hit_top5": any(
            answer_type_hit(doc_contents(doc), answer_type) for doc in top5
        ),
    }


def decide_route(features, force_route="auto"):
    if force_route != "auto":
        return force_route
    if (
        features["question_type"] == "comparison"
        and features.get("comparison_side_coverage_top5", 0) >= 2
    ):
        return "direct"
    if features.get("y1_contradictory", False) and features["answer_type_hit_top5"]:
        return "direct"
    if not features["question_is_multihop"]:
        return "direct"
    # EFC follows the IterRetGen principle: when the probe exposes a bridge
    # entity, use that generation to drive the next retrieval hop. Static
    # decomposition is reserved for cases where the probe exposes no entity.
    if features["y1_has_new_entity"]:
        return "generation_guided"
    if features["y1_bad"] or features.get("y1_uncertain", False):
        return "static_qd"
    if (
        features["q_centroid_sim"] is not None
        and features["q_centroid_sim"] < 0.35
    ):
        return "static_qd"
    # A complete, non-uncertain probe with no newly exposed entity has no
    # useful bridge for another retrieval hop. Keep its existing evidence
    # instead of forcing low-value decomposition.
    return "direct"


def answer_type_terms(answer_type, question):
    lowered = question.lower()
    if answer_type == "number":
        if any(term in lowered for term in ("seat", "capacity", "arena", "stadium")):
            return "seating capacity seats"
        return "number total"
    if answer_type == "date":
        return "year date"
    if answer_type == "location":
        return "birthplace location place"
    if answer_type == "person":
        return "person name"
    if answer_type == "yesno":
        return "nationality identity facts"
    return "facts"


def query_focus_terms(question, answer_type):
    for pattern, focus in QUERY_FOCUS_PATTERNS:
        if re.search(pattern, question, re.I):
            return focus
    return answer_type_terms(answer_type, question)


def build_heuristic_missing_query(question, features, r0_titles, probe_answer=""):
    entities = features.get("y1_new_entities", [])
    if not entities:
        return ""
    title_lookup = {title.lower(): title for title in r0_titles}
    lowered_probe = (probe_answer or "").lower()

    def entity_score(entity):
        lowered_entity = entity.lower()
        relation_match = bool(
            re.search(
                rf"\b(?:is|was|are|were|named|called|at|in)\s+(?:the\s+)?"
                rf"{re.escape(lowered_entity)}\b",
                lowered_probe,
            )
        )
        return (
            lowered_entity in title_lookup,
            relation_match,
            len(entity.split()),
            len(entity),
        )

    best_entity = max(
        entities,
        key=entity_score,
    )
    focus = query_focus_terms(question, features["answer_type"])
    return f"{best_entity} {focus}".strip()


def add_rrf_scores(candidate_pool, rrf_k=60):
    for doc in candidate_pool:
        score = 0.0
        for query_id, rank in doc.get("ranks", {}).items():
            if query_id == "original":
                source = "original"
            elif query_id.startswith("qd_"):
                source = "qd"
            else:
                source = "generation_guided"
            score += SOURCE_RRF_WEIGHTS[source] / (rrf_k + int(rank))
        doc["rrf_score"] = score
    return candidate_pool


def _token_set(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _title_in_text(title, text):
    normalized = " ".join(title.lower().split())
    return len(normalized) > 2 and normalized in " ".join(text.lower().split())


def extract_comparison_sides(question):
    cleaned = question.strip().rstrip("?")
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:,|$)", cleaned, re.I)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.search(
        r"^(?:were|are|was|is|did|do)\s+(.+?)\s+and\s+(.+?)\s+"
        r"(?:of|from|both|the|born|have|had)\b",
        cleaned,
        re.I,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.search(r",\s*(.+?)\s+or\s+(.+?)$", cleaned, re.I)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    proper = re.findall(r"\b[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)+", question)
    return (proper[0], proper[1]) if len(proper) >= 2 else ("", "")


def compute_role_scores(doc, question, probe_answer, features):
    title = doc_title(doc)
    text = f"{title}\n{doc_contents(doc)}"
    lowered_text = text.lower()
    question_tokens = _token_set(question)
    title_tokens = _token_set(title)
    overlap = len(question_tokens & title_tokens) / max(1, len(title_tokens))
    original_rank = doc.get("ranks", {}).get("original")
    anchor = min(
        1.0,
        (0.7 if _title_in_text(title, question) or overlap >= 0.5 else 0.0)
        + (0.3 if original_rank is not None and original_rank <= 5 else 0.0),
    )
    bridge = min(
        1.0,
        (
            0.7
            if _title_in_text(title, probe_answer) and not _title_in_text(title, question)
            else 0.0
        )
        + (0.3 if "generation_guided" in doc.get("sources", []) else 0.0),
    )
    answer = 1.0 if answer_type_hit(text, features["answer_type"]) else 0.0
    left, right = extract_comparison_sides(question)
    comparison_left = 1.0 if left and left.lower() in lowered_text else 0.0
    comparison_right = 1.0 if right and right.lower() in lowered_text else 0.0
    relation_terms = [term for term in RELATION_TERMS if term in question.lower()]
    constraint = (
        sum(term in lowered_text for term in relation_terms) / len(relation_terms)
        if relation_terms
        else 0.0
    )
    return {
        "anchor": anchor,
        "bridge": bridge,
        "answer": answer,
        "comparison_left": comparison_left,
        "comparison_right": comparison_right,
        "constraint": constraint,
    }


def role_goal_weights(question_type):
    if question_type == "comparison":
        return {
            "comparison_left": 0.35,
            "comparison_right": 0.35,
            "answer": 0.20,
            "constraint": 0.10,
        }
    if question_type in {"bridge", "constraint"}:
        return {
            "anchor": 0.25,
            "bridge": 0.35,
            "answer": 0.30,
            "constraint": 0.10,
        }
    return {"anchor": 0.40, "answer": 0.40, "constraint": 0.20}


def lexical_redundancy(doc, selected):
    if not selected:
        return 0.0
    doc_tokens = _token_set(doc_contents(doc))
    penalty = 0.0
    for other in selected:
        if doc_title(doc).lower() == doc_title(other).lower():
            penalty = max(penalty, 1.0)
            continue
        other_tokens = _token_set(doc_contents(other))
        union = doc_tokens | other_tokens
        if union:
            penalty = max(penalty, len(doc_tokens & other_tokens) / len(union))
    return penalty


def role_aware_pack(
    candidate_pool,
    question,
    probe_answer,
    features,
    final_topk,
    route,
    config,
):
    role_weights = role_goal_weights(features["question_type"])
    selected = []
    covered = {role: 0.0 for role in ROLE_NAMES}
    title_counts = Counter()
    covered_sources = set()
    for doc in candidate_pool:
        doc["role_scores"] = compute_role_scores(doc, question, probe_answer, features)

    original_seed_count = min(
        int(config.get("original_seed_count", 0)),
        final_topk,
    )
    if original_seed_count:
        original_docs = [
            doc for doc in candidate_pool if "original" in doc.get("sources", [])
        ]
        selected = select_ranked_title_diverse(
            original_docs, original_seed_count, max_same_title=1
        )
        for doc in selected:
            title_counts[doc_title(doc).lower()] += 1
            covered_sources.update(doc.get("sources", []))
            for role in ROLE_NAMES:
                covered[role] = max(
                    covered[role], doc["role_scores"].get(role, 0.0)
                )

    while len(selected) < min(final_topk, len(candidate_pool)):
        best_doc = None
        best_gain = None
        for doc in candidate_pool:
            if doc in selected:
                continue
            title_key = doc_title(doc).lower()
            if config.get("title_dedup_soft") and title_counts[title_key] >= int(
                config.get("max_same_title", 2)
            ):
                title_penalty = 1.0
            else:
                title_penalty = 0.0
            role_gain = sum(
                max(0.0, doc["role_scores"].get(role, 0.0) - covered[role]) * weight
                for role, weight in role_weights.items()
            )
            title_gain = 1.0 if title_counts[title_key] == 0 else 0.0
            source_gain = (
                1.0 if set(doc.get("sources", [])) - covered_sources else 0.0
            )
            redundancy = lexical_redundancy(doc, selected)
            gain = (
                float(config.get("rrf_weight", 1.0)) * float(doc.get("rrf_score", 0.0))
                + float(config.get("role_weight", 0.30)) * role_gain
                + float(config.get("title_weight", 0.05)) * title_gain
                + float(config.get("source_weight", 0.05)) * source_gain
                - float(config.get("redundancy_weight", 0.05)) * redundancy
                - title_penalty
            )
            tie_break = (
                gain,
                float(doc.get("rrf_score", 0.0)),
                -min(doc.get("ranks", {}).values(), default=10**9),
            )
            if best_gain is None or tie_break > best_gain:
                best_gain = tie_break
                best_doc = doc
        selected.append(best_doc)
        title_counts[doc_title(best_doc).lower()] += 1
        covered_sources.update(best_doc.get("sources", []))
        for role in ROLE_NAMES:
            covered[role] = max(covered[role], best_doc["role_scores"].get(role, 0.0))

    return order_context(selected, route), covered


def _min_rank_for_prefix(doc, prefix):
    ranks = [
        rank
        for query_id, rank in doc.get("ranks", {}).items()
        if query_id.startswith(prefix)
    ]
    return min(ranks) if ranks else 10**9


def static_bridge_pack(
    candidate_pool,
    question,
    probe_answer,
    features,
    final_topk,
    config,
):
    for doc in candidate_pool:
        doc["role_scores"] = compute_role_scores(doc, question, probe_answer, features)

    selected = []
    selected_uids = set()
    title_counts = Counter()

    def add_docs(docs, limit):
        added = 0
        for doc in docs:
            if len(selected) >= final_topk or added >= limit:
                break
            uid = doc.get("doc_uid")
            title_key = doc_title(doc).lower()
            if uid in selected_uids or title_counts[title_key] >= 1:
                continue
            selected.append(doc)
            selected_uids.add(uid)
            title_counts[title_key] += 1
            added += 1
        return added

    original_limit = min(
        int(config.get("static_bridge_original_count", 4)),
        final_topk,
    )
    qd_limit = min(
        int(config.get("static_bridge_qd_count", final_topk - original_limit)),
        final_topk - original_limit,
    )
    original_docs = sorted(
        [doc for doc in candidate_pool if "original" in doc.get("sources", [])],
        key=lambda doc: doc.get("ranks", {}).get("original", 10**9),
    )
    qd_docs = sorted(
        [doc for doc in candidate_pool if "qd" in doc.get("sources", [])],
        key=lambda doc: (
            _min_rank_for_prefix(doc, "qd_"),
            -float(doc.get("rrf_score", 0.0)),
        ),
    )

    add_docs(original_docs, original_limit)
    add_docs(qd_docs, qd_limit)

    remaining_docs = sorted(
        candidate_pool,
        key=lambda doc: (
            -float(doc.get("rrf_score", 0.0)),
            min(doc.get("ranks", {}).values(), default=10**9),
        ),
    )
    add_docs(remaining_docs, final_topk - len(selected))

    if len(selected) < min(final_topk, len(candidate_pool)):
        for doc in remaining_docs:
            if len(selected) >= min(final_topk, len(candidate_pool)):
                break
            uid = doc.get("doc_uid")
            if uid in selected_uids:
                continue
            selected.append(doc)
            selected_uids.add(uid)

    return order_context(selected, "static_qd"), score_selected_roles(
        selected, question, probe_answer, features
    )


def score_selected_roles(selected, question, probe_answer, features):
    covered = {role: 0.0 for role in ROLE_NAMES}
    for doc in selected:
        doc["role_scores"] = compute_role_scores(doc, question, probe_answer, features)
        for role in ROLE_NAMES:
            covered[role] = max(covered[role], doc["role_scores"].get(role, 0.0))
    return covered


def select_ranked_title_diverse(docs, final_topk, max_same_title=1):
    selected = []
    deferred = []
    title_counts = Counter()
    ranked_docs = sorted(
        docs, key=lambda doc: doc.get("ranks", {}).get("original", 10**9)
    )
    for doc in ranked_docs:
        title_key = doc_title(doc).lower()
        if title_counts[title_key] >= max_same_title:
            deferred.append(doc)
            continue
        selected.append(doc)
        title_counts[title_key] += 1
        if len(selected) == final_topk:
            return selected

    for doc in deferred:
        selected.append(doc)
        if len(selected) == final_topk:
            break
    return selected


def select_rrf_only(candidate_pool, final_topk, route):
    """Select the final context using only fused RRF scores.

    This is an isolated ablation selector.  It intentionally ignores role,
    title/source diversity, redundancy, original-document reservation, and
    static-QD quotas while preserving the route-specific presentation order.
    """
    ranked_docs = sorted(
        candidate_pool,
        key=lambda doc: (
            -float(doc.get("rrf_score", 0.0)),
            min(doc.get("ranks", {}).values(), default=10**9),
            str(doc.get("doc_uid", "")),
        ),
    )
    return order_context(ranked_docs[:final_topk], route)


def order_context(selected, route):
    def source_rank(doc, prefix):
        ranks = [
            rank for query_id, rank in doc.get("ranks", {}).items() if query_id.startswith(prefix)
        ]
        return min(ranks) if ranks else 10**9

    if route == "direct":
        return sorted(selected, key=lambda doc: doc.get("ranks", {}).get("original", 10**9))
    expansion_prefix = "qd_" if route == "static_qd" else "gen_"
    return sorted(
        selected,
        key=lambda doc: (
            0 if "original" in doc.get("sources", []) else 1,
            doc.get("ranks", {}).get("original", source_rank(doc, expansion_prefix)),
            doc.get("role_scores", {}).get("answer", 0.0),
        ),
    )


def role_coverage_ratio(covered, question_type):
    weights = role_goal_weights(question_type)
    return sum(covered.get(role, 0.0) * weight for role, weight in weights.items()) / max(
        1e-12, sum(weights.values())
    )
