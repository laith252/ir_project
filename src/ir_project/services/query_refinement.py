from dataclasses import dataclass, field
from collections import Counter

from ir_project.services.text_processing import TextProcessor


DEFAULT_SYNONYMS = {
    "treatment": ["therapy", "medication", "medicine", "management"],
    "treat": ["therapy", "medication", "manage"],
    "symptoms": ["signs", "indications", "manifestations"],
    "symptom": ["sign", "indication"],
    "diabetes": ["diabetic", "glucose", "insulin"],
    "covid": ["coronavirus", "virus"],
    "coronavirus": ["covid", "viral", "infection"],
    "cancer": ["tumor", "oncology"],
    "medicine": ["medication", "drug", "therapy"],
    "doctor": ["physician", "clinician"],
}

SPELLING_CORRECTIONS = {
    "diabete": "diabetes",
    "diabetis": "diabetes",
    "corona": "coronavirus",
    "medecine": "medicine",
    "symptons": "symptoms",
    "treatement": "treatment",
}


@dataclass
class QueryRefinementService:
    processor: TextProcessor = field(default_factory=TextProcessor)
    max_synonyms_per_token: int = 3
    max_history_terms: int = 4

    def refine(self, query: str) -> str:
        tokens = self.processor.tokenize(query)
        expanded: list[str] = []
        for token in tokens:
            corrected = SPELLING_CORRECTIONS.get(token, token)
            expanded.append(corrected)
            expanded.extend(DEFAULT_SYNONYMS.get(corrected, [])[: self.max_synonyms_per_token])
        if not expanded:
            return query
        return " ".join(expanded)

    def refine_with_history(self, query: str, history: list[str] | None = None) -> str:
        refined = self.refine(query)
        history_terms = self.history_expansion_terms(query, history or [])
        if not history_terms:
            return refined
        return " ".join([refined, *history_terms])

    def suggest_from_history(self, query: str, history: list[str], limit: int = 3) -> list[str]:
        query_tokens = set(self.processor.tokenize(query))
        ranked: list[tuple[float, int, str]] = []
        seen = set()
        for index, past_query in enumerate(reversed(history)):
            normalized = " ".join(self.processor.tokenize(past_query))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            past_tokens = set(normalized.split())
            overlap = len(query_tokens & past_tokens)
            union = len(query_tokens | past_tokens) or 1
            score = overlap / union
            ranked.append((score, index, past_query))

        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        suggestions = [past_query for score, _, past_query in ranked if score > 0][:limit]
        if len(suggestions) < limit:
            suggestions.extend(
                past_query
                for score, _, past_query in ranked
                if score == 0 and past_query not in suggestions
            )
        return suggestions[:limit]

    def history_expansion_terms(self, query: str, history: list[str]) -> list[str]:
        query_tokens = set(self.processor.tokenize(query))
        if not query_tokens or not history:
            return []

        counts: Counter[str] = Counter()
        for past_query in history:
            past_tokens = self.processor.tokenize(past_query)
            if not past_tokens:
                continue
            overlap = query_tokens & set(past_tokens)
            if not overlap:
                continue
            for token in past_tokens:
                if token not in query_tokens:
                    counts[token] += 1 + len(overlap)

        return [token for token, _ in counts.most_common(self.max_history_terms)]
