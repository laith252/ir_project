class QueryHistoryService:
    @staticmethod
    def add(history: list[str], query: str, limit: int = 5) -> list[str]:
        cleaned = " ".join(str(query or "").split())
        if not cleaned:
            return list(history[:limit])
        normalized = cleaned.casefold()
        without_duplicate = [item for item in history if item.casefold() != normalized]
        return [cleaned, *without_duplicate][:limit]
