"""Gerenciamento de histórico e exportação de dados."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from face_analyzer.analyzer.engine import AnalysisResult


class HistoryManager:
    """Gerencia histórico de análises para comparação temporal."""

    def __init__(self, storage_path: str | Path = "reports/history.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, object]] = self._load()

    def _load(self) -> list[dict[str, object]]:
        """Carrega histórico do disco."""
        if self.storage_path.exists():
            with open(self.storage_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        """Salva histórico no disco."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def add(self, result: AnalysisResult, label: Optional[str] = None) -> None:
        """Adiciona resultado ao histórico."""
        entry = result.to_flat_dict()
        entry["label"] = label or result.timestamp.strftime("%m/%d/%Y %H:%M")
        self._history.append(entry)
        self._save()

    def get_all(self) -> list[dict[str, object]]:
        """Retorna todo o histórico."""
        return self._history.copy()

    def clear(self) -> None:
        """Limpa histórico."""
        self._history = []
        self._save()


class DataExporter:
    """Exporta resultados para CSV, Excel e JSON."""

    @staticmethod
    def to_json(result: AnalysisResult, indent: int = 2) -> str:
        """Exporta resultado como JSON."""
        return json.dumps(result.to_flat_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def to_csv(result: AnalysisResult) -> str:
        """Exporta métricas principais como CSV."""
        flat = result.to_flat_dict()
        rows: list[dict[str, object]] = []
        for key, value in flat.items():
            if not isinstance(value, (dict, list)):
                rows.append({"metric": key, "value": value})
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    @staticmethod
    def to_excel(result: AnalysisResult) -> bytes:
        """Exporta métricas como Excel."""
        flat = result.to_flat_dict()
        rows: list[dict[str, object]] = []
        for key, value in flat.items():
            if not isinstance(value, (dict, list)):
                rows.append({"Metric": key, "Value": value})
        df = pd.DataFrame(rows)
        buffer = __import__("io").BytesIO()
        df.to_excel(buffer, index=False, sheet_name="Analysis")
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def save_json(result: AnalysisResult, path: str | Path) -> None:
        """Salva JSON em arquivo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(DataExporter.to_json(result))

    @staticmethod
    def save_csv(result: AnalysisResult, path: str | Path) -> None:
        """Salva CSV em arquivo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(DataExporter.to_csv(result))
