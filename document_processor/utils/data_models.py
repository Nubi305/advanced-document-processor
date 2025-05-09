from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class BoundingBox:
    """Represents a bounding box in a document."""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class ExtractedTable:
    """Represents an extracted table from a document."""
    data: List[List[str]]
    page: int
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    headers: Optional[List[str]] = None
    extraction_method: str = ""
    
    def to_dict(self) -> Dict:
        """Convert table to a dictionary."""
        result = {}
        if self.headers:
            for i, row in enumerate(self.data):
                row_dict = {}
                for j, cell in enumerate(row):
                    if j < len(self.headers):
                        row_dict[self.headers[j]] = cell
                    else:
                        row_dict[f"column_{j}"] = cell
                result[f"row_{i}"] = row_dict
        else:
            result = {"data": self.data}
            
        return result

@dataclass
class LayoutInfo:
    """Represents layout information extracted from a document."""
    elements: List[Dict]
    page_dimensions: List[Dict]
    
    def get_elements_by_type(self, element_type: str) -> List[Dict]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.get("type") == element_type]

@dataclass
class FinancialData:
    """Represents extracted financial information."""
    amounts: List[Dict] = field(default_factory=list)
    dates: List[Dict] = field(default_factory=list)
    invoice_numbers: List[Dict] = field(default_factory=list)
    account_numbers: List[Dict] = field(default_factory=list)
    parties: List[Dict] = field(default_factory=list)
    tax_ids: List[Dict] = field(default_factory=list)
    custom_entities: Dict[str, List[Dict]] = field(default_factory=dict)
    
    @property
    def total_amount(self) -> Optional[float]:
        """Get the highest confidence amount that appears to be a total."""
        total_candidates = [a for a in self.amounts if a.get("is_total", False)]
        if total_candidates:
            total_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return total_candidates[0].get("value")
        elif self.amounts:
            # Fallback: return the highest amount
            return max(self.amounts, key=lambda x: x.get("value", 0)).get("value")
        return None
    
    @property
    def invoice_date(self) -> Optional[datetime]:
        """Get the highest confidence date that appears to be an invoice date."""
        invoice_date_candidates = [d for d in self.dates if d.get("type") == "invoice_date"]
        if invoice_date_candidates:
            invoice_date_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return invoice_date_candidates[0].get("value")
        elif self.dates:
            # Fallback: return the earliest date
            return min(self.dates, key=lambda x: x.get("value")).get("value")
        return None

@dataclass
class DocumentResult:
    """Represents the result of document processing."""
    file_path: str
    text_content: str = ""
    tables: List[ExtractedTable] = field(default_factory=list)
    layout: Optional[LayoutInfo] = None
    financial_data: Optional[FinancialData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ocr_confidence: float = 0.0
    extraction_attempts: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert the document result to a dictionary."""
        result = {
            "file_path": self.file_path,
            "text_content": self.text_content,
            "tables": [table.to_dict() for table in self.tables],
            "ocr_confidence": self.ocr_confidence,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }
        
        if self.financial_data:
            result["financial_data"] = {
                "amounts": self.financial_data.amounts,
                "dates": self.financial_data.dates,
                "invoice_numbers": self.financial_data.invoice_numbers,
                "account_numbers": self.financial_data.account_numbers,
                "parties": self.financial_data.parties,
                "tax_ids": self.financial_data.tax_ids,
                "custom_entities": self.financial_data.custom_entities,
                "total_amount": self.financial_data.total_amount,
                "invoice_date": self.financial_data.invoice_date
            }
        
        return result