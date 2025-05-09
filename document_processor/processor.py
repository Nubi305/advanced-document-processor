import os
import logging
import yaml
from typing import Dict, List, Union, Optional, Tuple, Any
from pathlib import Path

from .extractors import (
    TextExtractor,
    TableExtractor,
    OCRExtractor,
    LayoutExtractor,
    FinancialExtractor
)
from .utils import (
    PreprocessingPipeline,
    QualityAssessor,
    DocumentResult,
    ExtractedTable
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processing class that orchestrates the extraction pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the document processor with configuration.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default config.
        """
        self.config = self._load_config(config_path)
        self.preprocessing = PreprocessingPipeline(self.config["image_preprocessing"])
        self.quality_assessor = QualityAssessor(self.config["quality"])
        
        # Initialize extractors
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor(self.config["table_extraction"])
        self.ocr_extractor = OCRExtractor(self.config["ocr"])
        self.layout_extractor = LayoutExtractor(self.config["layout"])
        
        if self.config["financial"]["enable_financial_extraction"]:
            self.financial_extractor = FinancialExtractor(self.config["financial"])
        else:
            self.financial_extractor = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use default."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Use default config from package
            default_config = Path(__file__).parent.parent / "config.yaml"
            with open(default_config, 'r') as f:
                return yaml.safe_load(f)
    
    def process(self, file_path: str, **kwargs) -> DocumentResult:
        """
        Process a document and extract information.
        
        Args:
            file_path: Path to the document file (PDF, image, etc.)
            **kwargs: Additional processing options that override config
            
        Returns:
            DocumentResult containing extracted information
        """
        logger.info(f"Processing document: {file_path}")
        
        # Determine document type and choose extraction strategy
        doc_type = self._determine_document_type(file_path)
        strategy = self._select_extraction_strategy(doc_type)
        
        result = DocumentResult(file_path=file_path)
        
        # Execute the selected extraction strategy
        if strategy == "direct":
            self._execute_direct_extraction(file_path, result)
        elif strategy == "layout":
            self._execute_layout_extraction(file_path, result)
        elif strategy == "ocr":
            self._execute_ocr_extraction(file_path, result)
        elif strategy == "hybrid":
            self._execute_hybrid_extraction(file_path, result)
        
        # Apply financial extraction if enabled
        if self.financial_extractor:
            result.financial_data = self.financial_extractor.extract(result)
        
        # Assess quality and validate results
        self.quality_assessor.assess(result)
        
        return result
    
    def _determine_document_type(self, file_path: str) -> str:
        """Determine the type of document for processing strategy selection."""
        # Implement logic to determine if document is text-based PDF,
        # scanned document, image, etc.
        extension = file_path.split('.')[-1].lower()
        
        if extension == "pdf":
            # Check if PDF is text-based or scanned
            is_text_based = self.text_extractor.is_text_based_pdf(file_path)
            return "text_pdf" if is_text_based else "scanned_pdf"
        elif extension in ["jpg", "jpeg", "png", "tiff", "bmp"]:
            return "image"
        else:
            return "unknown"
    
    def _select_extraction_strategy(self, doc_type: str) -> str:
        """Select the extraction strategy based on document type."""
        if doc_type == "text_pdf":
            return "direct"
        elif doc_type == "scanned_pdf":
            return "hybrid" if self.config["layout"]["use_layout_parser"] else "ocr"
        elif doc_type == "image":
            return "ocr"
        else:
            return "direct"  # Default fallback
    
    def _execute_direct_extraction(self, file_path: str, result: DocumentResult) -> None:
        """Execute direct text extraction for text-based PDFs."""
        result.text_content = self.text_extractor.extract_text(file_path)
        result.tables = self.table_extractor.extract_tables(file_path)
    
    def _execute_layout_extraction(self, file_path: str, result: DocumentResult) -> None:
        """Execute layout-based extraction for structured documents."""
        layout_result = self.layout_extractor.extract_layout(file_path)
        result.text_content = layout_result.get("text", "")
        result.tables = layout_result.get("tables", [])
        result.layout = layout_result.get("layout", {})
    
    def _execute_ocr_extraction(self, file_path: str, result: DocumentResult) -> None:
        """Execute OCR-based extraction for scanned documents or images."""
        # Preprocess images if enabled
        if self.config["image_preprocessing"]["enable"]:
            preprocessed_file = self.preprocessing.preprocess(file_path)
        else:
            preprocessed_file = file_path
            
        ocr_result = self.ocr_extractor.extract_text(preprocessed_file)
        result.text_content = ocr_result.get("text", "")
        result.ocr_confidence = ocr_result.get("confidence", 0.0)
        
        # Extract tables from OCR result if possible
        if self.table_extractor.supports_ocr_input():
            result.tables = self.table_extractor.extract_tables_from_ocr(ocr_result)
    
    def _execute_hybrid_extraction(self, file_path: str, result: DocumentResult) -> None:
        """Execute hybrid extraction combining multiple methods."""
        # First try layout analysis
        layout_result = self.layout_extractor.extract_layout(file_path)
        
        # Then do OCR with preprocessing
        if self.config["image_preprocessing"]["enable"]:
            preprocessed_file = self.preprocessing.preprocess(file_path)
        else:
            preprocessed_file = file_path
            
        ocr_result = self.ocr_extractor.extract_text(preprocessed_file)
        
        # Combine results based on quality
        layout_quality = self.quality_assessor.estimate_quality(layout_result.get("text", ""))
        ocr_quality = ocr_result.get("confidence", 0.0)
        
        if layout_quality > ocr_quality and layout_quality > self.config["table_extraction"]["hybrid_threshold"]:
            result.text_content = layout_result.get("text", "")
            result.tables = layout_result.get("tables", [])
            result.layout = layout_result.get("layout", {})
        else:
            result.text_content = ocr_result.get("text", "")
            result.ocr_confidence = ocr_quality
            # Extract tables from OCR result
            result.tables = self.table_extractor.extract_tables_from_ocr(ocr_result)
        
        # Store both results for potential later use
        result.extraction_attempts = {
            "layout": layout_result,
            "ocr": ocr_result
        }