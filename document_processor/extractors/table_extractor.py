import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
import numpy as np
import fitz  # PyMuPDF
from ..utils import ExtractedTable, BoundingBox

logger = logging.getLogger(__name__)

class TableExtractor:
    """
    Extracts tables from documents using multiple methods.
    
    Supported methods:
    - Tabula-py (Java-based table extraction)
    - PyMuPDF (direct extraction for simple tables)
    - PDFPlumber (cell-based extraction)
    - PaddleOCR (for image-based tables)
    - TableTransformer (deep learning approach)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the table extractor with configuration.
        
        Args:
            config: Table extraction configuration dictionary
        """
        self.config = config
        self.default_method = config.get("default_method", "hybrid")
        self.hybrid_threshold = config.get("hybrid_threshold", 0.7)
        self.table_transformer_model = config.get("table_transformer_model")
        
        # Initialize extractors on first use to save resources
        self._tabula_extractor = None
        self._pdfplumber_extractor = None
        self._paddle_extractor = None
        self._transformer_extractor = None
    
    def extract_tables(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of ExtractedTable objects
        """
        if self.default_method == "hybrid":
            return self._extract_tables_hybrid(file_path)
        elif self.default_method == "tabula":
            return self._extract_tables_tabula(file_path)
        elif self.default_method == "pdfplumber":
            return self._extract_tables_pdfplumber(file_path)
        elif self.default_method == "paddle":
            return self._extract_tables_paddle(file_path)
        elif self.default_method == "transformer":
            return self._extract_tables_transformer(file_path)
        else:
            logger.error(f"Unknown table extraction method: {self.default_method}")
            return []
    
    def supports_ocr_input(self) -> bool:
        """Check if the extractor supports OCR input."""
        return self.default_method in ["paddle", "transformer", "hybrid"]
    
    def extract_tables_from_ocr(self, ocr_result: Dict) -> List[ExtractedTable]:
        """
        Extract tables from OCR results.
        
        Args:
            ocr_result: OCR result dictionary containing text and bounding boxes
            
        Returns:
            List of ExtractedTable objects
        """
        if self.default_method == "paddle":
            return self._extract_tables_from_paddle_ocr(ocr_result)
        elif self.default_method == "transformer":
            return self._extract_tables_from_transformer(ocr_result)
        elif self.default_method == "hybrid":
            # Try both methods and select the best results
            paddle_tables = self._extract_tables_from_paddle_ocr(ocr_result)
            transformer_tables = self._extract_tables_from_transformer(ocr_result)
            
            # Choose based on confidence or other metrics
            if paddle_tables and transformer_tables:
                avg_paddle_conf = sum(t.confidence for t in paddle_tables) / len(paddle_tables)
                avg_transformer_conf = sum(t.confidence for t in transformer_tables) / len(transformer_tables)
                
                if avg_paddle_conf > avg_transformer_conf:
                    return paddle_tables
                else:
                    return transformer_tables
            
            return paddle_tables or transformer_tables
        else:
            logger.warning(f"Method {self.default_method} does not support OCR input")
            return []
    
    def _extract_tables_hybrid(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables using a hybrid approach.
        
        This method tries multiple extraction approaches and selects the best results.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of ExtractedTable objects
        """
        # Start with the simplest method (PyMuPDF)
        tables = self._extract_tables_pymupdf(file_path)
        
        # If PyMuPDF finds good tables, use them
        if tables and self._assess_table_quality(tables) > self.hybrid_threshold:
            logger.info(f"Using PyMuPDF tables for {file_path}")
            return tables
        
        # Try PDFPlumber next
        pdfplumber_tables = self._extract_tables_pdfplumber(file_path)
        if pdfplumber_tables and self._assess_table_quality(pdfplumber_tables) > self.hybrid_threshold:
            logger.info(f"Using PDFPlumber tables for {file_path}")
            return pdfplumber_tables
        
        # Try Tabula next
        tabula_tables = self._extract_tables_tabula(file_path)
        if tabula_tables and self._assess_table_quality(tabula_tables) > self.hybrid_threshold:
            logger.info(f"Using Tabula tables for {file_path}")
            return tabula_tables
        
        # As a last resort, try deep learning methods
        paddle_tables = self._extract_tables_paddle(file_path)
        transformer_tables = self._extract_tables_transformer(file_path)
        
        # Choose the best results
        all_tables = [
            (tables, self._assess_table_quality(tables)),
            (pdfplumber_tables, self._assess_table_quality(pdfplumber_tables)),
            (tabula_tables, self._assess_table_quality(tabula_tables)),
            (paddle_tables, self._assess_table_quality(paddle_tables)),
            (transformer_tables, self._assess_table_quality(transformer_tables))
        ]
        
        # Filter out empty results
        valid_tables = [(t, q) for t, q in all_tables if t]
        
        if not valid_tables:
            logger.warning(f"No tables found in {file_path}")
            return []
        
        # Return the tables with the highest quality score
        best_tables, _ = max(valid_tables, key=lambda x: x[1])
        return best_tables
    
    def _assess_table_quality(self, tables: List[ExtractedTable]) -> float:
        """
        Assess the quality of extracted tables.
        
        Args:
            tables: List of ExtractedTable objects
            
        Returns:
            Quality score between 0 and 1
        """
        if not tables:
            return 0.0
        
        total_score = 0.0
        
        for table in tables:
            # Check if table has content
            if not table.data:
                continue
                
            # Calculate metrics for quality assessment
            # 1. Consistency in row lengths
            row_lengths = [len(row) for row in table.data]
            length_consistency = 1.0 if len(set(row_lengths)) == 1 else 0.5
            
            # 2. Cell content quality
            empty_cells = sum(1 for row in table.data for cell in row if not cell.strip())
            total_cells = sum(row_lengths)
            cell_content_quality = 1.0 - (empty_cells / max(1, total_cells))
            
            # 3. Table confidence (if available)
            confidence = table.confidence if table.confidence else 0.8  # Default confidence
            
            # Calculate overall quality
            quality = (length_consistency * 0.4 + cell_content_quality * 0.4 + confidence * 0.2)
            total_score += quality
        
        return total_score / len(tables)
    
    def _extract_tables_pymupdf(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables using PyMuPDF.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of ExtractedTable objects
        """
        try:
            doc = fitz.open(file_path)
            tables = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Try to extract tables using PyMuPDF's built-in functionality
                # This works best for simple, well-structured tables
                try:
                    # Get tables as dictionaries
                    page_tables = page.find_tables()
                    
                    for table in page_tables:
                        # Convert to our ExtractedTable format
                        data = []
                        for row in table.rows:
                            data_row = []
                            for cell in row.cells:
                                # Get text from the cell region
                                cell_text = page.get_text("text", clip=cell.rect)
                                data_row.append(cell_text.strip())
                            data.append(data_row)
                        
                        # Create bounding box
                        bbox = BoundingBox(
                            x0=table.rect.x0,
                            y0=table.rect.y0,
                            x1=table.rect.x1,
                            y1=table.rect.y1,
                            page=page_num
                        )
                        
                        # Create ExtractedTable object
                        extracted_table = ExtractedTable(
                            data=data,
                            page=page_num,
                            bbox=bbox,
                            confidence=0.9,  # PyMuPDF is high confidence when it works
                            extraction_method="pymupdf"
                        )
                        
                        # Add headers if present
                        if data:
                            extracted_table.headers = data[0]
                        
                        tables.append(extracted_table)
                except Exception as e:
                    logger.debug(f"PyMuPDF table extraction failed: {e}")
            
            doc.close()
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables with PyMuPDF: {e}")
            return []
    
    def _extract_tables_pdfplumber(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables using PDFPlumber.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of ExtractedTable objects
        """
        try:
            import pdfplumber
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables
                    page_tables = page.extract_tables()
                    
                    for table in page_tables:
                        # Convert to our ExtractedTable format
                        # PDFPlumber returns tables as lists of lists
                        data = []
                        for row in table:
                            data_row = []
                            for cell in row:
                                # PDFPlumber can return None for empty cells
                                cell_text = cell if cell is not None else ""
                                data_row.append(cell_text.strip())
                            data.append(data_row)
                        
                        # Create ExtractedTable object
                        extracted_table = ExtractedTable(
                            data=data,
                            page=page_num,
                            confidence=0.85,  # PDFPlumber is generally reliable
                            extraction_method="pdfplumber"
                        )
                        
                        # Add headers if present
                        if data:
                            extracted_table.headers = data[0]
                        
                        tables.append(extracted_table)
            
            return tables
            
        except ImportError:
            logger.error("PDFPlumber not available. Install with: pip install pdfplumber")
            return []
        except Exception as e:
            logger.error(f"Error extracting tables with PDFPlumber: {e}")
            return []
    
    def _extract_tables_tabula(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables using Tabula.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of ExtractedTable objects
        """
        try:
            import tabula
            tables = []
            
            # Extract tables with Tabula
            tabula_tables = tabula.read_pdf(
                file_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                lattice=True
            )
            
            # If lattice mode doesn't find tables, try stream mode
            if not tabula_tables:
                tabula_tables = tabula.read_pdf(
                    file_path,
                    pages='all',
                    multiple_tables=True,
                    guess=True,
                    stream=True
                )
            
            # Convert to our ExtractedTable format
            for i, df in enumerate(tabula_tables):
                # Get page number (approximate since tabula doesn't return it)
                # We assume tables are in order of pages
                page_num = i  # This is an approximation
                
                # Convert DataFrame to list of lists
                data = [df.columns.tolist()]  # Headers first
                data.extend(df.values.tolist())
                
                # Convert any non-string data
                for i, row in enumerate(data):
                    data[i] = [str(cell) if cell is not None else "" for cell in row]
                
                # Create ExtractedTable object
                extracted_table = ExtractedTable(
                    data=data,
                    page=page_num,
                    confidence=0.8,  # Tabula is generally good but can miss formatting
                    headers=df.columns.tolist(),
                    extraction_method="tabula"
                )
                
                tables.append(extracted_table)
            
            return tables
            
        except ImportError:
            logger.error("Tabula not available. Install with: pip install tabula-py")
            return []
        except Exception as e:
            logger.error(f"Error extracting tables with Tabula: {e}")
            return []
    
    def _extract_tables_paddle(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables using PaddleOCR.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of ExtractedTable objects
        """
        try:
            # This method requires converting PDF to images and then using PaddleOCR
            # for table detection and structure recognition
            from paddleocr import PPStructure
            
            # Initialize PPStructure if not already done
            if not hasattr(self, '_pp_structure'):
                self._pp_structure = PPStructure(table=True, ocr=True, show_log=False)
            
            tables = []
            
            # If the file is a PDF, we need to convert pages to images first
            if file_path.lower().endswith('.pdf'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Convert PDF pages to images
                    doc = fitz.open(file_path)
                    
                    for page_num in range(doc.page_count):
                        page = doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        image_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                        pix.save(image_path)
                        
                        # Process each page image
                        result = self._pp_structure(image_path)
                        page_tables = self._convert_paddle_tables(result, page_num)
                        tables.extend(page_tables)
                    
                    doc.close()
            else:
                # For image files, process directly
                result = self._pp_structure(file_path)
                tables = self._convert_paddle_tables(result, 0)  # Assume page 0 for images
            
            return tables
            
        except ImportError:
            logger.error("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            return []
        except Exception as e:
            logger.error(f"Error extracting tables with PaddleOCR: {e}")
            return []
    
    def _convert_paddle_tables(self, pp_result: List, page_num: int) -> List[ExtractedTable]:
        """
        Convert PaddleOCR PPStructure results to ExtractedTable objects.
        
        Args:
            pp_result: PPStructure result
            page_num: Page number
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        for block in pp_result:
            if block.get('type') == 'table':
                table_res = block.get('res', {})
                html = table_res.get('html', '')
                
                # Parse HTML to get table data
                import pandas as pd
                from io import StringIO
                
                try:
                    # Use pandas to parse HTML table
                    dfs = pd.read_html(StringIO(html))
                    
                    if dfs:
                        df = dfs[0]
                        
                        # Convert DataFrame to list of lists
                        data = [df.columns.tolist()]  # Headers first
                        data.extend(df.values.tolist())
                        
                        # Convert any non-string data
                        for i, row in enumerate(data):
                            data[i] = [str(cell) if cell is not None else "" for cell in row]
                        
                        # Get bounding box if available
                        bbox = None
                        if 'bbox' in block:
                            x0, y0, x1, y1 = block['bbox']
                            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page_num)
                        
                        # Create ExtractedTable object
                        extracted_table = ExtractedTable(
                            data=data,
                            page=page_num,
                            bbox=bbox,
                            confidence=block.get('confidence', 0.0),
                            headers=df.columns.tolist(),
                            extraction_method="paddle"
                        )
                        
                        tables.append(extracted_table)
                except Exception as e:
                    logger.error(f"Error parsing Paddle table HTML: {e}")
        
        return tables
    
    def _extract_tables_transformer(self, file_path: str) -> List[ExtractedTable]:
        """
        Extract tables using TableTransformer.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of ExtractedTable objects
        """
        # This is a placeholder for TableTransformer implementation
        # which would require Hugging Face Transformers integration
        logger.warning("TableTransformer extraction not fully implemented")
        return []
    
    def _extract_tables_from_paddle_ocr(self, ocr_result: Dict) -> List[ExtractedTable]:
        """
        Extract tables from PaddleOCR results.
        
        Args:
            ocr_result: OCR result dictionary
            
        Returns:
            List of ExtractedTable objects
        """
        # This would implement table extraction from OCR text and boxes
        # using spatial analysis and grid detection
        logger.warning("Table extraction from OCR not fully implemented")
        return []
    
    def _extract_tables_from_transformer(self, ocr_result: Dict) -> List[ExtractedTable]:
        """
        Extract tables from OCR results using transformer models.
        
        Args:
            ocr_result: OCR result dictionary
            
        Returns:
            List of ExtractedTable objects
        """
        # This would use transformer models to identify and extract tables
        # from OCR results, but is not implemented in this version
        logger.warning("Transformer table extraction from OCR not implemented")
        return []