import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
import numpy as np
import fitz  # PyMuPDF
from ..utils import ExtractedTable, BoundingBox, LayoutInfo

logger = logging.getLogger(__name__)

class LayoutExtractor:
    """
    Extracts document layout information using LayoutParser.
    
    This extractor provides structure recognition for documents,
    identifying regions like text blocks, tables, figures, etc.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the layout extractor with configuration.
        
        Args:
            config: Layout extractor configuration dictionary
        """
        self.config = config
        self.use_layout_parser = config.get("use_layout_parser", True)
        self.model_path = config.get("detectron2_model", "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/model_final")
        self.label_map = config.get("label_map", {
            0: "Text",
            1: "Title",
            2: "List",
            3: "Table",
            4: "Figure"
        })
        
        # Initialize layout parser on first use to save resources
        self._layout_model = None
    
    def extract_layout(self, file_path: str) -> Dict:
        """
        Extract layout information from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing layout information, text content, and tables
        """
        # Initialize layout model if needed
        if self.use_layout_parser and not self._layout_model:
            self._init_layout_model()
        
        # For PDFs, we need to convert pages to images first
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path)
        else:
            # For images, process directly
            return self._process_image(file_path)
    
    def _init_layout_model(self):
        """Initialize LayoutParser model."""
        try:
            import layoutparser as lp
            
            # Initialize the model
            self._layout_model = lp.Detectron2LayoutModel(
                config_path=self.model_path,
                label_map=self.label_map,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
            )
            
            logger.info("LayoutParser model initialized successfully")
        except ImportError:
            logger.error("LayoutParser not available. Install with: pip install layoutparser detectron2")
            self.use_layout_parser = False
        except Exception as e:
            logger.error(f"Error initializing LayoutParser model: {e}")
            self.use_layout_parser = False
    
    def _process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file by converting pages to images and analyzing layout.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with layout information, text and tables
        """
        try:
            doc = fitz.open(pdf_path)
            layout_result = {
                "layout": {
                    "elements": [],
                    "page_dimensions": []
                },
                "text": "",
                "tables": []
            }
            
            text_blocks = []
            
            # Process each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Store page dimensions
                layout_result["layout"]["page_dimensions"].append({
                    "page": page_num,
                    "width": page.rect.width,
                    "height": page.rect.height
                })
                
                if self.use_layout_parser:
                    # Convert page to image for layout analysis
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                        pix.save(tmp_path)
                    
                    # Process with LayoutParser
                    page_layout = self._analyze_layout(tmp_path, page_num)
                    layout_result["layout"]["elements"].extend(page_layout["elements"])
                    
                    # Extract text using layout information
                    for element in page_layout["elements"]:
                        if element["type"] in ["Text", "Title", "List"]:
                            # Extract text from the region
                            bbox = element["bbox"]
                            clip_rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
                            text = page.get_text("text", clip=clip_rect)
                            text_blocks.append(text)
                            
                            # Add text to the element
                            element["text"] = text
                    
                    # Extract tables
                    tables = self._extract_tables_from_layout(page_layout, page, page_num)
                    layout_result["tables"].extend(tables)
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                else:
                    # Fallback to PyMuPDF's built-in text extraction
                    text = page.get_text("text")
                    text_blocks.append(text)
                    
                    # Get blocks
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        block_type = "Text"  # Default type
                        
                        # Try to determine block type
                        if "lines" in block:
                            # This is likely a text block
                            if block.get("type", 0) == 1:
                                block_type = "Image"
                            
                            # Check if it might be a title (simplistic approach)
                            if block_type == "Text" and "lines" in block and len(block["lines"]) == 1:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        for span in line["spans"]:
                                            # Larger font size might indicate a title
                                            if span.get("size", 0) > 14:
                                                block_type = "Title"
                                                break
                        
                        # Add to layout elements
                        layout_result["layout"]["elements"].append({
                            "type": block_type,
                            "page": page_num,
                            "bbox": {
                                "x0": block["bbox"][0],
                                "y0": block["bbox"][1],
                                "x1": block["bbox"][2],
                                "y1": block["bbox"][3]
                            },
                            "confidence": 0.9  # High confidence for PyMuPDF extraction
                        })
            
            # Combine text blocks
            layout_result["text"] = "\n\n".join(text_blocks)
            
            doc.close()
            return layout_result
            
        except Exception as e:
            logger.error(f"Error processing PDF with layout analysis: {e}")
            return {"text": "", "layout": {"elements": [], "page_dimensions": []}, "tables": []}
    
    def _process_image(self, image_path: str) -> Dict:
        """
        Process a single image with layout analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with layout information
        """
        if not self.use_layout_parser:
            logger.warning("LayoutParser is disabled, cannot process image")
            return {"text": "", "layout": {"elements": [], "page_dimensions": []}, "tables": []}
        
        try:
            # Process with LayoutParser
            layout_result = self._analyze_layout(image_path, page_num=0)
            
            # We need OCR to extract text from the image regions
            # This is handled by the OCR extractor, so here we just 
            # return the layout information
            
            return {
                "layout": {
                    "elements": layout_result["elements"],
                    "page_dimensions": [{
                        "page": 0,
                        "width": layout_result.get("width", 0),
                        "height": layout_result.get("height", 0)
                    }]
                },
                "text": "",  # OCR extractor will handle text extraction
                "tables": []  # Table extraction requires OCR for images
            }
            
        except Exception as e:
            logger.error(f"Error processing image with layout analysis: {e}")
            return {"text": "", "layout": {"elements": [], "page_dimensions": []}, "tables": []}
    
    def _analyze_layout(self, image_path: str, page_num: int) -> Dict:
        """
        Analyze document layout using LayoutParser.
        
        Args:
            image_path: Path to the document image
            page_num: Page number
            
        Returns:
            Dictionary with layout elements
        """
        import layoutparser as lp
        import cv2
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Get layout
            layout = self._layout_model.detect(image)
            
            # Convert layout to our format
            elements = []
            for block in layout:
                # Get coordinates
                x0 = block.block.x_1
                y0 = block.block.y_1
                x1 = block.block.x_2
                y1 = block.block.y_2
                
                # Get type and score
                block_type = block.type
                score = block.score
                
                # Add to elements
                elements.append({
                    "type": block_type,
                    "page": page_num,
                    "bbox": {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1
                    },
                    "confidence": float(score)
                })
            
            return {
                "elements": elements,
                "width": width,
                "height": height
            }
            
        except Exception as e:
            logger.error(f"Error in layout analysis: {e}")
            return {"elements": [], "width": 0, "height": 0}
    
    def _extract_tables_from_layout(self, layout: Dict, page, page_num: int) -> List[ExtractedTable]:
        """
        Extract tables using layout information.
        
        Args:
            layout: Layout information
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        for element in layout["elements"]:
            if element["type"] == "Table":
                # Get table region
                bbox = element["bbox"]
                clip_rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
                
                # Try to extract table using PyMuPDF
                try:
                    # Get all text blocks within the table region
                    text_blocks = []
                    blocks = page.get_text("dict", clip=clip_rect)["blocks"]
                    
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                text_line = []
                                if "spans" in line:
                                    for span in line["spans"]:
                                        text_line.append(span["text"])
                                
                                if text_line:
                                    text_blocks.append(" ".join(text_line))
                    
                    # Very naive table parsing - split by whitespace
                    # This is a simplistic approach and would need to be improved
                    data = []
                    for line in text_blocks:
                        if line.strip():
                            cells = line.split()
                            if cells:
                                data.append(cells)
                    
                    # Create bounding box
                    table_bbox = BoundingBox(
                        x0=bbox["x0"],
                        y0=bbox["y0"],
                        x1=bbox["x1"],
                        y1=bbox["y1"],
                        page=page_num
                    )
                    
                    # Create table object
                    table = ExtractedTable(
                        data=data,
                        page=page_num,
                        bbox=table_bbox,
                        confidence=element.get("confidence", 0.0),
                        extraction_method="layout_parser"
                    )
                    
                    tables.append(table)
                    
                except Exception as e:
                    logger.error(f"Error extracting table from layout: {e}")
        
        return tables