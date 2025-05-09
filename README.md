# Advanced Document Processor

A comprehensive document processing system with enhanced OCR, layout analysis, and financial document specialization.

## Features

- **Advanced OCR**: Tesseract 5.0+, PaddleOCR, and cloud-based OCR options
- **Layout Analysis**: Document structure understanding with LayoutParser
- **Table Extraction**: Multiple strategies for complex table extraction
- **Financial Document Processing**: Specialized tools for financial documents
- **Image Preprocessing**: Enhanced preprocessing for better extraction results
- **Quality Assessment**: Automatic quality evaluation and result validation

## Architecture

The system uses a tiered approach with multiple extraction strategies:

1. **Direct Extraction**: For text-based PDFs using PyMuPDF and PDFPlumber
2. **Layout Analysis**: For structured documents using LayoutParser
3. **OCR Processing**: For image-based documents using PaddleOCR or Tesseract
4. **Deep Learning Models**: For complex layouts using LayoutLM and TableTransformer
5. **Cloud APIs**: As fallback for difficult documents

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
results = processor.process("invoice.pdf")
print(results)
```

## Configuration

See `config.yaml` for configuration options for each extraction method.

## License

MIT