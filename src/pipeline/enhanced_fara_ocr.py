#!/usr/bin/env python3
"""
Enhanced FARA OCR Processing System
Specialized for Item 14(a) table extraction with improved accuracy.
"""

import os
import json
import re
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import traceback

# Core dependencies with error handling
try:
    import cv2  # type: ignore
    import numpy as np
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install missing packages with: pip install opencv-python pytesseract pdf2image pillow numpy")
    sys.exit(1)


@dataclass
class Item14aEntry:
    """Enhanced structure for Item 14(a) table entries"""
    foreign_principal: str = ""
    date_received: str = ""
    from_whom: str = ""
    purpose: str = ""
    amount: str = ""
    amount_numeric: float = 0.0
    subtotal: str = ""
    subtotal_numeric: float = 0.0
    row_confidence: float = 0.0


@dataclass
class Item14aData:
    """Complete Item 14(a) data structure"""
    document_type: str = "FARA_14a_Detail"
    received_timestamp: str = ""
    entries: List[Item14aEntry] = None
    total_entries: int = 0
    total_amount: float = 0.0
    table_detected: bool = False
    extraction_confidence: float = 0.0
    raw_table_text: str = ""

    def __post_init__(self):
        if self.entries is None:
            self.entries = []


class EnhancedImageProcessor:
    """Advanced image preprocessing specifically for table extraction"""
    
    @staticmethod
    def enhance_for_tables(image: Image.Image) -> Image.Image:
        """Enhanced preprocessing optimized for table extraction"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Morphological operations to enhance table structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive threshold for better text extraction
        binary = cv2.adaptiveThreshold(
            morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(binary)
    
    @staticmethod
    def detect_table_regions(image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect potential table regions using contour detection"""
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines to form table structure
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours that might represent tables
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 200 and h > 100:  # Filter for reasonable table sizes
                table_regions.append((x, y, w, h))
        
        return table_regions


class Item14aTableParser:
    """Specialized parser for Item 14(a) tables"""
    
    def __init__(self):
        self.logger = logging.getLogger('Item14aParser')
        
        # Pattern for identifying Item 14(a) sections
        self.item_14a_patterns = [
            r'Item\s+14\s*\(\s*a\s*\)\s*[-\s]*\s*Detail',
            r'Response\s+to\s+Item\s+14\s*\(\s*a\s*\)\s*[-\s]*\s*Detail',
            r'14\s*\(\s*a\s*\)\s*[-\s]*\s*Detail',
            r'RECEIPTS[-\s]*MONIES.*Detail'
        ]
        
        # Column header patterns
        self.header_patterns = [
            r'Foreign\s+Principal',
            r'Date\s+Received',
            r'From\s+Whom',
            r'Purpose',
            r'Amount',
            r'Subtotal'
        ]
        
        # Currency pattern
        self.currency_pattern = r'\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2})?)'
        
        # Date pattern
        self.date_pattern = r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
    
    def detect_item_14a_tables(self, text: str) -> List[str]:
        """Detect and extract Item 14(a) table sections"""
        tables = []
        
        for pattern in self.item_14a_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                start_pos = match.start()
                # Find the end of this table section (next major section or end of document)
                end_patterns = [
                    r'\n\s*Item\s+\d+',
                    r'\n\s*\d+\.\s*\([a-z]\)',
                    r'\n\s*Received\s+by\s+NSD',
                    r'\n\s*Page\s+\d+'
                ]
                
                end_pos = len(text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text[start_pos + 100:], re.IGNORECASE)
                    if end_match:
                        end_pos = start_pos + 100 + end_match.start()
                        break
                
                table_text = text[start_pos:end_pos]
                if self._validate_table_content(table_text):
                    tables.append(table_text)
        
        return tables
    
    def _validate_table_content(self, text: str) -> bool:
        """Validate that text contains actual table content"""
        # Check for presence of required column headers
        header_count = 0
        for pattern in self.header_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                header_count += 1
        
        # Check for financial amounts
        has_amounts = bool(re.search(self.currency_pattern, text))
        
        # Check for dates
        has_dates = bool(re.search(self.date_pattern, text))
        
        return header_count >= 3 and (has_amounts or has_dates)
    
    def parse_table_to_entries(self, table_text: str) -> List[Item14aEntry]:
        """Parse table text into structured entries"""
        entries = []
        
        # Clean and normalize the text
        cleaned_text = self._clean_table_text(table_text)
        
        # Split into potential rows
        lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        
        # Filter out header lines
        data_lines = self._filter_header_lines(lines)
        
        # Group lines into entries
        entry_groups = self._group_lines_into_entries(data_lines)
        
        for group in entry_groups:
            entry = self._parse_entry_group(group)
            if self._validate_entry(entry):
                entries.append(entry)
        
        return entries
    
    def _clean_table_text(self, text: str) -> str:
        """Clean and normalize table text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'\|', 'l', text)  # | often misread as l
        text = re.sub(r'(?<!\d),(?!\d)', '', text)  # Remove commas not in numbers
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix split numbers
        
        # Normalize line breaks
        text = re.sub(r'\s*\n\s*', '\n', text)
        
        return text
    
    def _filter_header_lines(self, lines: List[str]) -> List[str]:
        """Filter out header and separator lines"""
        data_lines = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Skip obvious header lines
            if any(header in line_lower for header in ['foreign principal', 'date received', 'from whom', 'purpose', 'amount', 'subtotal']):
                continue
            
            # Skip separator lines
            if re.match(r'^[-\s_=]+$', line):
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line.strip()) < 3:
                continue
            
            data_lines.append(line)
        
        return data_lines
    
    def _group_lines_into_entries(self, lines: List[str]) -> List[List[str]]:
        """Group lines into individual table entries"""
        entries = []
        current_entry = []
        
        for line in lines:
            # Check if this line starts a new entry (contains entity name or date)
            if self._is_new_entry_start(line) and current_entry:
                if len(current_entry) > 0:
                    entries.append(current_entry)
                current_entry = [line]
            else:
                current_entry.append(line)
        
        # Add the last entry
        if current_entry:
            entries.append(current_entry)
        
        return entries
    
    def _is_new_entry_start(self, line: str) -> bool:
        """Determine if a line starts a new table entry"""
        # Lines that start with an entity name (not a date or amount)
        has_date = bool(re.search(self.date_pattern, line))
        has_amount = bool(re.search(self.currency_pattern, line))
        
        # If line has only date/amount, it's likely a continuation
        if (has_date or has_amount) and len(line.strip()) < 50:
            return False
        
        # Lines with company-like names
        company_indicators = ['government', 'ministry', 'corporation', 'company', 'ltd', 'inc', 'board', 'authority', 'agency']
        has_company_name = any(indicator in line.lower() for indicator in company_indicators)
        
        # Long lines without dates/amounts are likely entity names
        is_long_text = len(line.strip()) > 30 and not has_date and not has_amount
        
        return has_company_name or is_long_text
    
    def _parse_entry_group(self, group: List[str]) -> Item14aEntry:
        """Parse a group of lines into a single entry"""
        entry = Item14aEntry()
        
        full_text = ' '.join(group)
        
        # Extract foreign principal (usually the first substantial text)
        entry.foreign_principal = self._extract_foreign_principal(group)
        
        # Extract date
        date_match = re.search(self.date_pattern, full_text)
        if date_match:
            entry.date_received = self._clean_date(date_match.group(1))
        
        # Extract amounts
        amounts = re.findall(self.currency_pattern, full_text)
        if amounts:
            # First amount is usually the main amount
            entry.amount = amounts[0]
            entry.amount_numeric = self._parse_currency(amounts[0])
            
            # If there's a second amount, it might be subtotal
            if len(amounts) > 1:
                entry.subtotal = amounts[-1]
                entry.subtotal_numeric = self._parse_currency(amounts[-1])
        
        # Extract from_whom (often same as foreign principal or extracted separately)
        entry.from_whom = entry.foreign_principal if not entry.from_whom else entry.from_whom
        
        # Extract purpose (remaining text after removing other fields)
        entry.purpose = self._extract_purpose(full_text, entry)
        
        # Calculate confidence based on completeness
        entry.row_confidence = self._calculate_entry_confidence(entry)
        
        return entry
    
    def _extract_foreign_principal(self, group: List[str]) -> str:
        """Extract the foreign principal name from the entry group"""
        for line in group:
            # Skip lines that are only dates or amounts
            if re.match(r'^\s*[\d\/\-\s,\$\.]+\s*$', line):
                continue
            
            # Take the first substantial text line
            if len(line.strip()) > 10:
                # Clean up the line
                cleaned = re.sub(r'\s+', ' ', line.strip())
                # Remove trailing amounts/dates
                cleaned = re.sub(self.currency_pattern + r'.*$', '', cleaned)
                cleaned = re.sub(self.date_pattern + r'.*$', '', cleaned)
                return cleaned.strip()
        
        return ""
    
    def _extract_purpose(self, full_text: str, entry: Item14aEntry) -> str:
        """Extract purpose description from the full text"""
        # Remove already extracted information
        text = full_text
        if entry.foreign_principal:
            text = text.replace(entry.foreign_principal, '')
        if entry.date_received:
            text = text.replace(entry.date_received, '')
        if entry.amount:
            text = text.replace(entry.amount, '')
        
        # Remove currency symbols and amounts
        text = re.sub(self.currency_pattern, '', text)
        text = re.sub(self.date_pattern, '', text)
        
        # Clean up remaining text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Look for purpose-indicating keywords
        purpose_indicators = ['for', 'payment', 'services', 'consulting', 'fees', 'expenses', 'account', 'activities']
        if any(indicator in text.lower() for indicator in purpose_indicators):
            return text
        
        return ""
    
    def _clean_date(self, date_str: str) -> str:
        """Clean and normalize date format"""
        # Convert various date formats to MM/DD/YYYY
        date_str = date_str.replace('-', '/')
        
        # Handle date ranges (take the first date)
        if '-' in date_str or '–' in date_str:
            date_parts = re.split(r'[-–]', date_str)
            date_str = date_parts[0].strip()
        
        return date_str.strip()
    
    def _parse_currency(self, amount_str: str) -> float:
        """Parse currency string to float value"""
        try:
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[\$,\s]', '', amount_str)
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_entry_confidence(self, entry: Item14aEntry) -> float:
        """Calculate confidence score for an entry"""
        score = 0.0
        total_checks = 5.0
        
        if entry.foreign_principal and len(entry.foreign_principal) > 5:
            score += 1.0
        if entry.date_received:
            score += 1.0
        if entry.amount and entry.amount_numeric > 0:
            score += 1.0
        if entry.from_whom:
            score += 1.0
        if entry.purpose and len(entry.purpose) > 10:
            score += 1.0
        
        return (score / total_checks) * 100.0
    
    def _validate_entry(self, entry: Item14aEntry) -> bool:
        """Validate that an entry contains minimum required information"""
        return (
            bool(entry.foreign_principal) and
            (bool(entry.amount) or bool(entry.date_received)) and
            len(entry.foreign_principal) > 3
        )


class EnhancedFARAOCRProcessor:
    """Enhanced FARA OCR processor with specialized table extraction"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = self._setup_logging()
        self.image_processor = EnhancedImageProcessor()
        self.table_parser = Item14aTableParser()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('EnhancedFARAOCR')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def process_document_for_14a_tables(self, pdf_path: Path) -> Item14aData:
        """Process a document specifically for Item 14(a) tables"""
        self.logger.info(f"Processing document for Item 14(a) tables: {pdf_path}")
        
        try:
            # Convert PDF to images with high DPI for better table detection
            images = convert_from_path(pdf_path, dpi=300, fmt='jpeg')
            if not images:
                return Item14aData(received_timestamp=datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))
            
            all_text = []
            total_confidence = 0.0
            
            # Process each page with enhanced table preprocessing
            for i, image in enumerate(images):
                enhanced_image = self.image_processor.enhance_for_tables(image)
                
                # Use table-optimized OCR settings
                config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$,.()/:- '
                
                try:
                    text = pytesseract.image_to_string(enhanced_image, config=config)
                    all_text.append(text)
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(
                        enhanced_image,
                        output_type=pytesseract.Output.DICT,
                        config=config
                    )
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    page_confidence = sum(confidences) / len(confidences) if confidences else 0
                    total_confidence += page_confidence
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process page {i+1}: {e}")
                    continue
            
            # Combine all text
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            # Extract Item 14(a) tables
            result = self._extract_14a_data(combined_text, avg_confidence)
            result.received_timestamp = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process document: {e}")
            self.logger.error(traceback.format_exc())
            return Item14aData(
                received_timestamp=datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'),
                extraction_confidence=0.0
            )
    
    def _extract_14a_data(self, text: str, ocr_confidence: float) -> Item14aData:
        """Extract Item 14(a) data from the full document text"""
        result = Item14aData()
        
        # Detect Item 14(a) table sections
        table_sections = self.table_parser.detect_item_14a_tables(text)
        
        if not table_sections:
            self.logger.warning("No Item 14(a) tables detected")
            return result
        
        result.table_detected = True
        all_entries = []
        
        # Process each detected table
        for i, table_text in enumerate(table_sections):
            self.logger.info(f"Processing Item 14(a) table {i+1}")
            result.raw_table_text += f"\n--- Table {i+1} ---\n{table_text}"
            
            entries = self.table_parser.parse_table_to_entries(table_text)
            all_entries.extend(entries)
        
        # Clean and validate entries
        validated_entries = []
        total_amount = 0.0
        
        for entry in all_entries:
            if entry.row_confidence > 30.0:  # Only include entries with reasonable confidence
                validated_entries.append(entry)
                total_amount += entry.amount_numeric
        
        result.entries = validated_entries
        result.total_entries = len(validated_entries)
        result.total_amount = total_amount
        
        # Calculate overall extraction confidence
        if validated_entries:
            entry_confidences = [entry.row_confidence for entry in validated_entries]
            avg_entry_confidence = sum(entry_confidences) / len(entry_confidences)
            result.extraction_confidence = min(ocr_confidence, avg_entry_confidence)
        else:
            result.extraction_confidence = 0.0
        
        self.logger.info(f"Extracted {len(validated_entries)} valid entries with total amount ${total_amount:,.2f}")
        
        return result
    
    def save_14a_results(self, result: Item14aData, output_path: Path) -> Path:
        """Save Item 14(a) results in the requested JSON format"""
        output_data = asdict(result)
        
        # Convert entries to the requested format
        entries_data = []
        for entry in result.entries:
            entry_dict = {
                "foreign_principal": entry.foreign_principal,
                "date_received": entry.date_received,
                "from_whom": entry.from_whom,
                "purpose": entry.purpose,
                "amount": f"${entry.amount}" if entry.amount else "",
                "amount_numeric": entry.amount_numeric,
                "subtotal": f"${entry.subtotal}" if entry.subtotal else "",
                "subtotal_numeric": entry.subtotal_numeric
            }
            entries_data.append(entry_dict)
        
        output_data["entries"] = entries_data
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path


def main():
    """Main function for testing the enhanced OCR"""
    from config.settings import settings
    
    processor = EnhancedFARAOCRProcessor()
    
    # Check for input files
    input_dir = settings.raw_documents
    pdf_files = list(input_dir.glob("*.pdf")) if input_dir.exists() else []
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each document
    for pdf_file in pdf_files[:3]:  # Process first 3 files for testing
        print(f"\nProcessing: {pdf_file.name}")
        
        result = processor.process_document_for_14a_tables(pdf_file)
        
        # Save results
        output_dir = settings.processed_data
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{pdf_file.stem}_14a_enhanced.json"
        
        processor.save_14a_results(result, output_file)
        
        # Print summary
        print(f"  Table detected: {result.table_detected}")
        print(f"  Entries found: {result.total_entries}")
        print(f"  Total amount: ${result.total_amount:,.2f}")
        print(f"  Confidence: {result.extraction_confidence:.1f}%")
        print(f"  Output: {output_file.name}")


if __name__ == "__main__":
    main()