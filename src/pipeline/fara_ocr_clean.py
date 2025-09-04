#!/usr/bin/env python3
"""
FARA OCR Processing System - Clean Tesseract-Only Implementation
Fixed version without regex syntax errors.
"""

import os
import json
import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import traceback

# Core dependencies with error handling
try:
    import cv2 # type: ignore
    import numpy as np
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install missing packages with: pip install opencv-python pytesseract pdf2image pillow numpy")
    sys.exit(1)

@dataclass
class ReceiptsEntry:
    """Structure for individual receipts entries"""
    foreign_principal: str = ""
    date_received: str = ""
    from_whom: str = ""
    purpose: str = ""
    amount: str = ""
    subtotal: str = ""

# --- in FARAData definition ---
@dataclass 
class FARAData:
    """FARA document data focused on filed responses only"""
    registrant_name: str = ""
    registration_number: str = ""
    receipts_response: str = ""  # "Yes" or "No"
    receipts_explanation: str = ""  # Explanation if No then keep only this
    receipts_entries: List[ReceiptsEntry] = None
    receipts_total: str = ""
    filing_date: str = ""
    foreign_principals_listed: List[str] = None
    confidence_score: float = 0.0
    source_file: str = ""
    processing_date: str = ""
    raw_text: str = ""

    def __post_init__(self):
        if self.foreign_principals_listed is None:
            self.foreign_principals_listed = []
        if self.receipts_entries is None:
            self.receipts_entries = []


class FARAConfig:
    """Simplified configuration for FARA OCR system"""
    
    def __init__(self):
        self.input_dir = Path("fara_documents")
        self.output_dir = Path("fara_output") 
        self.log_dir = Path("logs")
        self.tesseract_config = '--oem 3 --psm 6'
        self.supported_formats = ['.pdf']
        self.image_dpi = 300
        self.max_file_size_mb = 50
    
    def create_directories(self):
        """Create necessary directories"""
        for directory in [self.input_dir, self.output_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def validate_environment(self) -> Dict[str, bool]:
        """Validate system environment and dependencies"""
        results = {}
        
        # Check directories
        results['input_dir_exists'] = self.input_dir.exists()
        results['output_dir_writable'] = os.access(self.output_dir, os.W_OK)
        
        # Check Tesseract
        try:
            pytesseract.get_tesseract_version()
            results['tesseract_available'] = True
        except Exception:
            results['tesseract_available'] = False
            
        # Check other dependencies
        try:
            import cv2 # type: ignore
            results['opencv_available'] = True
        except ImportError:
            results['opencv_available'] = False
            
        try:
            from pdf2image import convert_from_path
            results['pdf2image_available'] = True
        except ImportError:
            results['pdf2image_available'] = False
            
        return results

class ImageProcessor:
    """Simplified image preprocessing for better OCR results"""
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Enhanced preprocessing for government forms"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Simple but effective preprocessing
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 10    # Brightness control  
        contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(contrasted, (3, 3), 0)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 8
        )
        
        return Image.fromarray(binary)

class FARAOCRProcessor:
    """Main FARA OCR processing class - simplified version"""
    
    def __init__(self, config: FARAConfig = None):
        self.config = config or FARAConfig()
        self.logger = self._setup_logging()
        self.image_processor = ImageProcessor()
        
        # Create directories
        self.config.create_directories()
        
        # Validate environment
        env_status = self.config.validate_environment()
        self._log_environment_status(env_status)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('FARA_OCR')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.config.log_dir / f"fara_ocr_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _log_environment_status(self, env_status: Dict[str, bool]):
        """Log environment validation results"""
        self.logger.info("Environment Status:")
        for check, status in env_status.items():
            symbol = "[OK]" if status else "[ERROR]"
            self.logger.info(f"  {symbol} {check.replace('_', ' ').title()}: {status}")
    
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.config.image_dpi,
                fmt='jpeg'
            )
            self.logger.info(f"Converted {len(images)} pages from {pdf_path.name}")
            return images
        except Exception as e:
            self.logger.error(f"Failed to convert PDF {pdf_path}: {str(e)}")
            return []
    
    def extract_text_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text from image using Tesseract with multiple configs"""
        try:
            enhanced_image = self.image_processor.enhance_image(image)
            
            # Try multiple OCR configurations
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 4',
                '--oem 3 --psm 3'
            ]
            
            best_text = ""
            best_confidence = 0.0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(enhanced_image, config=config)
                    
                    # Get confidence
                    data = pytesseract.image_to_data(
                        enhanced_image,
                        output_type=pytesseract.Output.DICT,
                        config=config
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence and len(text.strip()) > 0:
                        best_text = text.strip()
                        best_confidence = avg_confidence
                        
                except Exception:
                    continue
            
            return best_text, best_confidence
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return "", 0.0
    
    def extract_fara_filed_responses(self, text: str) -> Dict[str, Any]:
        """Extract only the filed responses, not form boilerplate"""
        data = {
            'registrant_name': "",
            'registration_number': "",
            'receipts_response': "",
            'receipts_explanation': "",
            'filing_date': "",
            'foreign_principals_listed': []
        }
        
        # Extract 1(a) registrant name - the actual filed answer
        registrant_section = self._extract_section_1a_response(text)
        if registrant_section:
            data['registrant_name'] = registrant_section
        
        # Extract registration number from 1(b)
        reg_number = self._extract_registration_number(text)
        if reg_number:
            data['registration_number'] = reg_number
        
        # Extract 14(a) receipts response and details
        receipts_info = self._extract_14a_response_and_details(text)
        data.update(receipts_info)
        
        # Extract filing period
        filing_date = self._extract_filing_date(text)
        if filing_date:
            data['filing_date'] = filing_date
        
        # Extract foreign principals from section 9
        principals = self._extract_section_9_principals(text)
        data['foreign_principals_listed'] = principals
        
        return data
    
    def _extract_section_1a_response(self, text: str) -> str:
        """Extract the actual response to 1(a) Name of Registrant"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if 'name of registrant' in line.lower() and '(b) registration number' in line.lower():
                # Look for the next non-empty line that contains the actual name
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j].strip()
                    # Filter out form text and find actual registrant name
                    if (len(candidate) > 10 and 
                        'primary business address' not in candidate.lower() and
                        not candidate.isdigit() and
                        'po box' not in candidate.lower()):
                        return candidate
        
        return ""
    
    def _extract_registration_number(self, text: str) -> str:
        """Extract registration number from 1(b)"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if 'registration number' in line.lower():
                # Look for 4-digit number in next few lines
                for j in range(i, min(i + 3, len(lines))):
                    numbers = re.findall(r'\b(\d{4})\b', lines[j])
                    if numbers:
                        return numbers[0]
        
        return ""
    
    def _extract_14a_response_and_details(self, text: str) -> Dict[str, Any]:
        """Extract complete 14(a) response including detailed entries if Yes"""
        receipts_data = {
            'receipts_response': "",
            'receipts_explanation': "",
            'receipts_entries': [],
            'receipts_total': ""
        }
        
        # Find 14(a) section
        section_14a = re.search(r'14\.\s*\(a\)\s*RECEIPTS-MONIES(.*?)(?=14\.\s*\(b\)|15\.\s*\(a\)|$)', 
                               text, re.DOTALL | re.IGNORECASE)
        
        if not section_14a:
            return receipts_data
        
        section_text = section_14a.group(1)
        
        # Check for Yes/No checkbox selections
        has_yes = bool(re.search(r'Yes\s*☒|Yes\s*\[X\]|Yes\s*&\]', section_text))
        has_no = bool(re.search(r'No\s*☒|No\s*\[X\]|No\s*&\]', section_text))
        
        if has_yes:
            receipts_data['receipts_response'] = "Yes"
            
            # Extract detailed entries from table format
            entries = self._parse_receipts_table(section_text)
            receipts_data['receipts_entries'] = entries
            
            # Extract total amount
            total_match = re.search(r'Total\s*\$?\s*([0-9,]+\.?\d*)', section_text)
            if total_match:
                receipts_data['receipts_total'] = total_match.group(1)
            
        elif has_no:
            receipts_data['receipts_response'] = "No"
            
            # Look for explanation
            if 'see appendix for response' in section_text.lower():
                receipts_data['receipts_explanation'] = "See Appendix for Response"
                # Get appendix details
                appendix_details = self._extract_appendix_14a_details(text)
                if appendix_details:
                    receipts_data['receipts_explanation'] = f"See Appendix: {appendix_details}"
            else:
                # Look for direct explanation
                explanation_match = re.search(r'if no,?\s*please explain[^:]*:?\s*([^\n]{10,200})', 
                                            section_text, re.IGNORECASE)
                if explanation_match:
                    receipts_data['receipts_explanation'] = explanation_match.group(1).strip()
        
        return receipts_data
    
    def _parse_receipts_table(self, section_text: str) -> List[ReceiptsEntry]:
        """Parse the tabular receipts data from the Item 14(a) section and return list of ReceiptsEntry."""
        entries: List[ReceiptsEntry] = []

        # Try to isolate the table body (everything after the header up to Total or end)
        table_pattern = (
            r'Foreign Principal.*?Date Received.*?From Whom.*?Purpose.*?Amount.*?Subtotal'
            r'(.*?)(?:Total|\$\s*[\d,]+\.?\d*\s*$|$)'
        )
        table_match = re.search(table_pattern, section_text, re.DOTALL | re.IGNORECASE)
        if not table_match:
            return entries

        table_data = table_match.group(1)

        # Normalize and split into non-empty lines
        raw_lines = [re.sub(r'\s+', ' ', l).strip() for l in re.split(r'[\r\n]+', table_data) if l.strip()]

        current_entry = ReceiptsEntry()
        collecting_purpose = False

        for line in raw_lines:
            low = line.lower()

            # Skip header-like fragments, separators or obvious non-data lines
            if (
                'foreign principal' in low or
                'date received' in low or
                'from whom' in low or
                'purpose' in low or
                'amount' in low or
                'subtotal' in low or
                'received by nsd' in low or
                line.count('-') > 8
            ):
                continue

            # If we don't yet have a foreign_principal and the line looks like a name, take it
            if (
                not current_entry.foreign_principal
                and not re.search(r'\$|\d{1,2}/\d{1,2}/\d{4}', line)
                and not re.match(r'^[\d\.\,\/\s-]+$', line)
                and len(line) > 1
            ):
                current_entry.foreign_principal = line
                continue

            # Look for a date on the line
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            if date_match and not current_entry.date_received:
                current_entry.date_received = date_match.group(1)
                # leftover text on same line might be "From Whom"
                remaining = line.replace(date_match.group(1), '').strip()
                if remaining and not current_entry.from_whom and re.search(r'[A-Za-z]', remaining):
                    current_entry.from_whom = remaining
                continue

            # Look for monetary amounts (heuristic)
            amount_match = re.search(r'\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?)', line)
            if amount_match:
                amount_str = amount_match.group(1)
                # If no main amount yet, set it; otherwise assign to subtotal if empty
                if not current_entry.amount:
                    current_entry.amount = amount_str
                elif not current_entry.subtotal:
                    current_entry.subtotal = amount_str
                continue

            # Collect purpose text (often multi-line). Use keywords to start purpose capture.
            if (
                current_entry.foreign_principal
                and not collecting_purpose
                and any(k in low for k in ('money', 'pay', 'services', 'consult', 'fee', 'compens'))
            ):
                current_entry.purpose = line
                collecting_purpose = True
                continue
            elif collecting_purpose and not amount_match:
                # append additional lines to purpose until we hit an amount or next entry
                current_entry.purpose = (current_entry.purpose + " " + line).strip()
                continue

            # If this line doesn't match any of the above, it might signal the start of a new entry
            # Save current entry if it's sufficiently complete, then try to treat this line as the next entry start
            if (
                current_entry.foreign_principal
                and current_entry.amount
                and len([f for f in [current_entry.date_received, current_entry.from_whom, current_entry.purpose] if f]) >= 2
            ):
                entries.append(current_entry)
                current_entry = ReceiptsEntry()
                collecting_purpose = False

                # Re-evaluate current line as start of the next entry (if it looks like a name)
                if (
                    not re.search(r'\$|\d{1,2}/\d{1,2}/\d{4}', line)
                    and not re.match(r'^[\d\.\,\/\s-]+$', line)
                    and len(line) > 1
                ):
                    current_entry.foreign_principal = line
                    continue

        # After the loop, append last entry if it looks complete
        if current_entry.foreign_principal and current_entry.amount:
            entries.append(current_entry)

        return entries
        
    def _extract_appendix_14a_details(self, text: str) -> str:
        """Extract specific appendix details for Item 14(a)"""
        # Look for "Response to Item 14(a)-Description"
        appendix_pattern = r'Response to Item 14\(a\)-Description(.*?)(?=Response to Item|Received by NSD/FARA|$)'
        appendix_match = re.search(appendix_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if appendix_match:
            appendix_text = appendix_match.group(1).strip()
            
            # Extract the actual responses (not the form headers)
            responses = []
            lines = appendix_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if (line and 
                    'none due' in line.lower() and 
                    ':' in line):
                    # Extract principal and response
                    responses.append(line)
            
            return '; '.join(responses) if responses else appendix_text[:200]
        
        return ""
    
    def _extract_filing_date(self, text: str) -> str:
        """Extract filing date from form header"""
        # Look for "For 6 Month Period Ending"
        period_match = re.search(r'For 6 Month Period Ending\s+(\d{1,2}/\d{1,2}/\d{4})', text, re.IGNORECASE)
        if period_match:
            return period_match.group(1)
        
        return ""
    
    def _extract_section_9_principals(self, text: str) -> List[str]:
        """Extract foreign principals listed in section 9"""
        principals = []
        
        # Find section 9
        section_9_match = re.search(r'9\.\s*In addition to those named.*?list the foreign principal\(s\)(.*?)(?=10\.|$)', 
                                   text, re.DOTALL | re.IGNORECASE)
        
        if section_9_match:
            section_text = section_9_match.group(1)
            lines = section_text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Filter out form text and extract actual principal names
                if (len(line) > 3 and 
                    line not in ['', 'during this 6 month reporting period.'] and
                    'received by nsd' not in line.lower() and
                    'page' not in line.lower()):
                    principals.append(line)
        
        return principals
    
    def validate_data(self, fara_data: FARAData) -> float:
        """Validate filed response data quality"""
        score = 0.0
        total_checks = 5
        
        if fara_data.registrant_name and len(fara_data.registrant_name) > 5:
            score += 1.0
        if fara_data.registration_number and fara_data.registration_number.isdigit():
            score += 1.0
        if fara_data.filing_date and '/' in fara_data.filing_date:
            score += 1.0
        if fara_data.receipts_response or fara_data.receipts_explanation:
            score += 1.0
        if fara_data.foreign_principals_listed and len(fara_data.foreign_principals_listed) > 0:
            score += 1.0
        
        return (score / total_checks) * 100.0
    
    def process_single_document(self, pdf_path: Path) -> Optional[FARAData]:
        """Process a single FARA PDF document"""
        self.logger.info(f"Processing document: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path)
            if not images:
                return None
            
            # Extract text from all pages
            all_text = []
            total_confidence = 0.0
            
            for i, image in enumerate(images):
                text, confidence = self.extract_text_from_image(image)
                all_text.append(text)
                total_confidence += confidence
                self.logger.debug(f"Page {i+1}: {len(text)} characters, {confidence:.1f}% confidence")
            
            # Combine all text
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            # Extract FARA filed responses (not form boilerplate)
            extracted_data = self.extract_fara_filed_responses(combined_text)
            
            # Create FARA data object
            fara_data = FARAData(
                registrant_name=extracted_data.get('registrant_name', ''),
                registration_number=extracted_data.get('registration_number', ''),
                receipts_response=extracted_data.get('receipts_response', ''),
                receipts_explanation=extracted_data.get('receipts_explanation', ''),
                receipts_entries=extracted_data.get('receipts_entries', []),
                receipts_total=extracted_data.get('receipts_total', ''),
                filing_date=extracted_data.get('filing_date', ''),
                foreign_principals_listed=extracted_data.get('foreign_principals_listed', []),
                confidence_score=avg_confidence,
                source_file=str(pdf_path),
                processing_date=datetime.now().isoformat(),
                raw_text=combined_text
            )
            
            # Validate and update confidence
            validation_score = self.validate_data(fara_data)
            fara_data.confidence_score = min(avg_confidence, validation_score)
            
            self.logger.info(f"Extraction complete: {validation_score:.1f}% field coverage")
            return fara_data
            
        except Exception as e:
            self.logger.error(f"Failed to process {pdf_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def process_batch(self, limit: int = None) -> List[FARAData]:
        """Process all FARA documents in input directory"""
        self.logger.info("Starting batch processing")
        
        # Find PDF files
        pdf_files = list(self.config.input_dir.glob("*.pdf"))
        if limit:
            pdf_files = pdf_files[:limit]
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            # Check file size
            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                self.logger.warning(f"Skipping {pdf_file.name}: {file_size_mb:.1f}MB exceeds limit")
                continue
                
            fara_data = self.process_single_document(pdf_file)
            if fara_data:
                results.append(fara_data)
        
        self.logger.info(f"Batch processing complete: {len(results)} documents processed")
        return results
    
    def save_results(self, results: List[FARAData], format_type: str = 'json'):
        """Save processing results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type.lower() == 'json':
            output_file = self.config.output_dir / f"fara_data_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(result) for result in results], f, indent=2, ensure_ascii=False)
        
        elif format_type.lower() == 'csv':
            output_file = self.config.output_dir / f"fara_data_{timestamp}.csv"
            if results:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                    writer.writeheader()
                    for result in results:
                        row_data = asdict(result)
                        row_data.pop('raw_text', None)
                        writer.writerow(row_data)
        
        self.logger.info(f"Results saved to: {output_file}")
        return output_file

def main():
    """Main execution function"""
    # Initialize system
    config = FARAConfig()
    processor = FARAOCRProcessor(config)
    
    # Check if input directory exists and has files
    if not config.input_dir.exists():
        print(f"\nERROR: Input directory '{config.input_dir}' not found!")
        print("Please create the directory and add FARA PDF files to process.")
        print(f"mkdir -p {config.input_dir}")
        return
    
    pdf_files = list(config.input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\nERROR: No PDF files found in '{config.input_dir}'")
        print("Please add FARA PDF documents to process.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files to process")
    
    # Process documents
    try:
        results = processor.process_batch(limit=3)
        
        if results:
            print(f"\nSuccessfully processed {len(results)} documents")
            
            # Save results
            json_file = processor.save_results(results, 'json')
            csv_file = processor.save_results(results, 'csv')
            
            # Print summary
            print("\nProcessing Summary:")
            for result in results:
                print(f"  File: {Path(result.source_file).name}")
                print(f"     1(a) Registrant: {result.registrant_name or 'Not found'}")
                print(f"     1(b) Reg Number: {result.registration_number or 'Not found'}")
                print(f"     14(a) Response: {result.receipts_response or 'Not found'}")
                
                if result.receipts_response == "Yes" and result.receipts_entries:
                    print(f"     14(a) Entries: {len(result.receipts_entries)} receipts found")
                    for entry in result.receipts_entries[:2]:  # Show first 2 entries
                        print(f"       - {entry.foreign_principal}: ${entry.amount} on {entry.date_received}")
                        if entry.purpose:
                            print(f"         Purpose: {entry.purpose[:60]}...")
                    if result.receipts_total:
                        print(f"     14(a) Total: ${result.receipts_total}")
                
                elif result.receipts_response == "No" and result.receipts_explanation:
                    print(f"     14(a) Explanation: {result.receipts_explanation[:80]}...")
                
                print(f"     Filing Period: {result.filing_date or 'Not found'}")
                if result.foreign_principals_listed:
                    valid_principals = [p for p in result.foreign_principals_listed 
                                      if len(p) > 5 and 'whom the registrant' not in p.lower()]
                    if valid_principals:
                        print(f"     Section 9 Principals: {len(valid_principals)} listed")
                        for principal in valid_principals[:3]:  # Show first 3
                            print(f"       - {principal}")
                print(f"     Confidence: {result.confidence_score:.1f}%")
                print()
        else:
            print("\nERROR: No documents were successfully processed")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nERROR: Processing failed: {str(e)}")
        logging.getLogger('FARA_OCR').error(traceback.format_exc())

    main()