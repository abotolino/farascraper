import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.common.logger import get_logger
from src.common.exceptions import FARAError, AuthenticationError, DocumentNotFoundError
from config.settings import settings

logger = get_logger(__name__)


class FARADocumentScraper:
    def __init__(self, base_url=None, username=None, password=None):
        self.base_url = base_url or settings.fara_base_url
        self.username = username or settings.fara_username
        self.password = password or settings.fara_password
        self.session = requests.Session()
        
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Use pipeline download directory
        self.download_dir = settings.raw_documents
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_logged_in = False
        logger.info("FARA scraper initialized", download_dir=str(self.download_dir))
    
    def login(self, login_url=None):
        """Handle login to the FARA site"""
        if not self.username or not self.password:
            raise AuthenticationError("Username and password required for login")
        
        # Use provided login URL or default to the known FARA login URL
        if not login_url:
            login_url = f"{self.base_url}/users/sign_in"
        
        try:
            logger.info("Attempting to log in", url=login_url)
            
            # Get the login page to find the form and any CSRF tokens
            response = self.session.get(login_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for login form - Rails/Devise typically uses a form with user fields
            login_form = soup.find('form')
            if not login_form:
                raise AuthenticationError("Could not find login form")
            
            # Get form action URL
            form_action = login_form.get('action', '')
            if form_action:
                login_post_url = urljoin(login_url, form_action)
            else:
                login_post_url = login_url
            
            # Prepare login data
            login_data = {}
            
            # Look for CSRF token (Rails uses authenticity_token)
            csrf_input = soup.find('input', {'name': re.compile(r'authenticity_token|csrf|_token', re.I)})
            if csrf_input:
                csrf_name = csrf_input.get('name')
                csrf_value = csrf_input.get('value')
                login_data[csrf_name] = csrf_value
                logger.info("Found CSRF token", token_name=csrf_name)
            
            # Look for hidden input fields in the form
            for hidden_input in login_form.find_all('input', type='hidden'):
                name = hidden_input.get('name')
                value = hidden_input.get('value')
                if name and value:
                    login_data[name] = value
            
            # Rails/Devise typically uses user[email] and user[password] format
            username_field = 'user[email]'
            password_field = 'user[password]'
            
            # Try to detect the correct field names from the form
            for input_field in login_form.find_all('input'):
                field_name = input_field.get('name', '')
                field_type = input_field.get('type', '').lower()
                
                if field_type == 'email' or 'email' in field_name.lower():
                    username_field = field_name
                elif 'user' in field_name.lower() and ('email' in field_name.lower() or 'login' in field_name.lower()):
                    username_field = field_name
                elif field_type == 'password':
                    password_field = field_name
            
            login_data[username_field] = self.username
            login_data[password_field] = self.password
            
            # Add common Rails fields
            login_data['utf8'] = '✓'
            login_data['commit'] = 'Log in'
            
            logger.info("Submitting login form", url=login_post_url)
            
            # Submit login form
            response = self.session.post(login_post_url, data=login_data, allow_redirects=True)
            response.raise_for_status()
            
            # Check if login was successful
            final_url = response.url.lower()
            
            # If we're still on the sign_in page, login likely failed
            if 'sign_in' in final_url:
                raise AuthenticationError("Login failed - still on sign_in page")
            
            # Look for success indicators in the response content
            success_indicators = [
                'document', 'logout', 'sign out', 'dashboard',
                'waiting to be processed', 'fara document importer'
            ]
            
            response_text = response.text.lower()
            has_success = any(indicator in response_text for indicator in success_indicators)
            
            if has_success:
                logger.info("Login successful", final_url=response.url)
                self.is_logged_in = True
                return True
            else:
                raise AuthenticationError("Login verification failed")
                
        except requests.RequestException as e:
            raise AuthenticationError(f"Login request failed: {e}")
    
    def get_page_content(self, url):
        """Get the HTML content of the page"""
        if not self.is_logged_in:
            logger.warning("Not logged in, attempting authentication")
            self.login()
            
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Check if we got redirected to login (session expired)
            if 'sign_in' in response.url.lower():
                logger.warning("Session expired, re-authenticating")
                self.is_logged_in = False
                self.login()
                response = self.session.get(url)
                response.raise_for_status()
                
            return response.text
        except requests.RequestException as e:
            raise FARAError(f"Error fetching page content: {e}")
    
    def extract_waiting_documents(self, html_content):
        """Extract document links from the 'documents waiting to be processed' section"""
        soup = BeautifulSoup(html_content, 'html.parser')
        waiting_docs = []
        
        # Find all columns
        columns = soup.find_all('div', class_='col')
        
        for col in columns:
            # Look for the header that indicates "documents waiting to be processed"
            h3_tag = col.find('h3', string=lambda text: text and 'documents waiting to be processed' in text.lower())
            
            if h3_tag:
                logger.info("Found 'documents waiting to be processed' section")
                
                # Find the scrollable container in this column
                scrollable_container = col.find('div', class_='scrollable-container')
                
                if scrollable_container:
                    # Find all document forms/buttons in this section
                    doc_forms = scrollable_container.find_all('form', action='/supplemental_statements')
                    
                    for form in doc_forms:
                        # Find the submit button which contains the document name
                        submit_button = form.find('input', {'type': 'submit'})
                        if submit_button:
                            doc_name = submit_button.get('value', '').strip()
                            
                            # Get the document ID from the hidden input
                            doc_id_input = form.find('input', {'name': 'supplemental_statement[document_id]'})
                            if doc_id_input:
                                doc_id = doc_id_input.get('value')
                                
                                if doc_name.endswith('.pdf') and doc_id:
                                    csrf_token_input = form.find('input', {'name': 'authenticity_token'})
                                    csrf_token = csrf_token_input.get('value') if csrf_token_input else None
                                    
                                    waiting_docs.append({
                                        'name': doc_name,
                                        'document_id': doc_id,
                                        'form': form,
                                        'csrf_token': csrf_token
                                    })
                break
        
        logger.info("Found waiting documents", count=len(waiting_docs))
        return waiting_docs
    
    def get_document_page_url(self, doc_info):
        """Submit the form to get the document page URL"""
        try:
            action = doc_info['form'].get('action', '/supplemental_statements')
            form_url = urljoin(self.base_url, action)
            
            # Prepare form data
            form_data = {
                'utf8': '✓',
                'supplemental_statement[document_id]': doc_info['document_id']
            }
            
            # Add CSRF token if available
            if doc_info['csrf_token']:
                form_data['authenticity_token'] = doc_info['csrf_token']
            
            # Submit the form
            response = self.session.post(form_url, data=form_data, allow_redirects=True)
            response.raise_for_status()
            
            return response.url
            
        except requests.RequestException as e:
            raise FARAError(f"Error submitting form for document {doc_info['name']}: {e}")
    
    def get_pdf_download_url(self, doc_url):
        """Get the actual PDF download URL from the document page"""
        try:
            response = self.session.get(doc_url)
            response.raise_for_status()
            
            # Check if we got redirected to login
            if 'sign_in' in response.url.lower():
                raise AuthenticationError(f"Redirected to login when accessing {doc_url}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for S3 URLs in <object> tags (this is what FARA uses!)
            pdf_links = []
            
            object_tag = soup.find('object')
            if object_tag and object_tag.get('data'):
                s3_url = object_tag['data']
                if 's3.amazonaws.com' in s3_url and s3_url.endswith('.pdf'):
                    pdf_links.append(s3_url)
            
            # Look for S3 URLs in any data attributes or src attributes
            for tag in soup.find_all(['object', 'embed', 'iframe']):
                for attr in ['data', 'src']:
                    url = tag.get(attr)
                    if url and 's3.amazonaws.com' in url and url.endswith('.pdf'):
                        pdf_links.append(url)
            
            # Search for S3 URLs in the entire HTML content
            if not pdf_links:
                s3_pattern = r'https://s3\.amazonaws\.com/[^"\s]*\.pdf'
                s3_matches = re.findall(s3_pattern, response.text)
                pdf_links.extend(s3_matches)
            
            if pdf_links:
                return pdf_links[0]
            else:
                raise DocumentNotFoundError(f"No PDF download URL found for {doc_url}")
            
        except requests.RequestException as e:
            raise FARAError(f"Error getting PDF download URL: {e}")
    
    def download_document(self, doc_info) -> Dict[str, Any]:
        """Download a single document - integrated with pipeline"""
        try:
            logger.info("Processing document", name=doc_info['name'])
            
            # Get the document page URL by submitting the form
            doc_page_url = self.get_document_page_url(doc_info)
            
            # Get the PDF download URL from the document page
            pdf_url = self.get_pdf_download_url(doc_page_url)
            
            # Sanitize filename
            safe_filename = self.sanitize_filename(doc_info['name'])
            filepath = self.download_dir / safe_filename
            
            # Check if file already exists
            if filepath.exists():
                logger.info("File already exists, skipping", filename=safe_filename)
                return {
                    "success": True,
                    "file_path": str(filepath),
                    "file_size": filepath.stat().st_size,
                    "url": pdf_url,
                    "skipped": True
                }
            
            # Download the PDF
            logger.info("Downloading PDF", url=pdf_url)
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status()
            
            # Check if we got redirected to login page
            if 'sign_in' in response.url.lower():
                raise AuthenticationError("Redirected to login page during download")
            
            # Read and validate content
            content_chunks = []
            first_chunk = None
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    if first_chunk is None:
                        first_chunk = chunk
                        # Check PDF magic number
                        if not chunk.startswith(b'%PDF'):
                            raise DocumentNotFoundError(f"Downloaded content is not a PDF")
                    content_chunks.append(chunk)
            
            # Write the file
            with open(filepath, 'wb') as f:
                for chunk in content_chunks:
                    f.write(chunk)
            
            # Verify the file
            file_size = filepath.stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                raise DocumentNotFoundError(f"PDF file is unusually small ({file_size} bytes)")
            
            logger.info("Document downloaded successfully", 
                       filename=safe_filename, file_size=file_size)
            
            return {
                "success": True,
                "file_path": str(filepath),
                "file_size": file_size,
                "url": pdf_url,
                "document_id": doc_info['document_id']
            }
            
        except Exception as e:
            logger.error("Document download failed", 
                        name=doc_info.get('name', 'unknown'), error=str(e))
            return {
                "success": False,
                "error": str(e),
                "document_id": doc_info.get('document_id')
            }
    
    def sanitize_filename(self, filename):
        """Sanitize filename for safe saving"""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        return filename
    
    def get_waiting_documents_list(self, main_page_url=None) -> List[Dict[str, Any]]:
        """Get list of all waiting documents"""
        if not main_page_url:
            main_page_url = self.base_url
            
        html_content = self.get_page_content(main_page_url)
        return self.extract_waiting_documents(html_content)
