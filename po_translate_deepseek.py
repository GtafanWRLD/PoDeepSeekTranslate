import os
import json
import requests
import time
import threading
import re
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import langdetect
from langdetect import detect
import requests 
from datetime import datetime
import subprocess


CONFIG_FILE = "config.json"
KEY_FILE = "deepseek_key.txt"

# Optimized default config for faster translations
DEFAULT_CONFIG = {
    "batch_size": 50,      # Increased from 20
    "max_workers": 12,     # Increased from 6
    "api_delay": 0.05,     # Reduced from 0.1
    "max_retries": 3,
    "min_text_length": 2,
    "max_tokens": 6000,    # Increased from 4000
    "temperature": 0.05    # Reduced for more consistent translations
}

LANGUAGE_CODES = {
    "EN": "English",
    "ES": "Spanish", 
    "FR": "French",
    "DE": "German",
}

# Language detection mapping (langdetect codes to our codes)
LANGDETECT_TO_CODE = {
    'en': 'EN',
    'es': 'ES', 
    'fr': 'FR',
    'de': 'DE',
}

COORD_MAP = {
    'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ж':'Zh','З':'Z','И':'I',
    'К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P','Р':'R','С':'S','Т':'T',
    'У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'Ch','Ш':'Sh','Щ':'Shch','Ы':'Y','Э':'E','Ю':'Yu','Я':'Ya'
}

progress_lock = threading.Lock()
api_semaphore = None

class DeepseekClient:
    """Enhanced Deepseek AI API Client for translation"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def translate(self, texts, source_language="auto", target_language="EN"):
        """
        Enhanced translation method with better batch processing
        """
        if not texts:
            return []
        
        target_lang_name = LANGUAGE_CODES.get(target_language, target_language)
        
        # Create optimized batch format
        batch_items = []
        for i, text in enumerate(texts):
            batch_items.append(f"[{i+1}] {text}")
        
        batch_text = "\n\n".join(batch_items)
        
        # Enhanced system prompt for better performance
        system_prompt = (
            f"Translate to {target_lang_name}. CRITICAL RULES:\n"
            "1. PRESERVE ALL variables: %(var)s, %d, {0}, {{key}}, etc.\n"
            "2. PRESERVE ALL formatting: \\n, \\t, spacing, line breaks\n"
            "3. PRESERVE ALL special chars and symbols\n"
            f"4. Output natural {target_lang_name} for gaming/software content\n"
            "5. Keep [N] numbering format\n"
            "6. Military terms: танк=tank, броня=armor, урон=damage\n"
            "7. NO extra text or explanations\n"
            "8. If text has \\n keep as literal \\n characters\n\n"
            "Output format: [N] translated_text"
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": batch_text
                }
            ],
            "temperature": config.get('temperature', 0.05),
            "max_tokens": config.get('max_tokens', 6000),
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=180
        )
        
        if response.status_code != 200:
            raise Exception(f"Deepseek API error: {response.status_code} - {response.text}")
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Enhanced parsing with multiple fallback methods
        translations = self._parse_translations(content, texts)
        
        return translations
    
    def _parse_translations(self, content, original_texts):
        """Enhanced translation parsing with multiple fallback methods"""
        translations = []
        
        # Method 1: Parse [N] format
        lines = content.split('\n')
        parsed = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for [N] pattern
            match = re.match(r'\[(\d+)\]\s*(.*)', line)
            if match:
                idx = int(match.group(1)) - 1
                translation = match.group(2).strip()
                if 0 <= idx < len(original_texts):
                    parsed[idx] = translation
        
        # Method 2: Fallback - split by double newlines and match by position
        if len(parsed) < len(original_texts):
            sections = re.split(r'\n\s*\n', content)
            for i, section in enumerate(sections):
                if i < len(original_texts) and i not in parsed:
                    # Clean up the section
                    cleaned = re.sub(r'^\[\d+\]\s*', '', section.strip())
                    if cleaned:
                        parsed[i] = cleaned
        
        # Build final translations list
        for i in range(len(original_texts)):
            if i in parsed and parsed[i].strip():
                translations.append(parsed[i].strip())
            else:
                # Fallback to original
                translations.append(original_texts[i])
        
        return translations

def ensure_and_load_deepseek_key():
    """
    Ensures deepseek_key.txt exists and contains a key. Opens it in notepad if needed.
    Waits for user to paste/save, then presses Enter to continue.
    Returns the API key string.
    """
    keyfile = "deepseek_key.txt"
    while True:
        if not os.path.exists(keyfile) or not open(keyfile, encoding="utf-8").read().strip():
            # Create empty file if missing
            if not os.path.exists(keyfile):
                with open(keyfile, "w", encoding="utf-8") as f:
                    f.write("")
            print(f"\n🔑 Please paste your Deepseek API key into the file: {os.path.abspath(keyfile)}")
            print("Opening it in your default text editor...")
            # Cross-platform open
            try:
                if sys.platform.startswith('win'):
                    os.startfile(keyfile)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', keyfile])
                else:
                    subprocess.Popen(['xdg-open', keyfile])
            except Exception as e:
                print(f"⚠️  Could not open editor automatically: {e}")
            input("After pasting & saving your key, press Enter to continue...")
        # Try to read the key
        try:
            key = open(keyfile, encoding="utf-8").read().strip()
            if key:
                return key
            else:
                print("⚠️  No key detected in the file. Please paste and save your key, then press Enter.")
        except Exception as e:
            print(f"❌ Error reading {keyfile}: {e}")
            sys.exit(1)


#checkpoint 1

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                config = DEFAULT_CONFIG.copy()
                config.update(loaded)
                return config
        except Exception:
            print(f"❌ Invalid JSON in '{CONFIG_FILE}'. Using defaults.")
    return None

def save_config(config):
    """Save configuration to JSON file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Could not save settings: {e}")

def interactive_setup():
    print("🚀 PO Translator - First-Time Setup")
    config = DEFAULT_CONFIG.copy()
    
    print(f"\nOptimized defaults for faster translation:")
    print(f"  Batch size: {config['batch_size']} (more texts per API call)")
    print(f"  Max workers: {config['max_workers']} (more parallel requests)")
    print(f"  API delay: {config['api_delay']}s (faster rate)")
    print(f"  Max tokens: {config['max_tokens']} (larger responses)")
    
    if input("\nUse optimized defaults? (Y/n): ").lower().startswith('n'):
        try:
            config['batch_size'] = int(input(f"Batch size [{config['batch_size']}]: ") or config['batch_size'])
            config['max_workers'] = int(input(f"Max workers [{config['max_workers']}]: ") or config['max_workers'])
            config['api_delay'] = float(input(f"API delay (s) [{config['api_delay']}]: ") or config['api_delay'])
            config['max_retries'] = int(input(f"Max retries [{config['max_retries']}]: ") or config['max_retries'])
            config['min_text_length'] = int(input(f"Min text length [{config['min_text_length']}]: ") or config['min_text_length'])
        except ValueError:
            print("❌ Invalid input; using defaults.")
    
    save_config(config)
    return config

def read_po_file(path):
    try:
        import chardet
        raw = open(path, 'rb').read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        text = raw.decode(enc, errors='replace').splitlines()
        return text, enc
    except ImportError:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().splitlines()
        return text, 'utf-8'

def detect_language_enhanced(text):
    """
    Enhanced language detection with better accuracy
    Returns language code (EN, RU, PL, etc.) or 'unknown'
    """
    if not text or len(text.strip()) < 3:
        return 'unknown'
    
    try:
        # Clean text for detection - remove variables and formatting but keep more text
        clean_text = re.sub(r'%\([^)]+\)s|%\w+|{\w+}|\\[ntr]', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Remove special chars but keep letters
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 3:
            return 'unknown'
        
        # Use langdetect with confidence check
        detected = detect(clean_text)
        
        # Convert langdetect code to our format
        our_code = LANGDETECT_TO_CODE.get(detected, 'unknown')
        
        # Additional checks for common cases
        if our_code == 'unknown':
            # Check for Cyrillic (Russian/Ukrainian)
            if re.search(r'[\u0400-\u04FF]', text):
                # More specific detection between RU and UK
                if re.search(r'[іїєґ]', text):  # Ukrainian specific chars
                    return 'UK'
                else:
                    return 'RU'
            # Check for Polish specific characters
            elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                return 'PL'
            # Check for basic Latin
            elif re.search(r'[a-zA-Z]', text):
                return 'EN'  # Default to English for Latin script
        
        return our_code
        
    except Exception as e:
        # Fallback to regex-based detection
        if re.search(r'[\u0400-\u04FF]', text):
            if re.search(r'[іїєґ]', text):
                return 'UK'
            else:
                return 'RU'
        elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
            return 'PL'
        elif re.search(r'[a-zA-Z]', text):
            return 'EN'
        return 'unknown'

def should_translate_content(content, target_language):
    """
    Enhanced logic to determine if content should be translated
    Returns (should_translate: bool, reason: str, detected_lang: str)
    """
    if not content or not content.strip():
        return False, 'empty', 'unknown'
    
    # Check minimum length
    if len(content.strip()) < config.get('min_text_length', 2):
        return False, 'too_short', 'unknown'
    
    # Detect language
    detected_lang = detect_language_enhanced(content)
    
    # Check if it's coordinate format
    if is_coordinate_format(content):
        return False, 'coordinate', detected_lang
    
    # Check for patterns that shouldn't be translated
    skip_patterns = [
        r'^[A-Z]\d+$',  # Coordinate patterns
        r'^\d+$',       # Pure numbers
        r'^[.,:;!?]+$', # Pure punctuation
        r'^[A-Za-z0-9_]+\.(png|jpg|jpeg|gif|svg|wav|mp3|ogg)$',  # File names
        r'^#[0-9A-Fa-f]{6}$',  # Hex colors
        r'^\w+://',     # URLs
        r'^[A-Za-z0-9_]+$',  # Single words that might be identifiers
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, content.strip()):
            return False, 'skip_pattern', detected_lang
    
    # Main logic: translate if detected language is different from target
    if detected_lang == 'unknown':
        # If we can't detect, assume it needs translation (conservative approach)
        return True, 'unknown_assume_translate', detected_lang
    elif detected_lang == target_language:
        # Same language as target - don't translate
        return False, 'same_as_target', detected_lang
    else:
        # Different language - translate it
        return True, 'different_language', detected_lang

def is_coordinate_format(text):
    """Check if text is a coordinate format like A1, B2, etc."""
    if not text or len(text.strip()) > 10:
        return False
    
    text = text.strip()
    
    # Check for simple coordinate patterns
    if re.match(r'^[A-Z]\d+$', text):
        return True
    
    # Check for Cyrillic coordinates
    if re.match(r'^[А-Я]\d+$', text):
        return True
    
    return False

def translate_coordinate(coord_text):
    """Transliterate Cyrillic coordinates to Latin"""
    if not coord_text:
        return coord_text
    
    result = ""
    for char in coord_text:
        if char in COORD_MAP:
            result += COORD_MAP[char]
        else:
            result += char
    
    return result

#checkpoint 2

def escape_po(text):
    """Properly escape text for PO format - FIXED to handle Russian quotes properly"""
    if not text:
        return ""
    
    escaped = text.replace('\\', '\\\\')  # Escape backslashes first
    escaped = escaped.replace('"', '\\"')  # Then escape ASCII double quotes only
    
    return escaped

def unescape_po(text):
    """Unescape PO format text back to original"""
    if not text:
        return ""
    
    # Reverse the escaping process
    unescaped = text.replace('\\"', '"')  # Unescape quotes first
    unescaped = unescaped.replace('\\\\', '\\')  # Then unescape backslashes
    
    return unescaped

def format_po_string(text, indent=""):
    """Format text for PO file with FIXED multiline handling"""
    if not text:
        return f'{indent}msgstr ""'
    
    # Check if text contains actual newlines (not \n literals)
    has_real_newlines = '\n' in text and not text.replace('\\n', '').find('\n') == -1
    
    if has_real_newlines:
        # Handle actual multiline content
        lines = text.split('\n')
        result = [f'{indent}msgstr ""']
        for i, line in enumerate(lines):
            escaped_line = escape_po(line)
            if i == len(lines) - 1:
                # Last line - don't add \n unless original had it
                result.append(f'{indent}"{escaped_line}"')
            else:
                result.append(f'{indent}"{escaped_line}\\n"')
        return '\n'.join(result)
    else:
        # Single line or text with \n literals - keep as is
        escaped = escape_po(text)
        return f'{indent}msgstr "{escaped}"'

#checkpoint 3

def find_msgstr_blocks(lines, target_language):
    """Enhanced msgstr block detection with plural form support"""
    blocks = []
    i = 0
    idx = 0
    stats = {
        'empty': 0, 'too_short': 0, 'same_as_target': 0, 
        'coordinate': 0, 'skip_pattern': 0, 'different_language': 0,
        'unknown_assume_translate': 0
    }
    lang_stats = {}
    
    while i < len(lines):
        # Check if this is a plural form block
        has_plural = False
        if i > 0 and 'msgid_plural' in lines[i-1]:
            has_plural = True
        
        if lines[i].lstrip().startswith('msgstr') and not has_plural:
            # Regular singular msgstr
            start = i
            original = []
            line = lines[i].lstrip()
            
            # Store the original indentation
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            
            if line.strip() == 'msgstr ""':
                # Multiline msgstr
                i += 1
                while i < len(lines) and lines[i].strip().startswith('"') and lines[i].strip().endswith('"'):
                    content = lines[i].strip()[1:-1]  # Remove quotes
                    original.append(content)
                    i += 1
            else:
                # Single line msgstr
                content = line.partition(' ')[2].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]  # Remove quotes
                original.append(content)
                i += 1
            
            # Join content and unescape for analysis
            content = ''.join(original)
            # Unescape for language detection
            analysis_content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            # Enhanced content analysis
            should_translate, reason, detected_lang = should_translate_content(analysis_content, target_language)
            
            # Update statistics
            stats[reason] = stats.get(reason, 0) + 1
            if detected_lang != 'unknown':
                lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
            
            blocks.append({
                'start': start, 
                'end': i-1, 
                'idx': idx, 
                'content': content,  # Keep original escaped format
                'original_unescaped': analysis_content,  # Store unescaped for translation
                'reason': reason,
                'should_translate': should_translate,
                'detected_lang': detected_lang,
                'indent': indent,
                'is_multiline': line.strip() == 'msgstr ""',
                'is_plural': False
            })
            
            idx += 1
            
        elif lines[i].lstrip().startswith('msgstr['):
            # Plural form - handle multiple msgstr[n] entries
            start = i
            plural_forms = []
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            
            # Collect all msgstr[n] entries
            while i < len(lines) and lines[i].lstrip().startswith('msgstr['):
                line = lines[i].lstrip()
                form_num = int(line.split('[')[1].split(']')[0])
                
                original = []
                if line.strip().endswith('""'):
                    # Multiline msgstr[n]
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith('"') and lines[i].strip().endswith('"'):
                        content = lines[i].strip()[1:-1]  # Remove quotes
                        original.append(content)
                        i += 1
                else:
                    # Single line msgstr[n]
                    content = line.partition(' ')[2].strip()
                    if content.startswith('"') and content.endswith('"'):
                        content = content[1:-1]  # Remove quotes
                    original.append(content)
                    i += 1
                
                plural_forms.append({
                    'form': form_num,
                    'content': ''.join(original)
                })
            
            # Analyze the first plural form for translation decision
            if plural_forms:
                first_content = plural_forms[0]['content']
                analysis_content = first_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                should_translate, reason, detected_lang = should_translate_content(analysis_content, target_language)
                
                # Update statistics
                stats[reason] = stats.get(reason, 0) + 1
                if detected_lang != 'unknown':
                    lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
                
                blocks.append({
                    'start': start,
                    'end': i-1,
                    'idx': idx,
                    'content': first_content,  # Use first form for translation
                    'original_unescaped': analysis_content,
                    'plural_forms': plural_forms,
                    'reason': reason,
                    'should_translate': should_translate,
                    'detected_lang': detected_lang,
                    'indent': indent,
                    'is_multiline': False,  # Will be determined per form
                    'is_plural': True
                })
                
                idx += 1
        else:
            i += 1
    
    # Print enhanced statistics
    total_blocks = len(blocks)
    translatable = sum(1 for b in blocks if b['should_translate'])
    
    print(f"📊 Content Analysis (Target: {LANGUAGE_CODES.get(target_language, target_language)}):")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Will translate: {translatable}")
    print(f"   Will skip: {total_blocks - translatable}")
    
    print(f"\n📈 Detected languages:")
    for lang, count in sorted(lang_stats.items(), key=lambda x: x[1], reverse=True):
        lang_name = LANGUAGE_CODES.get(lang, lang)
        print(f"   {lang_name}: {count}")
    
    print(f"\n📋 Skip reasons:")
    for reason, count in stats.items():
        if count > 0 and reason != 'different_language' and reason != 'unknown_assume_translate':
            print(f"   {reason.replace('_', ' ').title()}: {count}")
    
    return blocks

#checkpoint 4

def translate_batch(batch, batch_id, progress, target_language):
    """Enhanced batch translation with better language handling"""
    if not batch:
        return {}
    
    # Separate content by type
    coords = [b for b in batch if b['reason'] == 'coordinate']
    translatable = [b for b in batch if b['should_translate']]
    skippable = [b for b in batch if not b['should_translate']]
    
    results = {}
    
    # Handle coordinates (no API call)
    for b in coords:
        results[b['idx']] = translate_coordinate(b['content'])
    
    # Skip content that doesn't need translation (no API call)
    # IMPORTANT: Keep original content as-is without re-escaping
    for b in skippable:
        results[b['idx']] = b['content']  # Keep original escaped format
    
    # Only make API call if there's translatable content
    if translatable:
        with api_semaphore:
            time.sleep(config['api_delay'])
            # Use unescaped content for translation
            texts = [b['original_unescaped'] for b in translatable]
            
            for attempt in range(config['max_retries'] + 1):
                try:
                    translations = client.translate(
                        texts=texts,
                        source_language='auto',
                        target_language=target_language
                    )
                    
                    if len(translations) != len(texts):
                        raise Exception(f"Translation count mismatch: got {len(translations)}, expected {len(texts)}")
                    
                    for i, translation in enumerate(translations):
                        if translation and translation.strip():
                            # Clean up any contamination in translation
                            cleaned = translation.strip()
                            
                            # Check for contamination (paths that shouldn't be there)
                            original = texts[i]
                            if ('buyingPanel/' in cleaned or 'infoPanel/' in cleaned) and not ('buyingPanel/' in original or 'infoPanel/' in original):
                                print(f"⚠️  Detected contaminated translation, using original")
                                results[translatable[i]['idx']] = translatable[i]['content']
                            else:
                                # Escape the translated content for PO format
                                escaped_translation = escape_po(cleaned)
                                results[translatable[i]['idx']] = escaped_translation
                        else:
                            results[translatable[i]['idx']] = translatable[i]['content']
                    
                    break
                    
                except Exception as e:
                    if attempt < config['max_retries']:
                        wait_time = (attempt + 1) * 2
                        print(f"⚠️  Batch {batch_id} attempt {attempt + 1} failed: {e}")
                        print(f"   Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Batch {batch_id} failed after {config['max_retries']} retries: {e}")
                        for b in translatable:
                            results[b['idx']] = b['content']
    
    # Update progress for ALL items in batch, not just translated ones
    with progress_lock:
        progress.update(len(batch))
    
    return results

#checkpoint 5

def process_file(src, dst, target_language):
    print(f"Processing {src}...")
    lines, enc = read_po_file(src)
    blocks = find_msgstr_blocks(lines, target_language)
    total = len(blocks)
    
    if total == 0:
        print(f"No msgstr blocks found in {src}")
        return 0
    
    # Create batches
    batches = [blocks[i:i + config['batch_size']] for i in range(0, total, config['batch_size'])]
    progress = tqdm(total=total, desc='Translating', unit='str')
    
    all_results = {}
    
    # Process batches with threading
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = {executor.submit(translate_batch, batch, idx, progress, target_language): idx
                   for idx, batch in enumerate(batches)}
        
        for fut in futures:
            try:
                batch_results = fut.result()
                all_results.update(batch_results)
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
    
    progress.close()
    
    # Reconstruct the PO file
    out = []
    cur = 0
    
    for b in blocks:
        # Copy lines before this msgstr block
        out.extend(lines[cur:b['start']])
        
        # Get translation (already properly escaped if it was translated)
        trans = all_results.get(b['idx'], b['content'])
        indent = lines[b['start']].split('msgstr')[0]
        
        if b['is_plural']:
            # Handle plural forms
            for form_data in b['plural_forms']:
                form_num = form_data['form']
                # Use the same translation for all plural forms for now
                # In a more sophisticated version, you might want to handle different plural forms
                form_content = trans if form_num == 0 else trans
                
                # Check if this form was multiline originally
                original_line = None
                for line_idx in range(b['start'], b['end'] + 1):
                    if f'msgstr[{form_num}]' in lines[line_idx]:
                        original_line = lines[line_idx]
                        break
                
                is_multiline = original_line and original_line.strip().endswith('""')
                
                if is_multiline or '\n' in form_content.replace('\\n', ''):
                    # Multiline format
                    out.append(f"{indent}msgstr[{form_num}] \"\"")
                    
                    # Split by literal \n in the string
                    if '\\n' in form_content:
                        parts = form_content.split('\\n')
                        for j, part in enumerate(parts):
                            if j == len(parts) - 1 and not part:
                                # Last empty part from trailing \n
                                continue
                            if j == len(parts) - 1:
                                # Last part without \n - content is already escaped
                                out.append(f"{indent}\"{part}\"")
                            else:
                                # Part with \n - content is already escaped
                                out.append(f"{indent}\"{part}\\n\"")
                    else:
                        # No \n sequences, treat as single line in multiline format
                        out.append(f"{indent}\"{form_content}\"")
                else:
                    # Single line format - content is already escaped
                    out.append(f"{indent}msgstr[{form_num}] \"{form_content}\"")
        else:
            # Handle regular singular msgstr (existing logic)
            if b['is_multiline'] or '\n' in trans.replace('\\n', ''):
                # Multiline format
                out.append(f"{indent}msgstr \"\"")
                
                # Split by literal \n in the string
                if '\\n' in trans:
                    parts = trans.split('\\n')
                    for j, part in enumerate(parts):
                        if j == len(parts) - 1 and not part:
                            # Last empty part from trailing \n
                            continue
                        if j == len(parts) - 1:
                            # Last part without \n - content is already escaped
                            out.append(f"{indent}\"{part}\"")
                        else:
                            # Part with \n - content is already escaped
                            out.append(f"{indent}\"{part}\\n\"")
                else:
                    # No \n sequences, treat as single line in multiline format
                    out.append(f"{indent}\"{trans}\"")
            else:
                # Single line format - content is already escaped
                out.append(f"{indent}msgstr \"{trans}\"")
        
        cur = b['end'] + 1
    
    # Copy remaining lines
    out.extend(lines[cur:])
    
    # Write output
    try:
        with open(dst, 'w', encoding=enc) as f:
            f.write("\n".join(out))
    except Exception as e:
        print(f"❌ Error writing {dst}: {e}")
        return 0
    
    return len([b for b in blocks if b['should_translate']])

def main():
    global config, client, api_semaphore
    
    # Create directories
    for d in ('input', 'output'):
        os.makedirs(d, exist_ok=True)
    
    # Load or create config
    config = load_config() or interactive_setup()
    
    # Language selection
    print(f"\nAvailable target languages:")
    for i, (code, name) in enumerate(LANGUAGE_CODES.items(), 1):
        print(f"  {i:2d}. {code} — {name}")
    
    while True:
        sel = input("\nEnter language code or number: ").strip().upper()
        if sel.isdigit() and 1 <= int(sel) <= len(LANGUAGE_CODES):
            target_language = list(LANGUAGE_CODES.keys())[int(sel) - 1]
            break
        if sel in LANGUAGE_CODES:
            target_language = sel
            break
        print("❌ Invalid choice. Try again.")
    
    print(f"✅ Target language: {LANGUAGE_CODES[target_language]}")
    print(f"🧠 Smart translation: Only translates text that's NOT already in {LANGUAGE_CODES[target_language]}")
    
    # Get API key
    key = os.getenv('DEEPSEEK_API_KEY')
    if not key:
        key = ensure_and_load_deepseek_key()
    
    if not key and os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, 'r', encoding='utf-8') as f:
                key = f.read().strip()
            print(f"✅ API key loaded from {KEY_FILE}")
        except Exception as e:
            print(f"❌ Error reading {KEY_FILE}: {e}")
    elif key:
        print("✅ API key loaded from environment variable")
    
    if not key:
        print(f"❌ Missing Deepseek API key.")
        print(f"   Option 1: Set environment variable: export DEEPSEEK_API_KEY='your-key'")
        print(f"   Option 2: Create file '{KEY_FILE}' with your API key")
        sys.exit(1)
    
    # Initialize client
    client = DeepseekClient(api_key=key)
    api_semaphore = threading.Semaphore(config['max_workers'])
    
    # Find PO files
    po_files = [f for f in os.listdir('input') if f.endswith('.po')]
    if not po_files:
        print("❌ No .po files found in 'input' directory.")
        return
    
    print(f"\n📁 Found {len(po_files)} .po file(s) to process")
    print(f"⚡ Performance config: {config['batch_size']} batch size, {config['max_workers']} workers, {config['api_delay']}s delay")
    
    # Process files
    total_translated = 0
    start_time = time.time()
    
    for fname in po_files:
        in_path = os.path.join('input', fname)
        out_path = os.path.join('output', fname)
        
        try:
            count = process_file(in_path, out_path, target_language)
            print(f"✅ {fname}: {count} strings translated\n")
            total_translated += count
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}\n")
    
    elapsed = time.time() - start_time
    rate = total_translated / elapsed if elapsed > 0 else 0
    print(f"🎉 Done! Translated {total_translated} strings in {elapsed:.1f}s ({rate:.1f} str/s)")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
