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
from datetime import datetime
import subprocess

CONFIG_FILE = "config.json"
KEY_FILE = "deepseek_key.txt"

DEFAULT_CONFIG = {
    "batch_size": 60,
    "max_workers": 12,
    "api_delay": 0.05,
    "max_retries": 3,
    "min_text_length": 2,
    "max_tokens": 6000,
    "temperature": 0.05
}

LANGUAGE_CODES = {
    "EN": "English",
    "ES": "Spanish", 
    "FR": "French",
    "DE": "German",
    "PL": "Polish",
    "RU": "Russian",
    "UK": "Ukrainian"
}

LANGDETECT_TO_CODE = {
    'en': 'EN', 'es': 'ES', 'fr': 'FR', 'de': 'DE',
    'pl': 'PL', 'ru': 'RU', 'uk': 'UK'
}

COORD_MAP = {
    'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ж':'Zh','З':'Z','И':'I',
    'К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P','Р':'R','С':'S','Т':'T',
    'У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'Ch','Ш':'Sh','Щ':'Shch','Ы':'Y','Э':'E','Ю':'Yu','Я':'Ya'
}

progress_lock = threading.Lock()
api_semaphore = None

class DeepseekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def translate(self, texts, source_language="auto", target_language="EN"):
        if not texts:
            return []
        
        target_lang_name = LANGUAGE_CODES.get(target_language, target_language)
        
        # Create batch format
        batch_items = [f"[{i+1}] {text}" for i, text in enumerate(texts)]
        batch_text = "\n\n".join(batch_items)
        
        system_prompt = (
            f"Translate to {target_lang_name}. CRITICAL RULES:\n"
            "1. PRESERVE ALL variables: %(var)s, %d, {0}, {{key}}, etc.\n"
            "2. PRESERVE ALL formatting: \\n, \\t, spacing\n"
            "3. PRESERVE ALL special chars and symbols\n"
            f"4. Output natural {target_lang_name} for gaming/software\n"
            "5. Keep [N] numbering format\n"
            "6. NO extra text, explanations, or random characters\n"
            "7. Do NOT add backslashes randomly\n"
            "8. Military/gaming terms: танк=tank, броня=armor, урон=damage\n\n"
            "Output format: [N] translated_text"
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_text}
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
        
        return self._parse_translations(content, texts)
    
    def _parse_translations(self, content, original_texts):
        translations = []
        lines = content.split('\n')
        parsed = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(r'\[(\d+)\]\s*(.*)', line)
            if match:
                idx = int(match.group(1)) - 1
                translation = match.group(2).strip()
                if 0 <= idx < len(original_texts):
                    parsed[idx] = translation
        
        # Fallback parsing
        if len(parsed) < len(original_texts):
            sections = re.split(r'\n\s*\n', content)
            for i, section in enumerate(sections):
                if i < len(original_texts) and i not in parsed:
                    cleaned = re.sub(r'^\[\d+\]\s*', '', section.strip())
                    if cleaned:
                        parsed[i] = cleaned
        
        # Build final list with contamination check
        for i in range(len(original_texts)):
            if i in parsed and parsed[i].strip():
                translation = parsed[i].strip()
                # Clean up unwanted backslashes more carefully
                translation = self._clean_unwanted_backslashes(translation, original_texts[i])
                translations.append(translation)
            else:
                translations.append(original_texts[i])
        
        return translations
    
    def _clean_unwanted_backslashes(self, translation, original):
        """Remove backslashes that weren't in the original text"""
        if not translation or not original:
            return translation
        
        # Check if original had backslashes at all
        original_has_backslashes = '\\' in original
        
        if not original_has_backslashes and '\\' in translation:
            # Original had no backslashes, so remove any added ones
            # But be careful with escape sequences that might be legitimate
            original_unescaped = self._unescape_po_content(original)
            translation_cleaned = translation
            
            # Remove random backslash additions
            if '\\"' in translation and '"' not in original_unescaped:
                translation_cleaned = translation_cleaned.replace('\\"', '"')
            if '\\\\' in translation and '\\' not in original_unescaped:
                translation_cleaned = translation_cleaned.replace('\\\\', '')
            
            return translation_cleaned
        
        return translation
    
    def _unescape_po_content(self, content):
        """Properly unescape PO file content for analysis"""
        if not content:
            return content
        return content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')

def ensure_and_load_deepseek_key():
    keyfile = "deepseek_key.txt"
    while True:
        if not os.path.exists(keyfile) or not open(keyfile, encoding="utf-8").read().strip():
            if not os.path.exists(keyfile):
                with open(keyfile, "w", encoding="utf-8") as f:
                    f.write("")
            print(f"\n🔑 Please paste your Deepseek API key into: {os.path.abspath(keyfile)}")
            print("Opening it in your default text editor...")
            try:
                if sys.platform.startswith('win'):
                    os.startfile(keyfile)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', keyfile])
                else:
                    subprocess.Popen(['xdg-open', keyfile])
            except Exception as e:
                print(f"⚠️  Could not open editor: {e}")
            input("After pasting & saving your key, press Enter...")
        
        try:
            key = open(keyfile, encoding="utf-8").read().strip()
            if key:
                return key
            else:
                print("⚠️  No key detected. Please paste and save your key.")
        except Exception as e:
            print(f"❌ Error reading {keyfile}: {e}")
            sys.exit(1)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                config = DEFAULT_CONFIG.copy()
                config.update(loaded)
                return config
        except Exception:
            print(f"❌ Invalid JSON in '{CONFIG_FILE}'. Using defaults.")
    return None

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Could not save settings: {e}")

def interactive_setup():
    print("🚀 PO Translator - Setup")
    config = DEFAULT_CONFIG.copy()
    
    print(f"\nOptimized defaults:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if input("\nUse defaults? (Y/n): ").lower().startswith('n'):
        try:
            for key in ['batch_size', 'max_workers', 'max_retries']:
                config[key] = int(input(f"{key} [{config[key]}]: ") or config[key])
            config['api_delay'] = float(input(f"api_delay [{config['api_delay']}]: ") or config['api_delay'])
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
    if not text or len(text.strip()) < 3:
        return 'unknown'
    
    try:
        # More aggressive cleaning for better language detection
        clean_text = re.sub(r'%\([^)]+\)s|%\w+|{\w+}|\\[ntr]', ' ', text)  # Remove variables
        clean_text = re.sub(r'[^\w\s\u0080-\uFFFF]', ' ', clean_text)  # Keep unicode chars but remove punctuation
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 3:
            return 'unknown'
        
        # Check for obvious English patterns first
        english_indicators = [
            r'\b(the|and|or|to|of|a|an|is|are|was|were|have|has|had|will|would|could|should|can|may|might|this|that|these|those|with|for|from|by|in|on|at)\b',
            r'\b(button|click|menu|file|edit|view|help|save|open|close|cancel|ok|yes|no|options|settings)\b',
            r'\b(error|warning|info|message|alert|confirm|dialog|window|panel|tab|page)\b'
        ]
        
        english_matches = sum(1 for pattern in english_indicators if re.search(pattern, clean_text.lower()))
        
        # If we have multiple English indicators, it's likely English
        if english_matches >= 2:
            return 'EN'
        
        # For very short strings, be more conservative
        if len(clean_text) < 15:
            # Check for common English words
            common_english = ['ok', 'yes', 'no', 'save', 'open', 'close', 'cancel', 'help', 'file', 'edit', 'view', 'menu', 'button', 'click']
            if any(word in clean_text.lower() for word in common_english):
                return 'EN'
            
            # Check for obvious non-English characters
            if re.search(r'[\u0400-\u04FF]', text):  # Cyrillic
                return 'UK' if re.search(r'[іїєґ]', text) else 'RU'
            elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):  # Polish
                return 'PL'
            elif re.search(r'[àáâãäåçèéêëìíîïñòóôõöùúûüýÿ]', text, re.IGNORECASE):  # Romance languages
                # Try langdetect for these
                pass
            else:
                # Mostly ASCII, assume English for short strings
                if re.match(r'^[a-zA-Z0-9\s.,!?-]+$', clean_text):
                    return 'EN'
        
        # Use langdetect for longer strings or when unsure
        detected = detect(clean_text)
        our_code = LANGDETECT_TO_CODE.get(detected, 'unknown')
        
        # Verify langdetect results for common false positives
        if our_code == 'RU' or our_code == 'PL':
            # Double-check if this might actually be English
            if english_matches >= 1 and not re.search(r'[\u0400-\u04FF]', text) and not re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                return 'EN'
        
        if our_code == 'unknown':
            if re.search(r'[\u0400-\u04FF]', text):
                return 'UK' if re.search(r'[іїєґ]', text) else 'RU'
            elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                return 'PL'
            elif re.search(r'[a-zA-Z]', text):
                return 'EN'
        
        return our_code
        
    except Exception:
        # Fallback detection
        if re.search(r'[\u0400-\u04FF]', text):
            return 'UK' if re.search(r'[іїєґ]', text) else 'RU'
        elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
            return 'PL'
        elif re.search(r'[a-zA-Z]', text):
            return 'EN'
        return 'unknown'

def should_translate_content(content, target_language):
    if not content or not content.strip():
        return False, 'empty', 'unknown'
    
    if len(content.strip()) < config.get('min_text_length', 2):
        return False, 'too_short', 'unknown'
    
    detected_lang = detect_language_enhanced(content)
    
    if is_coordinate_format(content):
        return False, 'coordinate', detected_lang
    
    skip_patterns = [
        r'^[A-Z]\d+$', r'^\d+$', r'^[.,:;!?]+$',
        r'^[A-Za-z0-9_]+\.(png|jpg|jpeg|gif|svg|wav|mp3|ogg)$',
        r'^#[0-9A-Fa-f]{6}$', r'^\w+://', r'^[A-Za-z0-9_]+$'
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, content.strip()):
            return False, 'skip_pattern', detected_lang
    
    if detected_lang == 'unknown':
        return True, 'unknown_assume_translate', detected_lang
    elif detected_lang == target_language:
        return False, 'same_as_target', detected_lang
    else:
        return True, 'different_language', detected_lang

def is_coordinate_format(text):
    if not text or len(text.strip()) > 10:
        return False
    
    text = text.strip()
    return bool(re.match(r'^[A-ZА-Я]\d+$', text))

def translate_coordinate(coord_text):
    if not coord_text:
        return coord_text
    
    result = ""
    for char in coord_text:
        result += COORD_MAP.get(char, char)
    return result

def escape_po(text):
    if not text:
        return ""
    # Properly escape for PO format - only escape what needs escaping
    return text.replace('\\', '\\\\').replace('"', '\\"')

def unescape_po(text):
    """Properly unescape PO content"""
    if not text:
        return ""
    return text.replace('\\\\', '\\').replace('\\"', '"')

def parse_po_string_content(line):
    """Extract content from a PO string line, handling escaping properly"""
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        content = line[1:-1]  # Remove outer quotes
        return unescape_po(content)
    return line

def find_msgstr_blocks(lines, target_language):
    """Enhanced to handle both regular msgstr and plural forms msgstr[n]"""
    blocks = []
    i = 0
    idx = 0
    stats = {'empty': 0, 'too_short': 0, 'same_as_target': 0, 
             'coordinate': 0, 'skip_pattern': 0, 'different_language': 0,
             'unknown_assume_translate': 0}
    lang_stats = {}
    
    while i < len(lines):
        # Handle regular msgstr
        if lines[i].lstrip().startswith('msgstr '):
            block = parse_regular_msgstr(lines, i, idx, target_language, stats, lang_stats)
            if block:
                blocks.append(block)
                i = block['end'] + 1
                idx += 1
            else:
                i += 1
        
        # Handle plural forms msgstr[n]
        elif lines[i].lstrip().startswith('msgstr['):
            block = parse_plural_msgstr(lines, i, idx, target_language, stats, lang_stats)
            if block:
                blocks.append(block)
                i = block['end'] + 1
                idx += 1
            else:
                i += 1
        else:
            i += 1
    
    print_stats(blocks, target_language, stats, lang_stats)
    return blocks

def parse_regular_msgstr(lines, start_i, idx, target_language, stats, lang_stats):
    """Parse regular msgstr block"""
    i = start_i
    original_parts = []
    line = lines[i].lstrip()
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    
    if line.strip() == 'msgstr ""':
        # Multiline msgstr
        i += 1
        while i < len(lines) and lines[i].strip().startswith('"') and lines[i].strip().endswith('"'):
            content = parse_po_string_content(lines[i].strip())
            original_parts.append(content)
            i += 1
    else:
        # Single line msgstr
        content = line.partition(' ')[2].strip()
        if content.startswith('"') and content.endswith('"'):
            content = parse_po_string_content(content)
        original_parts.append(content)
        i += 1
    
    # Join parts but preserve original structure for analysis
    content = ''.join(original_parts)
    
    should_translate, reason, detected_lang = should_translate_content(content, target_language)
    
    stats[reason] = stats.get(reason, 0) + 1
    if detected_lang != 'unknown':
        lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
    
    return {
        'start': start_i, 'end': i-1, 'idx': idx, 'content': content,
        'original_parts': original_parts, 'reason': reason, 'should_translate': should_translate,
        'detected_lang': detected_lang, 'indent': indent,
        'is_multiline': line.strip() == 'msgstr ""',
        'type': 'regular'
    }

def parse_plural_msgstr(lines, start_i, idx, target_language, stats, lang_stats):
    """Parse plural msgstr[n] blocks"""
    i = start_i
    plural_forms = {}
    base_indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    
    # Collect all msgstr[n] forms
    while i < len(lines) and lines[i].lstrip().startswith('msgstr['):
        line = lines[i].lstrip()
        match = re.match(r'msgstr\[(\d+)\]\s*"(.*)"', line)
        if match:
            form_num = int(match.group(1))
            content = unescape_po(match.group(2))
            plural_forms[form_num] = content
        i += 1
    
    if not plural_forms:
        return None
    
    # Analyze first form for language detection
    first_content = list(plural_forms.values())[0]
    
    should_translate, reason, detected_lang = should_translate_content(first_content, target_language)
    
    stats[reason] = stats.get(reason, 0) + 1
    if detected_lang != 'unknown':
        lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
    
    return {
        'start': start_i, 'end': i-1, 'idx': idx, 'content': plural_forms,
        'reason': reason, 'should_translate': should_translate,
        'detected_lang': detected_lang, 'indent': base_indent,
        'is_multiline': False, 'type': 'plural'
    }

def print_stats(blocks, target_language, stats, lang_stats):
    total_blocks = len(blocks)
    translatable = sum(1 for b in blocks if b['should_translate'])
    
    print(f"📊 Analysis (Target: {LANGUAGE_CODES.get(target_language, target_language)}):")
    print(f"   Total: {total_blocks}, Translate: {translatable}, Skip: {total_blocks - translatable}")
    
    if lang_stats:
        print(f"\n📈 Languages:")
        for lang, count in sorted(lang_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {LANGUAGE_CODES.get(lang, lang)}: {count}")

def translate_batch(batch, batch_id, progress, target_language):
    if not batch:
        return {}
    
    coords = [b for b in batch if b['reason'] == 'coordinate']
    translatable = [b for b in batch if b['should_translate']]
    skippable = [b for b in batch if not b['should_translate']]
    
    results = {}
    
    # Handle coordinates and skippable content
    for b in coords:
        if b['type'] == 'regular':
            results[b['idx']] = translate_coordinate(b['content'])
        else:  # plural
            translated_forms = {}
            for form_num, content in b['content'].items():
                translated_forms[form_num] = translate_coordinate(content)
            results[b['idx']] = translated_forms
    
    for b in skippable:
        results[b['idx']] = b['content']
    
    # Translate content that needs translation
    if translatable:
        with api_semaphore:
            time.sleep(config['api_delay'])
            
            # Prepare texts for translation
            texts = []
            text_mapping = []  # Track which text belongs to which block
            
            for b in translatable:
                if b['type'] == 'regular':
                    texts.append(b['content'])
                    text_mapping.append((b['idx'], 'regular', None))
                else:  # plural
                    for form_num, content in b['content'].items():
                        texts.append(content)
                        text_mapping.append((b['idx'], 'plural', form_num))
            
            for attempt in range(config['max_retries'] + 1):
                try:
                    translations = client.translate(
                        texts=texts,
                        source_language='auto',
                        target_language=target_language
                    )
                    
                    if len(translations) != len(texts):
                        raise Exception(f"Translation count mismatch: got {len(translations)}, expected {len(texts)}")
                    
                    # Map translations back to blocks
                    for i, translation in enumerate(translations):
                        block_idx, block_type, form_num = text_mapping[i]
                        
                        if translation and translation.strip():
                            cleaned = translation.strip()
                            
                            # Check for contamination
                            original = texts[i]
                            if any(contamination in cleaned for contamination in ['buyingPanel/', 'infoPanel/']) and \
                               not any(contamination in original for contamination in ['buyingPanel/', 'infoPanel/']):
                                cleaned = original
                            
                            if block_type == 'regular':
                                results[block_idx] = cleaned
                            else:  # plural
                                if block_idx not in results:
                                    results[block_idx] = {}
                                results[block_idx][form_num] = cleaned
                        else:
                            if block_type == 'regular':
                                results[block_idx] = texts[i]
                            else:  # plural
                                if block_idx not in results:
                                    results[block_idx] = {}
                                results[block_idx][form_num] = texts[i]
                    
                    break
                    
                except Exception as e:
                    if attempt < config['max_retries']:
                        wait_time = (attempt + 1) * 2
                        print(f"⚠️  Batch {batch_id} retry {attempt + 1}: {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Batch {batch_id} failed: {e}")
                        for b in translatable:
                            results[b['idx']] = b['content']
    
    with progress_lock:
        progress.update(len(batch))
    
    return results

def write_po_string(content, indent, is_multiline=False):
    """Properly format a string for PO file output"""
    if not content:
        return f'{indent}msgstr ""'
    
    # Check if content contains newlines that should be preserved
    has_newlines = '\n' in content
    
    if is_multiline or has_newlines:
        lines = [f'{indent}msgstr ""']
        if has_newlines:
            parts = content.split('\n')
            for i, part in enumerate(parts):
                if i == len(parts) - 1 and not part:
                    continue
                if i == len(parts) - 1:
                    lines.append(f'{indent}"{escape_po(part)}"')
                else:
                    lines.append(f'{indent}"{escape_po(part)}\\n"')
        else:
            lines.append(f'{indent}"{escape_po(content)}"')
        return '\n'.join(lines)
    else:
        return f'{indent}msgstr "{escape_po(content)}"'

def process_file(src, dst, target_language):
    print(f"Processing {src}...")
    lines, enc = read_po_file(src)
    blocks = find_msgstr_blocks(lines, target_language)
    total = len(blocks)
    
    if total == 0:
        print(f"No msgstr blocks found in {src}")
        return 0
    
    batches = [blocks[i:i + config['batch_size']] for i in range(0, total, config['batch_size'])]
    progress = tqdm(total=total, desc='Translating', unit='str')
    
    all_results = {}
    
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = {executor.submit(translate_batch, batch, idx, progress, target_language): idx
                   for idx, batch in enumerate(batches)}
        
        for fut in futures:
            try:
                batch_results = fut.result()
                all_results.update(batch_results)
            except Exception as e:
                print(f"❌ Batch error: {e}")
    
    progress.close()
    
    # Reconstruct the PO file
    out = []
    cur = 0
    
    for b in blocks:
        out.extend(lines[cur:b['start']])
        
        trans = all_results.get(b['idx'], b['content'])
        indent = b['indent']
        
        if b['type'] == 'regular':
            # Regular msgstr
            if isinstance(trans, str):
                po_output = write_po_string(trans, indent, b['is_multiline'])
                out.extend(po_output.split('\n'))
        else:
            # Plural msgstr[n]
            if isinstance(trans, dict):
                for form_num in sorted(trans.keys()):
                    content = trans[form_num]
                    out.append(f"{indent}msgstr[{form_num}] \"{escape_po(content)}\"")
            else:
                # Fallback - reconstruct original
                for form_num in sorted(b['content'].keys()):
                    content = b['content'][form_num]
                    out.append(f"{indent}msgstr[{form_num}] \"{escape_po(content)}\"")
        
        cur = b['end'] + 1
    
    out.extend(lines[cur:])
    
    try:
        with open(dst, 'w', encoding=enc) as f:
            f.write("\n".join(out))
    except Exception as e:
        print(f"❌ Error writing {dst}: {e}")
        return 0
    
    return len([b for b in blocks if b['should_translate']])

def main():
    global config, client, api_semaphore
    
    for d in ('input', 'output'):
        os.makedirs(d, exist_ok=True)
    
    config = load_config() or interactive_setup()
    
    print(f"\nAvailable languages:")
    for i, (code, name) in enumerate(LANGUAGE_CODES.items()):
        print(f"  {i+1}. {code} - {name}")
    
    while True:
        try:
            choice = input(f"\nSelect target language (1-{len(LANGUAGE_CODES)}): ")
            target_code = list(LANGUAGE_CODES.keys())[int(choice) - 1]
            break
        except (ValueError, IndexError):
            print("❌ Invalid choice")
    
    api_key = ensure_and_load_deepseek_key()
    client = DeepseekClient(api_key)
    api_semaphore = threading.Semaphore(config['max_workers'])
    
    po_files = [f for f in os.listdir('input') if f.endswith('.po')]
    if not po_files:
        print("❌ No .po files found in 'input' folder")
        return
    
    print(f"\n🌍 Translating to {LANGUAGE_CODES[target_code]}...")
    start = time.time() 
    total_translated = 0
    
    for po_file in po_files:
        src = os.path.join('input', po_file)
        dst = os.path.join('output', po_file)
        translated = process_file(src, dst, target_code)
        total_translated += translated
        print(f"✅ {po_file}: {translated} strings translated")
    
    elapsed = time.time() - start
    print(f"\n🎉 Done! {total_translated} strings in {elapsed:.1f}s")
    
    if total_translated > 0:
        rate = total_translated / elapsed
        print(f"📈 Rate: {rate:.1f} strings/second")

if __name__ == "__main__":
    main()