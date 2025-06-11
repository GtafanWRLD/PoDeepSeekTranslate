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
                # Remove random backslashes that shouldn't be there
                if '\\' in translation and '\\' not in original_texts[i]:
                    # Only keep backslashes that were in original
                    translation = translation.replace('\\', '')
                translations.append(translation)
            else:
                translations.append(original_texts[i])
        
        return translations

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
        clean_text = re.sub(r'%\([^)]+\)s|%\w+|{\w+}|\\[ntr]', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 3:
            return 'unknown'
        
        detected = detect(clean_text)
        our_code = LANGDETECT_TO_CODE.get(detected, 'unknown')
        
        if our_code == 'unknown':
            if re.search(r'[\u0400-\u04FF]', text):
                return 'UK' if re.search(r'[іїєґ]', text) else 'RU'
            elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                return 'PL'
            elif re.search(r'[a-zA-Z]', text):
                return 'EN'
        
        return our_code
        
    except Exception:
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
    return text.replace('\\', '\\\\').replace('"', '\\"')

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
    original = []
    line = lines[i].lstrip()
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    
    if line.strip() == 'msgstr ""':
        # Multiline msgstr
        i += 1
        while i < len(lines) and lines[i].strip().startswith('"') and lines[i].strip().endswith('"'):
            content = lines[i].strip()[1:-1]
            original.append(content)
            i += 1
    else:
        # Single line msgstr
        content = line.partition(' ')[2].strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        original.append(content)
        i += 1
    
    content = ''.join(original)
    analysis_content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    
    should_translate, reason, detected_lang = should_translate_content(analysis_content, target_language)
    
    stats[reason] = stats.get(reason, 0) + 1
    if detected_lang != 'unknown':
        lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
    
    return {
        'start': start_i, 'end': i-1, 'idx': idx, 'content': content,
        'reason': reason, 'should_translate': should_translate,
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
            content = match.group(2)
            plural_forms[form_num] = content
        i += 1
    
    if not plural_forms:
        return None
    
    # Analyze first form for language detection
    first_content = list(plural_forms.values())[0]
    analysis_content = first_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    
    should_translate, reason, detected_lang = should_translate_content(analysis_content, target_language)
    
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
            if b['is_multiline'] or '\n' in str(trans).replace('\\n', ''):
                out.append(f"{indent}msgstr \"\"")
                if '\\n' in str(trans):
                    parts = str(trans).split('\\n')
                    for j, part in enumerate(parts):
                        if j == len(parts) - 1 and not part:
                            continue
                        if j == len(parts) - 1:
                            out.append(f"{indent}\"{escape_po(part)}\"")
                        else:
                            out.append(f"{indent}\"{escape_po(part)}\\n\"")
                else:
                    out.append(f"{indent}\"{escape_po(str(trans))}\"")
            else:
                out.append(f"{indent}msgstr \"{escape_po(str(trans))}\"")
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
        print("❌ Invalid choice.")
    
    print(f"✅ Target: {LANGUAGE_CODES[target_language]}")
    
    key = os.getenv('DEEPSEEK_API_KEY') or ensure_and_load_deepseek_key()
    client = DeepseekClient(api_key=key)
    api_semaphore = threading.Semaphore(config['max_workers'])
    
    po_files = [f for f in os.listdir('input') if f.endswith('.po')]
    if not po_files:
        print("❌ No .po files in 'input' directory.")
        return
    
    print(f"\n📁 Found {len(po_files)} .po file(s)")
    
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
    print(f"🎉 Done! {total_translated} strings in {elapsed:.1f}s ({rate:.1f} str/s)")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)