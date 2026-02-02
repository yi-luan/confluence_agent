import spacy
import warnings
import re
from typing import List, Dict
from services.llm_preprocessor import LLMPreprocessor

warnings.filterwarnings("ignore", category=FutureWarning)

class EntityExtractor:
    def __init__(self):
        print("📥 正在初始化 EntityExtractor...")
        model_name = "zh_core_web_trf"
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ 模型載入成功！ (語言: {self.nlp.meta.get('lang')})")
        except OSError:
            print(f"❌ 錯誤: 找不到模型 '{model_name}'")
            raise

        # --- 定義黑名單 (常見的誤判詞彙) ---
        self.blocklist = {
            "Hello World", "Lorem Ipsum", "Example User", 
            "Test Case", "None", "Null", "Unknown"
        }

        # LLM preprocessor (可選) - 若未設定 GROK_API_URL/GROK_API_KEY 則為 no-op
        try:
            self.llm_preprocessor = LLMPreprocessor()
        except Exception:
            self.llm_preprocessor = None

    def extract_entities(self, text: str) -> List[Dict]:
        # optional LLM pre-processing
        if getattr(self, 'llm_preprocessor', None):
            try:
                lp = self.llm_preprocessor.format_text(text)
                clean_text = lp.get('clean_text') or text
                # handle both string and list format
                if isinstance(clean_text, list):
                    clean_text = ' '.join(clean_text)
                candidates = lp.get('candidates', [])
            except Exception:
                clean_text = text
                candidates = []
        else:
            clean_text = text
            candidates = []

        doc = self.nlp(clean_text)
        entities = []
        occupied_ranges = []

        # --- 第一階段：AI 模型提取 ---
        for ent in doc.ents:
            text_val = ent.text.strip()
            label = ent.label_

            # 1. 基礎雜訊過濾
            # 過濾空字串、單個字符(除非是GPE如'US')、以及黑名單
            if (not text_val) or \
               (len(text_val) < 2 and label != "GPE") or \
               (text_val in self.blocklist):
                continue
            
            # 2. 過濾純數字 (CARDINAL/QUANTITY) - 視需求而定，通常圖譜不需要存 "一" 或 "100"
            if label in ["CARDINAL", "QUANTITY", "PERCENT", "MONEY"]:
                continue

            # 3. 規則修正 (John Doe 被誤判為 ORG)
            if label == "ORG" and self._is_english_name_pattern(text_val):
                label = "PERSON"

            entities.append({
                "text": text_val,
                "label": label
            })
            occupied_ranges.append(range(ent.start_char, ent.end_char))

        # --- 第二階段：遺漏補捕 (Regex Fallback) ---
        name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b')
        
        for match in name_pattern.finditer(text):
            start, end = match.span()
            matched_text = match.group()

            # 檢查黑名單
            if matched_text in self.blocklist:
                continue

            # 檢查重疊
            is_overlap = any((start in r or end - 1 in r) for r in occupied_ranges)
            
            if not is_overlap:
                entities.append({
                    "text": matched_text,
                    "label": "PERSON" # 補捕的一律視為人名
                })
                occupied_ranges.append(range(start, end))

        # include LLM candidates (if any) as initial entities (prefer LLM types)
        for c in candidates:
            t = c.get('text')
            typ = c.get('type') or c.get('label') or 'PERSON'
            if t and isinstance(t, str):
                entities.append({'text': t.strip(), 'label': typ})

        # --- 第三階段：實體正規化與去重 ---
        normalized = []
        seen = set()
        for ent in entities:
            norm = self._normalize_entity(ent, text)
            if not norm:
                continue
            key = (norm['text'], norm['label'])
            if key in seen:
                continue
            seen.add(key)
            normalized.append(norm)

        return normalized

    def _is_english_name_pattern(self, text: str) -> bool:
        pattern = r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$'
        return bool(re.match(pattern, text))

    def _normalize_entity(self, entity: Dict, full_text: str) -> Dict | None:
        """Normalize / trim entity text to avoid full-sentence captures.

        - Trim surrounding punctuation
        - If entity is overly long, try to extract a shorter name inside
        - Prefer English name regex for PERSON, or Chinese 2-4 char name
        - Return None to drop the entity
        """
        text = entity.get('text', '').strip()
        label = entity.get('label')

        # trim surrounding punctuation and trailing clauses
        text = re.sub(r'^[\W_]+|[\W_]+$', '', text)
        # split on major punctuation and take the first segment
        text = re.split(r'[，,。；;:\-\n]', text)[0].strip()

        # if too long, attempt to extract an English name inside
        if len(text) > 30:
            # try English name
            m = re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
            if m:
                text = m.group()
                label = 'PERSON'
            else:
                # try Chinese short name
                m2 = re.search(r'[\u4e00-\u9fff]{2,4}', text)
                if m2:
                    text = m2.group()
                    label = 'PERSON'
                else:
                    return None

        # If label is PERSON but contains long descriptive sentence, extract english name if present
        if label == 'PERSON' and (len(text) > 12 or ' ' in text and len(text.split()) > 4):
            m = re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
            if m:
                text = m.group()

        # drop blacklisted or too-short results
        if not text or text in self.blocklist or len(text) < 2:
            return None

        return {'text': text, 'label': label}