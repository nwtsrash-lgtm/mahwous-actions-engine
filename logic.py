"""
logic.py — Mahwous Hybrid Semantic Engine v8.0 (Golden Match Edition)
===================================================================
5-Layer Pipeline with Mathematical Rigor:
  L1  Deterministic Blocking & Feature Parsing
  L2  Semantic Vector Search (multilingual FAISS)
  L3  Weighted Fusion Match (Golden Equation: Brand 30% | Name 40% | Specs 20% | Visual 10%)
  L4  Triple Reverse Lookup (Safety Net)
  L5  LLM Oracle (Gemini 1.5 Flash) — final verification

Architecture: Zero-Error Tolerance for Mahwous Store.
"""

from __future__ import annotations

import gc
import os
import hashlib
import io
import json
import logging
import re
import threading
import time
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess

try:
    import google.generativeai as _google_genai
    _GENAI_OK = True
except ImportError:
    _google_genai = None
    _GENAI_OK = False

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

log = logging.getLogger("mahwous")
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

# ── Configuration ───────────────────────────────────────────────────────────
MIN_VOLUME_ML = 10.0  # استبعاد العينات أقل من 10 مل

# ═══════════════════════════════════════════════════════════════════════════
#  LLM-Powered Content Generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_product_description_with_llm(
    llm_client,
    product_name: str,
    brand: str,
    price: str,
    internal_links: dict = None,
    fragrantica_url: str = "",
) -> str:
    # Placeholder for actual LLM prompt based on the expert document
    # This prompt needs to be carefully crafted to meet all SEO and style requirements
    # For now, a simplified prompt to get started.
    prompt = f"""أنت خبير عالمي في كتابة أوصاف منتجات العطور محسّنة لمحركات البحث التقليدية (Google SEO) ومحركات بحث الذكاء الصناعي (GEO/AIO). تعمل حصرياً لمتجر "مهووس" (Mahwous) - الوجهة الأولى للعطور الفاخرة والنادرة. مهمتك هي كتابة وصف منتج احترافي وجذاب لمنتج جديد، مع الالتزام الصارم بالمعايير التالية:

**اسم المنتج:** {product_name}
**الماركة:** {brand}
**السعر:** {price} ريال سعودي

**التعليمات:**
1.  **الطول:** 1200-1500 كلمة.
2.  **البنية:** 9 أقسام رئيسية (مقدمة، نبذة عن العطر، مكونات العطر، قصة العطر، لمسة خبير من مهووس، متى وأين ترتدي، أسئلة متكررة (FAQ)، روابط داخلية وخارجية، خاتمة).
3.  **الكلمات المفتاحية (SEO):**
    *   الكلمة الرئيسية (اسم المنتج) في H1، وأول 50 كلمة، وآخر 100 كلمة، وتكرار 5-7 مرات.
    *   3 كلمات ثانوية (تكرار 3-5 مرات لكل منها).
    *   10-15 كلمة دلالية (تكرار 2-3 مرات لكل منها).
    *   5-8 عبارات حوارية في FAQ.
4.  **الروابط:** 3-5 روابط داخلية (إذا توفرت) ورابط خارجي واحد (Fragrantica).
5.  **الأسلوب:** مزيج من الراقي، الودود، العاطفي، والتسويقي. لا تستخدم الإيموجي. استخدم Bold للكلمات المهمة (بدون مبالغة). ضمن أرقام وإحصائيات إن أمكن.
6.  **المصادر:** استخدم Fragrantica Arabia و Google للبحث عن معلومات دقيقة حول العطر.
7.  **التنسيق النهائي:** يجب أن يكون جاهزاً للنسخ واللصق مباشرة بصيغة Markdown، ومنظماً بالترتيب المذكور أعلاه.

**مثال على الروابط الداخلية (إذا توفرت):**
{internal_links}

**رابط Fragrantica (إذا توفر):**
{fragrantica_url}

**الآن، اكتب الوصف الكامل لـ {product_name}:**
"""
    try:
        response = llm_client.chat.completions.create(
            model="gemini-1.5-flash", # أو gpt-4.1-mini حسب التوفر
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000, # لضمان طول الوصف المطلوب
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"فشل توليد وصف المنتج بواسطة LLM: {e}")
        return ""

def _generate_brand_description_with_llm(
    llm_client,
    brand_name_ar: str,
    brand_name_en: str,
) -> str:
    # بناء اسم الماركة الموحد مع مراعاة حد الـ 30 حرفاً
    combined_brand_name = f"{brand_name_ar} | {brand_name_en}"
    if len(combined_brand_name) > 30:
        # محاولة اختصار الاسم الإنجليزي إذا كان الاسم العربي طويلاً
        if len(brand_name_ar) < 25: # ترك مساحة لـ | و EN
            brand_name_en_short = brand_name_en[:(30 - len(brand_name_ar) - 3)] # -3 for ' | '
            combined_brand_name = f"{brand_name_ar} | {brand_name_en_short}"
        else:
            combined_brand_name = brand_name_ar[:27] + "..." # قص الاسم العربي إذا كان طويلاً جداً

    prompt = f"""أنت خبير في كتابة أوصاف الماركات لمتجر مهووس. مهمتك هي كتابة وصف موجز وجذاب للماركة التالية، مع الالتزام بالمعايير:

**اسم الماركة بالعربية:** {brand_name_ar}
**اسم الماركة بالإنجليزية:** {brand_name_en}
**اسم الماركة الموحد (عربي | إنجليزي، بحد أقصى 30 حرفاً):** {combined_brand_name}

**التعليمات:**
1.  **الطول:** 50-100 كلمة.
2.  **الأسلوب:** احترافي، موجز، يعكس هوية الماركة، ومناسب لمتجر مهووس.
3.  **التركيز:** أبرز تاريخ الماركة، فلسفتها، وأهم ما يميزها في عالم العطور. اذكر أي تفاصيل فريدة أو إنجازات.
4.  **المصادر:** استخدم معلومات عامة موثوقة عن الماركة.
5.  **التنسيق:** نص عادي، بدون Markdown أو عناوين. يجب أن يكون جاهزاً للنسخ واللصق.

**الآن، اكتب الوصف الكامل للماركة {brand_name_ar}:**
"""
    try:
        response = llm_client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"فشل توليد وصف الماركة بواسطة LLM: {e}")
        return ""

# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProductFeatures:
    """Parsed product attributes extracted in Layer 1."""
    volume_ml:     float = 0.0
    concentration: str = ""
    brand_ar:      str = ""
    brand_en:      str = ""
    category:      str = ""   # perfume | beauty | unknown
    gtin:          str = ""
    sku:           str = ""
    model_num:     str = ""   # Numbers extracted from name (excluding volume)


@dataclass
class MatchResult:
    """Full match record for one competitor product."""
    verdict:          str   = "review"   # new | duplicate | review
    confidence:       float = 0.0
    layer_used:       str   = ""         
    store_name:       str   = ""
    store_image:      str   = ""
    comp_name:        str   = ""
    comp_image:       str   = ""
    comp_price:       str   = ""
    comp_source:      str   = ""
    feature_details:  str   = ""
    faiss_score:      float = 0.0
    lex_score:        float = 0.0
    llm_reasoning:    str   = ""
    product_type:     str   = "perfume"
    brand:            str   = ""
    brand_ar:         str   = ""
    brand_en:         str   = ""
    comp_brand_raw:   str   = "" # الماركة المستخرجة من منتج المنافس
    salla_category:   str   = ""
    generated_product_description: str = ""
    generated_brand_description: str = ""
    gtin:             str   = ""


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 1 — Deterministic Blocking & Feature Parsing
# ═══════════════════════════════════════════════════════════════════════════

class FeatureParser:
    """Extracts structured features from raw product name strings."""

    _VOL = re.compile(
        r"(\d+\.?\d*)\s*(ml|مل|g|gr|غ|oz|fl\.?\s*oz|مل|cc)",
        re.IGNORECASE | re.UNICODE,
    )
    _CONC_MAP = {
        "EDP": ["او دو برفيوم","او دي بارفيوم","اودو بارفيوم","او دو بيرفيوم","او دو برفوم","اودي برفيوم","اود برفيوم","بارفيوم","برفيوم","بيرفيوم","بارفان","لو بارفان","لو دي بارفان", r"\bedp\b", r"eau\s+de\s+parfum", r"eau\s+du?\s+parfu"],
        "EDT": ["او دو تواليت","او دي تواليت","اودي تواليت","تواليت", r"\bedt\b", r"eau\s+de\s+toilette"],
        "EDC": ["او دو كولون","كولون","كولونيا", r"\bedc\b", r"eau\s+de\s+cologne"],
        "Extrait": ["اكستريت","إكستريت","اليكسير دي بارفيوم","اليكسير دو بارفيوم","اليكسير دي بارفان","انتنس اكستريت", r"\bextrait\b", r"elixir\s+de\s+parfum", r"\belixir\b", r"\bintense\b", r"\bintens\b"],
        "Parfum": ["بارفيوم ناتورال","ماء العطر", r"\bparfum\b", r"\bperfume\b"],
        "HairMist": ["رذاذ الشعر","بخاخ الشعر","معطر الشعر", r"hair\s+mist", r"hair\s+perfume"],
        "BodyMist": ["بخاخ الجسم","بخاخ للجسم","بخاخ معطر", r"body\s+mist", r"body\s+spray"],
    }

    @classmethod
    def parse(cls, name: str, sku: str = "", gtin: str = "", brands_list: list[str] = []) -> ProductFeatures:
        name_lower = name.lower().strip()
        
        # Volume extraction
        vol_val = 0.0
        m = cls._VOL.search(name)
        if m:
            try:
                vol_val = float(m.group(1))
                unit = m.group(2).lower()
                if "oz" in unit: vol_val *= 29.57
            except: pass

        # Concentration (هروب احتياطي إذا كان النمط نصاً حرفياً يحتوي أحرف regex)
        conc = ""
        for k, patterns in cls._CONC_MAP.items():
            for pat in patterns:
                try:
                    m = re.search(pat, name_lower, re.IGNORECASE)
                except re.error:
                    m = re.search(re.escape(pat), name_lower, re.IGNORECASE)
                if m:
                    conc = k
                    break
            if conc:
                break

        # Brand
        brand_ar, brand_en = cls._extract_brand(name_lower, brands_list)

        # Model numbers (excluding volume)
        all_nums = re.findall(r"\d+\.?\d*", name)
        vol_str = str(int(vol_val)) if vol_val > 0 else "____"
        model_nums = "-".join([n for n in all_nums if n != vol_str])

        return ProductFeatures(
            volume_ml=vol_val,
            concentration=conc,
            brand_ar=brand_ar,
            brand_en=brand_en,
            gtin=str(gtin).strip() if gtin else "",
            sku=str(sku).strip() if sku else "",
            model_num=model_nums
        )

    @staticmethod
    def _extract_brand(name_lower: str, brands: list[str]) -> tuple[str, str]:
        best_match_ar, best_match_en = "", ""
        best_score = 0

        for b_entry in brands:
            # b_entry can be 'العربية | English' or just 'العربية'
            parts = [p.strip() for p in b_entry.split("|")]
            brand_ar = parts[0] if len(parts) > 0 else ""
            brand_en = parts[1] if len(parts) > 1 else ""

            # Try matching Arabic brand name
            if brand_ar:
                score_ar = rfuzz.partial_ratio(brand_ar.lower(), name_lower)
                if score_ar > best_score and score_ar > 80: # Threshold for partial match
                    best_score = score_ar
                    best_match_ar = brand_ar
                    best_match_en = brand_en
            
            # Try matching English brand name
            if brand_en:
                score_en = rfuzz.partial_ratio(brand_en.lower(), name_lower)
                if score_en > best_score and score_en > 80: # Threshold for partial match
                    best_score = score_en
                    best_match_ar = brand_ar
                    best_match_en = brand_en

        return best_match_ar, best_match_en


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 3 — The Golden Match Equation
# ═══════════════════════════════════════════════════════════════════════════

class GoldenMatchEngine:
    """
    Mathematical Fusion of multiple signals to reach 100% precision.
    """
    WEIGHTS = {
        "brand": 0.30,  # تطابق الماركة
        "name":  0.40,  # تطابق الاسم (بدون ماركة وحجم)
        "specs": 0.20,  # تطابق الحجم والتركيز
        "visual": 0.10  # تطابق الصورة (Hash)
    }

    @classmethod
    def calculate_score(cls, comp_name: str, store_name: str, comp_feat: ProductFeatures, store_feat: ProductFeatures, comp_img: str, store_img: str) -> float:
        # 1. Brand Score (Binary-ish)
        brand_score = 1.0 if (comp_feat.brand_ar and comp_feat.brand_ar == store_feat.brand_ar) or \
                             (comp_feat.brand_en and comp_feat.brand_en == store_feat.brand_en) else 0.0
        if not comp_feat.brand_ar and not store_feat.brand_ar: brand_score = 0.5 # Neutral if unknown
        
        # 2. Name Score (Clean similarity)
        c_name_clean = cls._clean_name(comp_name, comp_feat)
        s_name_clean = cls._clean_name(store_name, store_feat)
        name_score = rfuzz.token_sort_ratio(c_name_clean, s_name_clean) / 100
        
        # 3. Specs Score (Volume + Conc + Model)
        vol_match = 1.0 if abs(comp_feat.volume_ml - store_feat.volume_ml) < 2.0 else 0.0
        conc_match = 1.0 if comp_feat.concentration == store_feat.concentration else 0.5
        model_match = 1.0 if comp_feat.model_num == store_feat.model_num else 0.0
        specs_score = (vol_match * 0.4) + (conc_match * 0.3) + (model_match * 0.3)
        
        # 4. Visual Score (Simplified Fingerprint via URL/Path)
        visual_score = 1.0 if comp_img and store_img and comp_img.split('/')[-1] == store_img.split('/')[-1] else 0.5

        # Final Fusion
        final = (brand_score * cls.WEIGHTS["brand"]) + \
                (name_score * cls.WEIGHTS["name"]) + \
                (specs_score * cls.WEIGHTS["specs"]) + \
                (visual_score * cls.WEIGHTS["visual"])
        
        # Absolute Penalty: If volume is significantly different, it's NOT a match
        if vol_match == 0 and comp_feat.volume_ml > 0 and store_feat.volume_ml > 0:
            final *= 0.5
            
        return final

    @staticmethod
    def _clean_name(name: str, feat: ProductFeatures) -> str:
        n = name.lower()
        for trash in [feat.brand_ar, feat.brand_en, feat.concentration, "عطر", "تستر", "مل", "ml"]:
            if trash: n = n.replace(trash.lower(), "")
        return re.sub(r"\s+", " ", n).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 4 — Triple Reverse Lookup (Safety Net)
# ═══════════════════════════════════════════════════════════════════════════

class ReverseLookup:
    """
    Search back into store to ensure 'New Opportunity' is truly missing.
    """
    @classmethod
    def verify(cls, comp_name: str, comp_feat: ProductFeatures, store_df: pd.DataFrame, idx: SemanticIndex) -> bool:
        # Method 1: Semantic Reverse
        hits = idx.search(comp_name, k=3)
        for sname, score in hits:
            if score > 0.92: return True # Found in store
            
        # Method 2: Brand + Volume + Model Filter (هروب regex — أسماء الماركات قد تحتوي على ) ( ? …)
        if comp_feat.brand_ar or comp_feat.brand_en:
            b = comp_feat.brand_ar or comp_feat.brand_en
            vol_token = str(int(comp_feat.volume_ml)) if comp_feat.volume_ml > 0 else "____"
            mask = (
                store_df["product_name"].str.contains(re.escape(b), case=False, na=False, regex=True)
                & store_df["product_name"].str.contains(vol_token, case=False, na=False, regex=False)
            )
            if mask.any():
                return True

        # Method 3: Keyword Match (هروب كل كلمة — تجنّب missing ), unterminated subpattern)
        keywords = [w for w in comp_name.split() if len(w) > 3][:3]
        if keywords:
            pattern = ".*".join(re.escape(w) for w in keywords)
            if store_df["product_name"].str.contains(pattern, case=False, na=False, regex=True).any():
                return True
                
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MahwousEngine:
    def __init__(
        self,
        semantic_index: SemanticIndex,
        brands_list: list[str] = [],
        gemini_oracle=None,
        openai_key: str = "",
        search_api_key: str = "",
        search_cx: str = "",
        fetch_images: bool = False,
    ):
        self.idx = semantic_index
        self.brands = brands_list
        self.oracle = gemini_oracle
        self.all_comp_brands = set() # لجمع كل الماركات المستخرجة من المنافسين
        self.search_api_key = search_api_key
        self.search_cx = search_cx
        self.fetch_images = fetch_images
        self.new_brands_to_add = set() # لجمع الماركات الجديدة التي لم يتم العثور عليها في ملفنا
        # أوصاف المنتج/الماركة عبر OpenAI فقط عند وجود OPENAI_API_KEY (لا نستدعي OpenAI() بدون مفتاح)
        self.llm_client = None
        if self.oracle:
            _okey = os.environ.get("OPENAI_API_KEY", "").strip()
            if _okey:
                try:
                    self.llm_client = OpenAI(api_key=_okey)
                except Exception:
                    self.llm_client = None

    def _llm_batch_verify(self, batch: list[MatchResult]) -> list[str]:
        """Verifies a batch of 20 products via LLM with Retry Logic and Error Handling."""
        if not self.llm_client: return ["review"] * len(batch)
        
        prompt = "Compare these COMPETITOR products with our STORE products. \n"
        prompt += "Rules: Reply 'duplicate' if identical (brand, model, volume), 'new' if different, 'review' if unsure.\n"
        for i, r in enumerate(batch):
            prompt += f"ID:{i} | Comp: {r.comp_name} | Store: {r.store_name}\n"
        prompt += "\nReturn JSON: {\"results\": [\"duplicate\", \"new\", ...]}"
        
        for attempt in range(3): # Auto-Resilience: 3 Retries
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" },
                    timeout=30
                )
                data = json.loads(response.choices[0].message.content)
                res = data.get("results", [])
                if len(res) == len(batch): return res
                log.warning(f"LLM Batch mismatch (attempt {attempt+1}): expected {len(batch)}, got {len(res)}")
            except Exception as e:
                log.error(f"LLM Attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1)) # Exponential backoff
        return ["review"] * len(batch)

    def run(
        self,
        store_df: pd.DataFrame,
        comp_df: pd.DataFrame,
        use_llm: bool = False,
        progress_cb: Optional[Callable] = None,
        log_cb: Optional[Callable] = None,
    ) -> tuple[list[MatchResult], list[MatchResult], list[MatchResult]]:
        def _log(msg):
            if log_cb: log_cb(msg)
            else: log.info(msg)
        store_names = store_df["product_name"].tolist()
        store_imgs  = store_df.get("image_url", pd.Series([""] * len(store_df))).tolist()
        store_feats = {name: FeatureParser.parse(name, brands_list=self.brands) for name in store_names}
        
        new_opps, duplicates, reviews = [], [], []
        
        total = len(comp_df)
        for i, (_, row) in enumerate(comp_df.iterrows()):
            if progress_cb: progress_cb(i, total, str(row.get("product_name", "")))
            comp_name = str(row.get("product_name","")).strip()
            comp_img  = str(row.get("image_url","")).strip()
            
            if not comp_name or len(comp_name) < 3: continue
            
            # 1. Parsing & Exclusion (Samples < 10ml)
            comp_feat = FeatureParser.parse(comp_name, brands_list=self.brands)
            
            # جمع الماركات المستخرجة من منتجات المنافسين
            if comp_feat.brand_ar: self.all_comp_brands.add(comp_feat.brand_ar)
            elif comp_feat.brand_en: self.all_comp_brands.add(comp_feat.brand_en)

            if comp_feat.volume_ml > 0 and comp_feat.volume_ml < MIN_VOLUME_ML:
                continue # Skip small samples as requested
                
            # 2. Golden Match Calculation
            faiss_hits = self.idx.search(comp_name, k=5)
            best_score, best_store, best_store_img = 0.0, "", ""
            
            for (sname, f_score) in faiss_hits:
                score = GoldenMatchEngine.calculate_score(comp_name, sname, comp_feat, store_feats[sname], comp_img, store_imgs[store_names.index(sname)])
                if score > best_score:
                    best_score, best_store = score, sname
                    best_store_img = store_imgs[store_names.index(sname)]
            
            result = MatchResult(
                comp_name=comp_name, comp_image=comp_img, 
                comp_price=str(row.get("price","")), comp_source=str(row.get("source_file","")),
                store_name=best_store, store_image=best_store_img,
                confidence=best_score, 
                brand=comp_feat.brand_ar or comp_feat.brand_en,
                brand_ar=comp_feat.brand_ar,
                brand_en=comp_feat.brand_en,
                comp_brand_raw=comp_feat.brand_ar or comp_feat.brand_en,
                gtin=str(row.get("barcode", "")) or str(row.get("gtin", ""))
            )

            # توليد وصف المنتج والماركة (يتطلب عميل OpenAI صالحاً — Gemini لا يُستخدم هنا حالياً)
            if result.verdict == "new" and self.oracle and self.llm_client:
                internal_links = {}
                fragrantica_url = ""
                result.generated_product_description = _generate_product_description_with_llm(
                    self.llm_client, result.comp_name, result.brand, result.comp_price,
                    internal_links, fragrantica_url
                )
                if result.comp_brand_raw and result.comp_brand_raw not in self.brands:
                    brand_name_en_for_llm = result.brand_en if result.brand_en else ""
                    result.generated_brand_description = _generate_brand_description_with_llm(
                        self.llm_client, result.comp_brand_raw, brand_name_en_for_llm
                    )

            # 3. Decision Logic based on Golden Score
            if best_score >= 0.88:
                result.verdict, result.layer_used = "duplicate", "GOLDEN-HIGH"
                duplicates.append(result)
            elif best_score < 0.55:
                # 4. Triple Reverse Lookup for safety
                is_actually_in_store = ReverseLookup.verify(comp_name, comp_feat, store_df, self.idx)
                if not is_actually_in_store:
                    result.verdict, result.layer_used = "new", "GOLDEN-LOW-VERIFIED"
                    new_opps.append(result)
                else:
                    result.verdict, result.layer_used = "duplicate", "REVERSE-LOOKUP-FOUND"
                    duplicates.append(result)
            else:
                result.verdict, result.layer_used = "review", "GOLDEN-GRAY"
                reviews.append(result)

        # --- L5: LLM Final Polish ---
        if use_llm and reviews:
            log.info(f"Refining {len(reviews)} gray-zone items via LLM...")
            final_new, final_dups, final_revs = [], [], []
            batch_size = 20
            for i in range(0, len(reviews), batch_size):
                batch = reviews[i:i+batch_size]
                verdicts = self._llm_batch_verify(batch)
                for res, v in zip(batch, verdicts):
                    if v == "duplicate":
                        res.verdict, res.layer_used = "duplicate", "LLM-VERIFIED"
                        duplicates.append(res)
                    elif v == "new":
                        res.verdict, res.layer_used = "new", "LLM-VERIFIED"
                        new_opps.append(res)
                    else:
                        final_revs.append(res)
            reviews = final_revs
                
        # استخراج الماركات الجديدة
        known_brands_lower = {b.lower() for b in self.brands}
        new_brands = [b for b in self.all_comp_brands if b.lower() not in known_brands_lower]

        return new_opps, duplicates, reviews, new_brands

# (Keep helper functions _read_csv, load_store_products, load_competitor_products, SemanticIndex as before)
# ... [rest of the file remains same as previous patched version]
# ... (I will omit them for brevity in this block but they are present in the final file)
# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS (Re-included for completeness)
# ═══════════════════════════════════════════════════════════════════════════

def _read_csv(file_obj, header: int | None = 0, **kwargs) -> pd.DataFrame:
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, str): raw = raw.encode("utf-8")
    else:
        with open(file_obj, "rb") as fh: raw = fh.read()
    # Try different encodings and delimiters
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, header=header, **kwargs)
            except Exception:
                continue
    raise ValueError("Cannot decode CSV with common encodings/delimiters")

def _read_excel(file_obj, header: int | None = 0) -> pd.DataFrame:
    return pd.read_excel(file_obj, header=header)

def _read_file(file_path, header: int | None = 0) -> pd.DataFrame:
    if str(file_path).lower().endswith(".csv"):
        return _read_csv(file_path, header=header)
    elif str(file_path).lower().endswith(".xlsx"):
        return _read_excel(file_path, header=header)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def load_store_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            frame = pd.DataFrame()
            # 1) محاولة القراءة الطبيعية مع Header والبحث بالاسم.
            df = _read_file(f)
            df.columns = [str(c).strip() for c in df.columns]

            name_keywords = ["name", "اسم", "product", "title", "عنوان", "منتج", "المنتج"]
            img_keywords = ["image", "img", "صورة", "photo", "picture", "url", "src"]

            def _find_col(keywords, fallback_idx=None):
                for c in df.columns:
                    low = str(c).lower()
                    if any(k in low for k in keywords):
                        return c
                if fallback_idx is not None and len(df.columns) > fallback_idx:
                    return df.columns[fallback_idx]
                return None

            nc = _find_col(name_keywords)
            ic = _find_col(img_keywords)

            if nc is not None:
                frame["product_name"] = df[nc].fillna("").astype(str)
                if ic is not None:
                    frame["image_url"] = df[ic].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "")
                else:
                    frame["image_url"] = ""
            else:
                # 2) احتياط: ملفات بلا Header (الترتيب الشائع: ID, name, image)
                raw_df = _read_file(f, header=None)
                if len(raw_df.columns) >= 3:
                    frame["product_name"] = raw_df.iloc[:, 1].fillna("").astype(str)
                    frame["image_url"] = raw_df.iloc[:, 2].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "")
                else:
                    log.warning(f"الملف {getattr(f, 'name', str(f))} لا يحتوي على أعمدة كافية لقراءة بيانات المتجر.")
                    continue

            # تصحيح ذكي: إذا كان عمود الاسم رقميّاً غالباً (ID/SKU)، اختر عموداً نصيّاً أنسب.
            def _text_score(s: pd.Series) -> float:
                vals = s.fillna("").astype(str).str.strip()
                vals = vals[vals != ""].head(200)
                if len(vals) == 0:
                    return 0.0
                alpha = vals.str.contains(r"[A-Za-z\u0600-\u06FF]", regex=True, na=False).mean()
                longish = (vals.str.len() >= 8).mean()
                pure_num = vals.str.fullmatch(r"[\d\-\._]+", na=False).mean()
                return (alpha * 0.7) + (longish * 0.3) - (pure_num * 0.9)

            name_is_mostly_numeric = frame["product_name"].fillna("").astype(str).str.strip().str.fullmatch(r"[\d\-\._]+", na=False).mean() > 0.7
            if name_is_mostly_numeric:
                candidate_cols = [c for c in df.columns if str(c) != str(ic)]
                if candidate_cols:
                    best_col = max(candidate_cols, key=lambda c: _text_score(df[c]))
                    best_series = df[best_col].fillna("").astype(str)
                    if _text_score(best_series) > 0.15:
                        frame["product_name"] = best_series

            frame = frame[frame["product_name"].str.strip() != ""]
            # تجاهل صفوف Header التي قد تُقرأ كنص داخل البيانات
            frame = frame[~frame["product_name"].str.lower().str.contains(r"^(name|اسم المنتج|product name)$", na=False)]
            frames.append(frame)
        except Exception as e: log.error(f"load_store: {e}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["product_name"]).reset_index(drop=True) if frames else pd.DataFrame()

def load_competitor_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = _read_file(f)
            df.columns = [str(c).strip() for c in df.columns]
            # البحث الشامل عن عمود الاسم في ملفات المنافسين بأي صيغة
            name_keywords = ["name", "اسم", "productcard", "product", "title", "عنوان", "منتج", "المنتج"]
            img_keywords  = ["src", "image", "img", "صورة", "photo", "picture", "url"]
            price_keywords = ["price", "سعر", "cost", "تكلفة", "ثمن"]
            
            def _find_col(keywords, fallback_idx=None):
                for c in df.columns:
                    cl = c.lower()
                    if any(h in cl for h in keywords):
                        return c
                if fallback_idx is not None and len(df.columns) > fallback_idx:
                    return df.columns[fallback_idx]
                return None
            
            nc = _find_col(name_keywords, 2)
            ic = _find_col(img_keywords, 1)
            pc = _find_col(price_keywords, 3)
            
            if nc is None:
                log.warning(f"load_competitor: لم يُعثر على عمود اسم في {getattr(f, 'name', str(f))} | الأعمدة: {list(df.columns)[:8]}")
                continue
            
            frame = pd.DataFrame()
            frame["product_name"] = df[nc].fillna("").astype(str)
            frame["image_url"]    = df[ic].fillna("").astype(str) if ic else ""
            frame["price"]        = df[pc].fillna("").astype(str) if pc else ""
            frame["source_file"]  = getattr(f, 'name', str(f))
            frame = frame[frame["product_name"].str.strip() != ""]
            if not frame.empty:
                frames.append(frame)
            else:
                log.warning(f"load_competitor: الملف فارغ بعد التصفية: {getattr(f, 'name', str(f))}")
        except Exception as e:
            log.error(f"load_competitor error: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_brands(file) -> list[str]:
    try:
        df = _read_file(file)
        col = next((c for c in df.columns if "اسم" in str(c)), df.columns[0])
        return df[col].dropna().astype(str).tolist()
    except: return []

class SemanticIndex:
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    def __init__(self, model):
        self._model, self._index, self._store_names = model, None, []
    def build(self, store_df: pd.DataFrame, progress_cb: Optional[Callable] = None):
        """بناء الفهرس على دفعات لتقليل ذروة الذاكرة (يخفّف خطأ Windows 1455)."""
        self._store_names = store_df["product_name"].tolist()
        n = len(self._store_names)
        row_batch = int(os.environ.get("MAHWOUS_ENCODE_BATCH", "64"))
        st_batch = int(os.environ.get("MAHWOUS_ST_BATCH", "8"))
        row_batch = max(8, row_batch)
        st_batch = max(1, st_batch)
        if progress_cb:
            progress_cb(f"⏳ ترميز {n:,} منتج على دفعات ({row_batch} سطر / دفعة) لتوفير RAM…")
        self._index = None
        for start in range(0, n, row_batch):
            chunk = self._store_names[start : start + row_batch]
            emb = self._model.encode(
                chunk,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=min(st_batch, len(chunk)),
                convert_to_numpy=True,
            )
            emb = np.asarray(emb, dtype=np.float32)
            if self._index is None:
                self._index = faiss.IndexFlatIP(emb.shape[1])
            self._index.add(emb)
            del emb
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if progress_cb:
            progress_cb(f"✅ FAISS بُني: {n:,} متجه")
    def search(self, query: str, k: int = 3):
        if self._index is None: return []
        qvec = self._model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self._index.search(qvec, k)
        return [(self._store_names[i], s) for s, i in zip(scores[0], idxs[0]) if i >= 0]

def export_salla_csv(results: list[MatchResult]) -> bytes:
    cols = ["النوع ","أسم المنتج","تصنيف المنتج","صورة المنتج","وصف صورة المنتج","نوع المنتج","سعر المنتج","الوصف","هل يتطلب شحن؟","رمز المنتج sku","سعر التكلفة","السعر المخفض","تاريخ بداية التخفيض","تاريخ نهاية التخفيض","اقصي كمية لكل عميل","إخفاء خيار تحديد الكمية","اضافة صورة عند الطلب","الوزن","وحدة الوزن","الماركة","العنوان الترويجي","تثبيت المنتج","الباركود","السعرات الحرارية","MPN","GTIN","خاضع للضريبة ؟","سبب عدم الخضوع للضريبة","[1] الاسم","[1] النوع","[1] القيمة","[1] الصورة / اللون","[2] الاسم","[2] النوع","[2] القيمة","[2] الصورة / اللون","[3] الاسم","[3] النوع","[3] القيمة","[3] الصورة / اللون"]
    rows = []
    for r in results:
        # بناء اسم الماركة الموحد مع مراعاة حد الـ 30 حرفاً
        brand_ar = r.brand_ar if r.brand_ar else r.comp_brand_raw
        brand_en = r.brand_en if r.brand_en else ""
        combined_brand_name = f"{brand_ar} | {brand_en}"
        if len(combined_brand_name) > 30:
            if len(brand_ar) < 25:
                brand_en_short = brand_en[:(30 - len(brand_ar) - 3)]
                combined_brand_name = f"{brand_ar} | {brand_en_short}"
            else:
                combined_brand_name = brand_ar[:27] + "..."

        rows.append({
            "النوع ": "منتج",
            "أسم المنتج": r.comp_name,
            "تصنيف المنتج": "العطور",
            "صورة المنتج": r.comp_image,
            "وصف صورة المنتج": r.comp_name,
            "نوع المنتج": "منتج جاهز",
            "سعر المنتج": r.comp_price,
            "الوصف": r.generated_product_description, # استخدام الوصف الذي تم توليده بواسطة LLM
            "هل يتطلب شحن؟": "نعم",
            "الوزن": "0.5",
            "وحدة الوزن": "kg",
            "الماركة": combined_brand_name, # استخدام اسم الماركة الموحد
            "العنوان الترويجي": f"{r.comp_name} | {combined_brand_name} - مهووس العطور", # SEO Title
            "تثبيت المنتج": "",
            "الباركود": r.gtin, # استخدام GTIN المستخرج
            "السعرات الحرارية": "",
            "MPN": "",
            "GTIN": r.gtin,
            "خاضع للضريبة ؟": "نعم",
            "سبب عدم الخضوع للضريبة": "",
            "[1] الاسم": "",
            "[1] النوع": "",
            "[1] القيمة": "",
            "[1] الصورة / اللون": "",
            "[2] الاسم": "",
            "[2] النوع": "",
            "[2] القيمة": "",
            "[2] الصورة / اللون": "",
            "[3] الاسم": "",
            "[3] النوع": "",
            "[3] القيمة": "",
            "[3] الصورة / اللون": "",
        })
    df = pd.DataFrame(rows, columns=cols)
    buf = io.StringIO()
    buf.write("بيانات المنتج" + "," * (len(cols) - 1) + "\n")
    df.to_csv(buf, index=False, encoding="utf-8")
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI ORACLE (L4-LLM) — Stub for compatibility with app.py
# ═══════════════════════════════════════════════════════════════════════════

class GeminiOracle:
    """
    تحقق LLM للمنطقة الرمادية (مكرر أم جديد).
    يدعم Google Gemini (حزمة google-genai أو generativeai القديمة) ومفتاح OpenAI.
    """

    _GEMINI_MODELS = ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-preview-05-20")

    def __init__(self, gemini_key: str = "", openai_key: str = ""):
        import os as _os

        self._openai = None
        self._gemini = None
        self._gemini_legacy = False

        gk = (gemini_key or _os.environ.get("GEMINI_API_KEY") or _os.environ.get("GOOGLE_API_KEY") or "").strip()
        ok = (openai_key or _os.environ.get("OPENAI_API_KEY") or "").strip()

        if _GENAI_OK and gk:
            try:
                if hasattr(_google_genai, "Client"):
                    self._gemini = _google_genai.Client(api_key=gk)
                else:
                    _google_genai.configure(api_key=gk)
                    self._gemini_legacy = True
            except Exception as e:
                log.warning("تعذّر تهيئة عميل Gemini: %s", e)

        if ok:
            try:
                self._openai = OpenAI(api_key=ok)
            except Exception as e:
                log.warning("تعذّر تهيئة عميل OpenAI: %s", e)

        if not self._openai:
            env_ok = _os.environ.get("OPENAI_API_KEY", "").strip()
            if env_ok:
                try:
                    self._openai = OpenAI(api_key=env_ok)
                except Exception as e:
                    log.warning("تعذّر تهيئة OpenAI من البيئة: %s", e)

        if not self._gemini and not self._gemini_legacy and _GENAI_OK:
            env_gk = (
                _os.environ.get("GEMINI_API_KEY", "").strip()
                or _os.environ.get("GOOGLE_API_KEY", "").strip()
            )
            if env_gk and hasattr(_google_genai, "Client"):
                try:
                    self._gemini = _google_genai.Client(api_key=env_gk)
                except Exception:
                    pass

    @property
    def has_client(self) -> bool:
        return self._gemini is not None or self._gemini_legacy or self._openai is not None

    def _verify_gemini(self, prompt: str) -> str:
        if self._gemini_legacy:
            try:
                model = _google_genai.GenerativeModel("gemini-1.5-flash")
                r = model.generate_content(prompt)
                return (getattr(r, "text", None) or "").strip().lower()
            except Exception as e:
                log.error("Gemini (legacy) verify failed: %s", e)
                return ""

        last_err = None
        for model in self._GEMINI_MODELS:
            try:
                r = self._gemini.models.generate_content(model=model, contents=prompt)
                text = (getattr(r, "text", None) or "").strip().lower()
                if text:
                    return text
            except Exception as e:
                last_err = e
                continue
        if last_err:
            log.error("Gemini verify failed: %s", last_err)
        return ""

    def _verify_openai(self, prompt: str) -> str:
        response = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=25,
        )
        return (response.choices[0].message.content or "").strip().lower()

    def verify(self, comp_name: str, store_name: str) -> str:
        """يعيد 'duplicate' أو 'new' أو 'review'."""
        if not self.has_client:
            return "review"
        prompt = (
            "Are these the same product? Reply with ONE word only: 'duplicate' or 'new'.\n"
            f"Product A: {comp_name}\n"
            f"Product B: {store_name}"
        )
        for attempt in range(3):
            try:
                answer = ""
                if self._gemini or self._gemini_legacy:
                    answer = self._verify_gemini(prompt)
                if not answer and self._openai:
                    answer = self._verify_openai(prompt)
                if "duplicate" in answer:
                    return "duplicate"
                if "new" in answer:
                    return "new"
                return "review"
            except Exception as e:
                log.error("GeminiOracle محاولة %s فشلت: %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        return "review"


def export_brands_csv(new_brand_results: list[Union[MatchResult, str]]) -> bytes:
    """Export new brands list as a UTF-8 CSV file in Mahwous format."""
    rows = []
    for r in new_brand_results:
        if isinstance(r, str):
            brand_ar, brand_en = r.strip(), ""
            gen_desc = ""
        else:
            brand_ar = r.brand_ar if r.brand_ar else r.comp_brand_raw
            brand_en = r.brand_en if r.brand_en else ""
            gen_desc = r.generated_brand_description

        # بناء اسم الماركة الموحد مع مراعاة حد الـ 30 حرفاً
        combined_brand_name = f"{brand_ar} | {brand_en}"
        if len(combined_brand_name) > 30:
            if len(brand_ar) < 25: # ترك مساحة لـ | و EN
                brand_en_short = brand_en[:(30 - len(brand_ar) - 3)] # -3 for ' | '
                combined_brand_name = f"{brand_ar} | {brand_en_short}"
            else:
                combined_brand_name = brand_ar[:27] + "..." # قص الاسم العربي إذا كان طويلاً جداً
        
        rows.append({
            "اسم الماركة": combined_brand_name,
            "وصف مختصر عن الماركة": gen_desc,
            "صورة شعار الماركة": "", # لا يمكن توليدها حالياً
            "(إختياري) صورة البانر": "",
            "(Page Title) عنوان صفحة العلامة التجارية": f"{brand_ar} | {brand_en} - مهووس العطور",
            "(SEO Page URL) رابط صفحة العلامة التجارية": f"ماركة-{brand_ar.replace(' ', '-')}", # Placeholder slug
            "(Page Description) وصف صفحة العلامة التجارية": gen_desc,
        })
    
    cols = ["اسم الماركة","وصف مختصر عن الماركة","صورة شعار الماركة","(إختياري) صورة البانر","(Page Title) عنوان صفحة العلامة التجارية","(SEO Page URL) رابط صفحة العلامة التجارية","(Page Description) وصف صفحة العلامة التجارية"]
    df = pd.DataFrame(rows, columns=cols)
    buf = io.StringIO()
    buf.write("\ufeff") # Add BOM for UTF-8 compatibility with Excel
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")
